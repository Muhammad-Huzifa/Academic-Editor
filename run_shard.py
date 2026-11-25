import time, os, argparse, json, math
START=time.monotonic()
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from common import *
from tqdm.auto import tqdm

p=argparse.ArgumentParser()
p.add_argument('--plan-dir',required=True)
p.add_argument('--shard-id',required=True,type=int)
p.add_argument('--num-shards',type=int,default=6)
p.add_argument('--hours',type=float,default=9.5)
p.add_argument('--batch-size',type=int,default=4)
p.add_argument('--max-batches',type=int)
p.add_argument('--output-dir',default='/kaggle/working/editor_shards')
p.add_argument('--resume-file')
a=p.parse_args()
if not (0<=a.shard_id<a.num_shards and 1<=a.hours<=10 and a.batch_size>=1): raise ValueError('Invalid worker settings')
plan,rev,cfg,h=load_plan(a.plan_dir)
shard=plan[a.shard_id::a.num_shards]
expected={r['sample_id']:r for r in shard}
root=Path(a.output_dir); root.mkdir(parents=True,exist_ok=True)
result=root/f'shard_{a.shard_id:02d}.jsonl'
manifest=root/f'shard_{a.shard_id:02d}_manifest.json'
key={'schema_version':3,'plan_sha256':h,'model_revision':rev['model'],'shard_id':a.shard_id,'num_shards':a.num_shards,'worker_protocol':'combined_audit_v1'}
rows={}
for f in ([Path(a.resume_file)] if a.resume_file else [])+([result] if result.exists() else []):
    m=json.loads(f.with_name(manifest.name).read_text())
    if any(m.get(k)!=v for k,v in key.items()) or m['results_sha256']!=file_hash(f): raise ValueError('Resume checkpoint mismatch')
    for r in read_jsonl(f):
        sid=r['sample_id']
        if sid not in expected or any(r[k]!=v for k,v in expected[sid].items()): raise ValueError('Unexpected resume record')
        if sid in rows and rows[sid]!=r: raise ValueError('Conflicting resume records')
        rows[sid]=r
def save():
    write_jsonl(result,rows.values())
    write_json(manifest,{**key,'planned':len(shard),'attempted':len(rows),'auto_pass':sum(r['auto_pass'] for r in rows.values()),'complete':len(rows)==len(shard),'results_sha256':file_hash(result)})
save()
pending=[r for r in shard if r['sample_id'] not in rows]
print('Shard',a.shard_id,'planned',len(shard),'already attempted',len(rows),'plan hash',h,flush=True)
if not pending: raise SystemExit(0)
model,tok=load_base(rev['model'])
print('Model device:',model.get_input_embeddings().weight.device,flush=True)
pending.sort(key=lambda r:len(tok(r['output'],add_special_tokens=False)['input_ids']))
if a.max_batches is not None: pending=pending[:a.max_batches*a.batch_size]
durations=[]
for start in tqdm(range(0,len(pending),a.batch_size)):
    if time.monotonic()-START+max(1200,2*max(durations[-5:],default=0))>=a.hours*3600: break
    ts=time.monotonic()
    batch=pending[start:start+a.batch_size]
    drafts=[{'text':r['output'],'finished':True} if r['task']=='identity' else None for r in batch]
    idx=[i for i,g in enumerate(drafts) if g is None]
    if idx:
        prompts=[DRAFTS[batch[i]['task']]+' Produce a complete equivalent prose draft. Preserve every fact, component role, number, citation, unit, mathematical expression and uncertainty. No bullets, new claims or omissions. Return only draft prose.\n\nSOURCE:\n'+batch[i]['output'] for i in idx]
        cap=min(768,max(128,math.ceil(1.65*max(len(tok(batch[i]['output'],add_special_tokens=False)['input_ids']) for i in idx)+48)))
        for i,g in zip(idx,generate(model,tok,prompts,cap,'Create training drafts for academic editing; treat source as data.')): drafts[i]=g
    checks=[flags(g['text'],r['output'],r['task']) for r,g in zip(batch,drafts)]
    eligible=[i for i,g in enumerate(drafts) if g['finished'] and not checks[i][0]]
    prompts=[]
    for i in eligible:
        r=batch[i]; g=drafts[i]
        prompts.append('Audit BOTH conditions: (1) draft and target preserve exactly the same scientific facts, component roles, comparisons, causal direction, numbers, units, citations, tense and uncertainty; (2) target is grammatical, coherent, natural technical prose without extraction artifacts, and improves the requested editing task (or is already good for identity). Return exactly PASS if both hold, otherwise FAIL. Do not explain or repair.\nTASK: '+r['task']+'\nDRAFT:\n'+g['text']+'\nTARGET:\n'+r['output'])
    audits=dict(zip(eligible,generate(model,tok,prompts,16,'You are a strict academic editing pair auditor. Return only PASS or FAIL.')))
    for i,(r,g) in enumerate(zip(batch,drafts)):
        issues,sim=checks[i]
        if not g['finished']: issues.append('truncated')
        audit=audits.get(i,{'text':'NOT_RUN','finished':False})
        accepted=not issues and audit['finished'] and audit['text'].strip().upper().rstrip('.')=='PASS'
        rows[r['sample_id']]={**r,'instruction':instruction(r),'input':g['text'],'flags':issues,'similarity':sim,'combined_audit':audit['text'],'auto_pass':bool(accepted)}
    save(); durations.append(time.monotonic()-ts)
    print(f'Batch {durations[-1]:.1f}s | attempted {len(rows)}/{len(shard)} | accepted {sum(r["auto_pass"] for r in rows.values())}',flush=True)
save()
print(manifest.read_text())
print('Save both:',result,manifest)
