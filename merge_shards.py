import argparse,json,shutil
import pandas as pd
from common import *
p=argparse.ArgumentParser(); p.add_argument('--plan-dir',required=True); p.add_argument('--shards-dir',required=True); p.add_argument('--num-shards',type=int,default=6); p.add_argument('--out',default='/kaggle/working/editor_v3_merged'); p.add_argument('--review-only',action='store_true'); p.add_argument('--require-human-approval',action='store_true'); p.add_argument('--review-file'); a=p.parse_args()
plan,rev,cfg,h=load_plan(a.plan_dir)
rows=[]
for i in range(a.num_shards):
    found=list(Path(a.shards_dir).rglob(f'shard_{i:02d}_manifest.json'))
    if len(found)!=1: raise ValueError(f'Select exactly one latest checkpoint for shard {i}: {found}')
    m=json.loads(found[0].read_text()); f=found[0].with_name(f'shard_{i:02d}.jsonl')
    if m['plan_sha256']!=h or m['model_revision']!=rev['model'] or m['num_shards']!=a.num_shards or m['shard_id']!=i or m.get('worker_protocol')!='combined_audit_v1' or m['results_sha256']!=file_hash(f): raise ValueError('Checkpoint mismatch')
    part=read_jsonl(f); expected={r['sample_id']:r for r in plan[i::a.num_shards]}
    if not m['complete'] or len(part)!=len(expected) or {r['sample_id'] for r in part}!=set(expected): raise ValueError(f'Shard {i} incomplete')
    for r in part:
        if any(r[k]!=v for k,v in expected[r['sample_id']].items()): raise ValueError('Frozen record altered')
    rows.extend(part)
root=Path(a.out); root.mkdir(parents=True,exist_ok=True)
review=root/'review.csv'; prior=Path(a.review_file) if a.review_file else review
decisions={}
if prior.exists():
    old=pd.read_csv(prior,dtype=str,keep_default_na=False)
    decisions=dict(zip(old.sample_id,old.human_decision.str.strip().str.lower()))
df=pd.DataFrame(rows); df['human_decision']=df.sample_id.map(decisions).fillna('')
df.to_csv(review,index=False)
print(pd.crosstab(df.split,df.auto_pass))
if a.review_only:
    print('Review:',review); raise SystemExit(0)
accepted=[]
for r in rows:
    decision=decisions.get(r['sample_id'],'')
    if not r['auto_pass'] or decision=='reject' or (a.require_human_approval and decision!='approve'): continue
    if r.get('license_status') not in ('permitted_metadata','manually_verified'): continue
    accepted.append({**r,'quality_status':'human_approved' if decision=='approve' else 'auto_screened_unreviewed'})
export=root/'dataset_export'; export.mkdir(exist_ok=True)
sets={}
for split in ('train','validation','test'):
    part=[r for r in accepted if r['split']==split]
    if not part: raise ValueError('No accepted records in '+split)
    sets[split]={r['paper_id'] for r in part}
    write_jsonl(export/f'{split}.jsonl',part)
if sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['validation']&sets['test']: raise ValueError('Paper leakage')
if len({r['output'] for r in accepted})!=len(accepted): raise ValueError('Duplicate targets')
write_json(export/'manifest.json',{'schema_version':3,'model_id':MODEL_ID,'model_revision':rev['model'],'dataset_revision':rev['dataset'],'plan_sha256':h,'config':cfg,'counts':dict(Counter(r['split'] for r in accepted)),'quality_counts':dict(Counter(r['quality_status'] for r in accepted)),'files':{s+'.jsonl':file_hash(export/(s+'.jsonl')) for s in sets}})
print('Export:',shutil.make_archive(str(root/'academic_editor_v3_dataset'),'zip',export))
