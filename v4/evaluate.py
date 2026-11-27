import argparse,os,json
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import pandas as pd
from peft import PeftModel
from tqdm.auto import tqdm
p=argparse.ArgumentParser(); p.add_argument('--dataset-dir',required=True); p.add_argument('--run-dir',default='/kaggle/working/academic_editor_training_v4'); p.add_argument('--resume-from'); p.add_argument('--out',default='/kaggle/working/editor_v4_evaluation'); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--v3-adapter',help='Optional path to V3 best_lora_adapter; compare on identical examples'); a=p.parse_args()
run=Path(a.resume_from or a.run_dir); data=Path(a.dataset_dir); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
if json.loads((run/'session_status.json').read_text())['status']!='training_complete': raise ValueError('Finish training first')
m=json.loads((data/'manifest.json').read_text()); cfg=json.loads((run/'training_config.json').read_text())
if m['schema_version']!=4 or cfg.get('schema_version')!=4 or cfg['files']!=m['files'] or cfg['revision']!=m['model_revision'] or cfg['human_approval'] is not True: raise ValueError('Run/dataset mismatch')
if file_hash(data/'test.jsonl')!=m['files']['test.jsonl']: raise ValueError('Test hash mismatch')
rows=read_jsonl(data/'test.jsonl')
if any(r['quality_status']!='human_approved' for r in rows): raise ValueError('Unreviewed test example')
base,tok=load_base(m['model_revision']); model=PeftModel.from_pretrained(base,run/'best_lora_adapter'); model.eval()
if a.v3_adapter:
    model.load_adapter(a.v3_adapter,adapter_name="v3")
model.set_adapter("default")
result=out/'comparison.jsonl'; done={}
signature={'test':m['files']['test.jsonl'],'adapter':file_hash(run/'best_lora_adapter'/'adapter_model.safetensors'),'approval':True,'v3':file_hash(Path(a.v3_adapter)/'adapter_model.safetensors') if a.v3_adapter else None,'max_new_tokens':1536}
meta=out/'evaluation_config.json'
if meta.exists() and json.loads(meta.read_text())!=signature: raise ValueError('Evaluation configuration changed')
write_json(meta,signature)
if result.exists(): done={r['sample_id']:r for r in read_jsonl(result)}
def words(s): return re.findall(r'\w+',s.lower())
def similarity(a,b): return SequenceMatcher(None,words(a),words(b),autojunk=False).ratio()
def f1(a,b):
    x,y=Counter(words(a)),Counter(words(b)); return 2*sum((x&y).values())/max(1,sum(x.values())+sum(y.values()))
pending=[r for r in rows if r['sample_id'] not in done]
for start in tqdm(range(0,len(pending),a.batch_size)):
    batch=pending[start:start+a.batch_size]
    prompts=[r['instruction']+'\n\nInput:\n'+r['input'] for r in batch]
    model.set_adapter('default')
    with model.disable_adapter(): before=generate(model,tok,prompts,1536)
    after=generate(model,tok,prompts,1536)
    if a.v3_adapter:
        model.set_adapter('v3'); old=generate(model,tok,prompts,1536); model.set_adapter('default')
    else: old=[None]*len(batch)
    for r,b,c,v3 in zip(batch,before,after,old):
        row={'sample_id':r['sample_id'],'section':r['section'],'task':r['task'],'input':r['input'],'reference':r['output'],'baseline':b['text'],'finetuned':c['text']}
        row['copy']=r['input']
        candidates=[('copy',{'text':r['input'],'finished':True}),('baseline',b),('finetuned',c)]
        if v3 is not None:
            row['v3']=v3['text']; candidates.append(('v3',v3))
        for name,g in candidates:
            row[name+'_finished']=g['finished']
            row[name+'_reference_f1']=f1(g['text'],r['output'])
            row[name+'_input_similarity']=similarity(g['text'],r['input'])
            row[name+'_number_match']=numbers(g['text'])==numbers(r['input'])
            row[name+'_citation_match']=citations(g['text'])==citations(r['input'])
            row[name+'_exact_copy']=g['text'].strip()==r['input'].strip()
        done[r['sample_id']]=row
    write_jsonl(result,done.values())
df=pd.DataFrame(done.values()); df.to_csv(out/'comparison.csv',index=False)
print(df.select_dtypes(include=['number','bool']).mean().to_string())
df.groupby('task')[df.select_dtypes(include=['number','bool']).columns.tolist()].mean().to_csv(out/'task_metrics.csv')
rng=random.Random(42); blind=[]; key=[]
for r in done.values():
    methods=['copy','baseline','finetuned']+(['v3'] if 'v3' in r else []); rng.shuffle(methods)
    labels=dict(zip('ABCD',methods))
    for label,method in labels.items():
        blind.append({'sample_id':r['sample_id'],'task':r['task'],'input':r['input'],'candidate':label,'output':r[method],'critical_scientific_error':'','technical_clarity_1_5':'','coherence_1_5':'','meaningful_improvement_1_5':'','preferred':'','review_note':''})
    key.append({'sample_id':r['sample_id'],**labels})
pd.DataFrame(blind).to_csv(out/'blind_review.csv',index=False); write_json(out/'blind_key.json',key)
print('Lexical/numerical metrics are diagnostics, not proof of semantic accuracy. Review outputs blind.')
