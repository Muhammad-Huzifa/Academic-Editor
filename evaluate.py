import argparse,os,json
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
from common import *
import pandas as pd
from peft import PeftModel
from tqdm.auto import tqdm
p=argparse.ArgumentParser(); p.add_argument('--dataset-dir',required=True); p.add_argument('--run-dir',default='/kaggle/working/academic_editor_training_v3'); p.add_argument('--resume-from'); p.add_argument('--out',default='/kaggle/working/editor_evaluation'); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--require-human-approval',action='store_true'); a=p.parse_args()
run=Path(a.resume_from or a.run_dir); data=Path(a.dataset_dir); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
if json.loads((run/'session_status.json').read_text())['status']!='training_complete': raise ValueError('Finish training first')
m=json.loads((data/'manifest.json').read_text()); cfg=json.loads((run/'training_config.json').read_text())
if cfg['files']!=m['files'] or cfg['revision']!=m['model_revision'] or cfg['human_approval']!=a.require_human_approval: raise ValueError('Run/dataset mismatch')
if file_hash(data/'test.jsonl')!=m['files']['test.jsonl']: raise ValueError('Test hash mismatch')
rows=read_jsonl(data/'test.jsonl')
if a.require_human_approval: rows=[r for r in rows if r['quality_status']=='human_approved']
base,tok=load_base(m['model_revision']); model=PeftModel.from_pretrained(base,run/'best_lora_adapter'); model.eval()
result=out/'comparison.jsonl'; done={}
signature={'test':m['files']['test.jsonl'],'adapter':file_hash(run/'best_lora_adapter'/'adapter_model.safetensors'),'approval':a.require_human_approval,'max_new_tokens':768}
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
    with model.disable_adapter(): before=generate(model,tok,prompts,768)
    after=generate(model,tok,prompts,768)
    for r,b,c in zip(batch,before,after):
        row={'sample_id':r['sample_id'],'section':r['section'],'task':r['task'],'input':r['input'],'reference':r['output'],'baseline':b['text'],'finetuned':c['text']}
        for name,g in [('baseline',b),('finetuned',c)]:
            row[name+'_finished']=g['finished']
            row[name+'_reference_f1']=f1(g['text'],r['output'])
            row[name+'_input_similarity']=similarity(g['text'],r['input'])
            row[name+'_number_match']=numbers(g['text'])==numbers(r['input'])
        done[r['sample_id']]=row
    write_jsonl(result,done.values())
df=pd.DataFrame(done.values()); df.to_csv(out/'comparison.csv',index=False)
print(df.select_dtypes(include=['number','bool']).mean().to_string())
rng=random.Random(42); blind=[]; key=[]
for r in done.values():
    methods=['baseline','finetuned']; rng.shuffle(methods)
    blind.append({'sample_id':r['sample_id'],'input':r['input'],'A':r[methods[0]],'B':r[methods[1]],'preferred':'','scientific_errors_A':'','scientific_errors_B':''})
    key.append({'sample_id':r['sample_id'],'A':methods[0],'B':methods[1]})
pd.DataFrame(blind).to_csv(out/'blind_review.csv',index=False); write_json(out/'blind_key.json',key)
print('Lexical/numerical metrics are diagnostics, not proof of semantic accuracy. Review outputs blind.')
