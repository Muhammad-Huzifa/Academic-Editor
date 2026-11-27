import os,time,argparse,math,shutil,json
START=time.monotonic()
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
from transformers import Trainer,TrainingArguments,TrainerCallback,EarlyStoppingCallback,set_seed
from transformers.trainer_utils import get_last_checkpoint
p=argparse.ArgumentParser()
p.add_argument('--github-repo', help='Academic-Editor repository for checkpoint release assets'); p.add_argument('--github-tag', default='academic-editor-v4-pilot-sft');
p.add_argument('--allow-public',action='store_true',help='Explicitly permit public checkpoint publication');
p.add_argument('--dataset-dir',required=True); p.add_argument('--run-dir',default='/kaggle/working/academic_editor_training_v4'); p.add_argument('--resume-from'); p.add_argument('--epochs',type=int,default=8); p.add_argument('--hours',type=float,default=9.5); p.add_argument('--approve-pilot',action='store_true',help='Confirm review and split audit completed'); a=p.parse_args()
if not a.approve_pilot: p.error('Review the frozen pilot and confirm --approve-pilot before training')
if a.github_repo and not a.github_tag.startswith('academic-editor-v4-'): p.error('Use a unique V4 release tag')
if 'v3' in Path(a.run_dir).name.lower(): p.error('Never use a V3 run directory for V4')
if not 1<=a.hours<=10 or a.epochs<1: raise ValueError('Invalid training budget')
run=Path(a.run_dir); run.mkdir(parents=True,exist_ok=True)
if a.resume_from and Path(a.resume_from).resolve()!=run.resolve():
    if any(run.iterdir()): raise ValueError('Resume into an empty run directory')
    shutil.copytree(a.resume_from,run,dirs_exist_ok=True)
if (run/'session_status.json').exists() and json.loads((run/'session_status.json').read_text()).get('status')=='training_complete':
    raise SystemExit('Training is already complete; evaluate the saved adapter instead.')
data=Path(a.dataset_dir); m=json.loads((data/'manifest.json').read_text())
if m['schema_version']!=4 or m['model_id']!=MODEL_ID or m.get('human_review_required') is not True:
    raise ValueError('Expected a reviewed V4 dataset')
if a.allow_public and m.get('public_checkpoint_ok') is not True:
    raise ValueError('Public checkpoint permission is not established; keep all checkpoints private.')
backup=None
parts={}
for s in ('train','validation','test'):
    f=data/(s+'.jsonl')
    if m['files'][f.name]!=file_hash(f): raise ValueError('Dataset hash mismatch')
    rows=read_jsonl(f)
    if any(r['split']!=s or not r['auto_pass'] or r['license_status'] not in ('permitted_metadata','manually_verified') for r in rows): raise ValueError('Invalid dataset record')
    if any(r.get('quality_status')!='human_approved' or r.get('license_status')!='manually_verified' for r in rows): raise ValueError('Unreviewed record')
    if a.allow_public and any(r.get('public_checkpoint_ok') is not True for r in rows): raise ValueError('Public checkpoint permission missing')
    if not rows: raise ValueError('Empty '+s)
    parts[s]=rows
ids=[{r['paper_id'] for r in parts[s]} for s in parts]
if any(ids[i]&ids[j] for i in range(3) for j in range(i)): raise ValueError('Paper leakage')
allrows=sum(parts.values(),[])
if len({r['output'] for r in allrows})!=len(allrows): raise ValueError('Duplicate targets')
cfg={'epochs':a.epochs,'files':m['files'],'revision':m['model_revision'],'human_approval':True,'schema_version':4,'script_sha256':file_hash(Path(__file__)),'common_sha256':file_hash(Path(__file__).resolve().parents[1]/'common.py'),'seed':42,'lr':2e-5,'rank':16,'alpha':32,'dropout':.05,'max_length':4096,'gradient_accumulation':8}
cf=run/'training_config.json'
if cf.exists() and json.loads(cf.read_text())!=cfg: raise ValueError('Cannot change configuration when resuming')
write_json(cf,cfg)
if a.github_repo:
    from github_backup import GitHubBackup
    backup=GitHubBackup(a.github_repo,a.github_tag,allow_public=a.allow_public)
    backup.check()
    print('GitHub checkpoint backup enabled',flush=True)
set_seed(42)
model,tok=load_base(m['model_revision']); model.config.use_cache=False
model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules='all-linear',bias='none',task_type='CAUSAL_LM'))
model.print_trainable_parameters()
def encode(r):
    msgs=[{'role':'system','content':SYSTEM},{'role':'user','content':r['instruction']+'\n\nInput:\n'+r['input']}]
    prompt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    full=tok.apply_chat_template(msgs+[{'role':'assistant','content':r['output']}],tokenize=False,add_generation_prompt=False)
    p=tok(prompt,add_special_tokens=False)['input_ids']; f=tok(full,add_special_tokens=False)['input_ids']
    if f[:len(p)]!=p or len(f)>4096 or len(f)-len(p)<2: return None
    return {'input_ids':f,'attention_mask':[1]*len(f),'labels':[-100]*len(p)+f[len(p):]}
def prep(rows):
    out=[x for r in rows if (x:=encode(r)) is not None]
    print('Retained',len(out),'of',len(rows))
    if len(out)!=len(rows): raise ValueError('Tokenization/prompt boundary failed or example exceeds 4096 tokens. Audit and rebuild the dataset; do not silently discard rows.')
    return Dataset.from_list(out)
tr=prep(parts['train']); va=prep(parts['validation'])
def collate(batch):
    n=((max(len(r['input_ids']) for r in batch)+7)//8)*8
    return {k:torch.tensor([r[k]+[v]*(n-len(r[k])) for r in batch],dtype=torch.long) for k,v in [('input_ids',tok.pad_token_id),('attention_mask',0),('labels',-100)]}
interval=max(1,min(105,math.ceil(math.ceil(len(tr)/8)/2)))
ckpt=run/'checkpoints'; ckpt.mkdir(exist_ok=True)
bf16=torch.cuda.is_bf16_supported()
args=TrainingArguments(output_dir=str(ckpt),num_train_epochs=a.epochs,per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=8,learning_rate=2e-5,weight_decay=.01,warmup_ratio=.05,lr_scheduler_type='cosine',optim='paged_adamw_8bit',max_grad_norm=1.,fp16=not bf16,bf16=bf16,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},eval_strategy='steps',eval_steps=interval,save_strategy='steps',save_steps=interval,load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,save_total_limit=3,save_only_model=False,prediction_loss_only=True,remove_unused_columns=False,label_names=['labels'],report_to='none',logging_steps=min(10,interval),seed=42,data_seed=42,restore_callback_states_from_checkpoint=True)
class Budget(TrainerCallback):
    paused=False
    def on_step_end(self,args,state,control,**kw):
        if time.monotonic()-START>=a.hours*3600-2700:
            self.paused=True; control.should_save=True; control.should_training_stop=True
        return control
budget=Budget()
class RemoteBackup(TrainerCallback):
    def on_save(self,args,state,control,**kwargs):
        if backup is not None and state.is_world_process_zero:
            try:
                backup.save(run)
            except Exception as exc:
                print('WARNING: remote backup failed ('+type(exc).__name__+'); pausing at the saved local checkpoint.',flush=True)
                budget.paused=True
                control.should_training_stop=True
        return control
trainer=Trainer(model=model,args=args,train_dataset=tr,eval_dataset=va,data_collator=collate,processing_class=tok,callbacks=[EarlyStoppingCallback(early_stopping_patience=4,early_stopping_threshold=.005),budget,RemoteBackup()])
resume=get_last_checkpoint(str(ckpt))
if resume:
    for f in ('trainer_state.json','optimizer.pt','scheduler.pt'):
        if not (Path(resume)/f).exists(): raise ValueError('Incomplete checkpoint: '+f)
trainer.train(resume_from_checkpoint=resume)
pd.DataFrame(trainer.state.log_history).to_csv(run/'training_history.csv',index=False)
if budget.paused:
    write_json(run/'session_status.json',{'status':'paused','step':trainer.state.global_step})
    if backup is not None:
        try: backup.save(run)
        except Exception as exc: print('WARNING: session-end remote backup failed ('+type(exc).__name__+'); save the local run directory.',flush=True)
    print('Training paused. Verify the latest remote checkpoint or save the full local run directory before closing Kaggle.'); raise SystemExit(0)
best=run/'best_lora_adapter'; trainer.save_model(str(best)); tok.save_pretrained(best)
write_json(run/'training_summary.json',{'best_eval_loss':trainer.state.best_metric,'best_checkpoint':trainer.state.best_model_checkpoint,'epoch':trainer.state.epoch,'steps':trainer.state.global_step})
shutil.make_archive(str(run/'best_adapter'),'zip',best)
write_json(run/'session_status.json',{'status':'training_complete'})
if backup is not None:
    try: backup.save(run,final=True)
    except Exception as exc: print('WARNING: final remote backup failed ('+type(exc).__name__+'); save the local run directory.',flush=True)
print('Training complete:',best)
