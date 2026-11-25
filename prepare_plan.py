import argparse,json,random,re,math
from collections import Counter,defaultdict
from itertools import islice
from common import *
from huggingface_hub import HfApi
from datasets import load_dataset
from datasketch import MinHash,MinHashLSH
from tqdm.auto import tqdm
p=argparse.ArgumentParser(); p.add_argument('--workspace',default='/kaggle/working/editor_v3_dataset_workspace'); p.add_argument('--pilot',action='store_true'); a=p.parse_args()
root=Path(a.workspace); root.mkdir(parents=True,exist_ok=True)
if (root/'task_plan.jsonl').exists():
    rows,rev,cfg,h=load_plan(root); print('Existing frozen plan:',len(rows),h); raise SystemExit(0)
seed=42; rng=random.Random(seed)
cfg={'schema_version':3,'seed':seed,'papers':24 if a.pilot else 400,'max_tasks':96 if a.pilot else 5000,'min_year':2020,'max_year':2024}
config=root/'config.json'
if config.exists() and json.loads(config.read_text())!=cfg: raise ValueError('Use a new workspace for changed configuration')
write_json(config,cfg)
api=HfApi(); dataset='common-pile/arxiv_papers'
revision_file=root/'revisions.json'
rev=json.loads(revision_file.read_text()) if revision_file.exists() else {'dataset':api.dataset_info(dataset).sha,'model':api.model_info(MODEL_ID).sha}
write_json(revision_file,rev)
topics={'sign_pose_action':r'sign language|skeleton.based|action recognition|human pose|gesture recognition|facial animation|talking head','vision':r'computer vision|vision transformer|object detection|image segmentation|semantic segmentation|image restoration|video understanding|image classification|point cloud|depth estimation','graph':r'graph neural|graph convolution|graph attention','generation':r'diffusion model|generative model|image generation|video generation','multimodal':r'multimodal|multi.modal|audio.visual|vision.language|cross.modal|feature fusion','efficient_ml':r'knowledge distillation|model compression|lightweight network|pruning|quantization|parameter.efficient','other_ml':r'large language model|deep learning|neural network|transformer|representation learning'}
def license_values(x):
    if isinstance(x,dict):
        for k,v in x.items():
            if 'licen' in k.lower() or k.lower()=='rights': yield str(v)
            elif isinstance(v,(dict,list)): yield from license_values(v)
    elif isinstance(x,list):
        for y in x: yield from license_values(y)
def allowed(x):
    t=x.lower().replace('_','-')
    if re.search(r'by[- /]?(nc|nd)|noncommercial|non-commercial|no derivatives|nonexclusive|all rights reserved',t): return False
    return bool(re.search(r'creativecommons\.org/(licenses/by(?:-sa)?/|publicdomain/zero/)|\bcc[ -]?by(?:-sa)?(?:\s|[-0-9]|$)|\bcc0\b|creative commons attribution',t))
def heading(x):
    x=x.strip().lstrip('#').strip().replace('\\.','.')
    return re.sub(r'^(?:\d+(?:\.\d+)*[.)]?\s*|[IVX]+[.)]\s+)','',x).lower().strip(' .:')
patterns=[('abstract',r'^abstract\b'),('introduction',r'^introduction\b|^motivation\b'),('related_work',r'related work|literature review|prior work|background|preliminaries'),('results',r'\bresults?\b|ablation|performance analysis|quantitative|qualitative|comparison'),('experiments',r'experiment|implementation details|evaluation protocol|dataset|training settings'),('discussion',r'discussion|limitations|failure cases'),('conclusion',r'conclusion|future work'),('methodology',r'method|approach|framework|architecture|feature extraction|feature representation|attention|network design|fusion|learning objective|loss function|model formulation|encoder|decoder|proposed model|optimization')]
def sections(text):
    stack=[]; buf=[]; kind=None; title=''
    for line in text.splitlines():
        line=line.strip()
        m=re.match(r'^(#{1,6})\s*(\S.*)$',line)
        if m:
            if kind and buf: yield kind,title,'\n'.join(buf)
            buf=[]; h=heading(m[2]); level=len(m[1]); title=m[2]
            if re.match(r'^(references|bibliography|appendix|supplementary|acknowledg)',h): return
            while stack and stack[-1][0]>=level: stack.pop()
            k=next((s for s,p in patterns if re.search(p,h)),None)
            if level==1 or re.match(r'^(pacs|keywords|index terms|authors|affiliations|funding|copyright)',h): k=None
            elif not k and stack and stack[-1][1] not in (None,'abstract'): k=stack[-1][1]
            stack.append((level,k)); kind=k
        elif re.match(r'^(keywords|index terms|pacs)\s*:',line,re.I) or re.match(r'^(figure|fig\.?|table|algorithm)\s+\d+[a-z]?\s*[:.]',line,re.I) or line.startswith(('|','![','$$',chr(96)*3)):
            if kind and buf: yield kind,title,'\n'.join(buf)
            buf=[]
            if re.match(r'^(keywords|index terms|pacs)',line,re.I): kind=None
        else: buf.append(line)
    if kind and buf: yield kind,title,'\n'.join(buf)
def valid(t,s):
    return 45<=len(t.split())<=(360 if s=='abstract' else 230) and bool(re.search(r'[.!?]["”)\]]?$',t)) and not re.search(r'\b(?:figure|fig\.|table|algorithm)\s+\d+[a-z]?\s*:|\b(?:keywords|index terms|pacs)\s*:|https?://|\\(?:begin|end|frac|label|cite|ref)\b|\uFFFD',t,re.I) and t.count('$')%2==0 and t.count('|')<=3 and len(re.findall(r'[A-Za-z]',t))/max(1,len(t))>.55 and len(re.findall(r'[.!?](?:\s|$)',t))>=2
def passages(text):
    for s,h,body in sections(text):
        for para in ([body] if s=='abstract' else re.split(r'\n\s*\n',body)):
            t=re.sub(r'\s+',' ',para).strip()
            if valid(t,s): yield s,h,t; continue
            if s=='abstract' or len(t.split())<=230: continue
            buf=[]
            for sent in re.split(r'(?<=[.!?])\s+(?=[A-Z])',t):
                if len(sent.split())>210:
                    old=' '.join(buf)
                    if valid(old,s): yield s,h,old
                    buf=[]; continue
                if len(' '.join(buf+[sent]).split())>210:
                    old=' '.join(buf)
                    if valid(old,s): yield s,h,old
                    buf=[]
                buf.append(sent)
            old=' '.join(buf)
            if valid(old,s): yield s,h,old
paperfile=root/'papers.jsonl'; papers=read_jsonl(paperfile) if paperfile.exists() else []
seen={r['paper_id'] for r in papers}; counts=Counter(r['topic'] for r in papers)
overrides=root/'license_overrides.json'
if not overrides.exists(): write_json(overrides,{})
override=json.loads(overrides.read_text())
files=sorted([f for f in api.list_repo_files(dataset,repo_type='dataset',revision=rev['dataset']) if f.endswith('.jsonl.gz')],reverse=True)
if not files: raise ValueError('No data files')
per_file=max(12,math.ceil(cfg['papers']/min(len(files),6))); file_counts=Counter(r['shard'] for r in papers)
for f in files[:16]:
    if len(papers)>=cfg['papers']: break
    if file_counts[f]>=per_file: continue
    stream=load_dataset(dataset,revision=rev['dataset'],data_files=f,split='train',streaming=True)
    for sample in tqdm(islice(stream,18000),desc=f):
        m=re.search(r'\d{4}\.\d{4,5}',str(sample.get('id','')))
        if not m: continue
        pid=m.group(); year=2000+int(pid[:2])
        if pid in seen or not cfg['min_year']<=year<=cfg['max_year']: continue
        text=sample.get('text') or ''
        if len(text.split())<1200: continue
        meta=sample.get('metadata') or {}
        if isinstance(meta,str):
            try: meta=json.loads(meta)
            except ValueError: meta={}
        title=str(meta.get('title') or next((l.strip()[2:] for l in text.splitlines()[:45] if l.strip().startswith('# ')),''))
        topic=next((k for k,p in topics.items() if re.search(p,title,re.I)),None)
        if not topic or counts[topic]>=max(4,math.ceil(cfg['papers']*.35)): continue
        license_=str(override.get(pid) or ' | '.join(license_values(meta)))
        if not allowed(license_): continue
        papers.append({'paper_id':pid,'year':year,'title':title,'topic':topic,'license':license_,'license_status':'manually_verified' if pid in override else 'permitted_metadata','authors':str(meta.get('authors','')),'source_url':'https://arxiv.org/abs/'+pid,'shard':f,'text':text})
        seen.add(pid); counts[topic]+=1; file_counts[f]+=1
        if len(papers)%10==0: write_jsonl(paperfile,papers)
        if len(papers)>=cfg['papers'] or file_counts[f]>=per_file: break
    write_jsonl(paperfile,papers)
if len(papers)<12: raise ValueError('Too few licensed papers; inspect source metadata, verify licenses before adding overrides.')
print('Collected',len(papers),counts)
lsh=MinHashLSH(threshold=.85,num_perm=64); exact=set(); unique=[]
for paper in tqdm(papers,desc='Extract and deduplicate'):
    group=defaultdict(list)
    for s,h,t in passages(paper['text']): group[s].append((h,t))
    local=[]
    for s,items in sorted(group.items()):
        rng.shuffle(items); local.extend((s,h,t) for h,t in items[:(1 if s=='abstract' else 5)])
    rng.shuffle(local)
    for s,h,t in local[:20]:
        words=re.findall(r'\w+',t.lower()); key=' '.join(words)
        if key in exact: continue
        mh=MinHash(num_perm=64,seed=seed)
        for k in sorted({' '.join(words[i:i+5]) for i in range(len(words)-4)}): mh.update(k.encode())
        if lsh.query(mh): continue
        pid=digest(paper['paper_id']+'|'+t); lsh.insert(pid,mh); exact.add(key)
        unique.append({**{k:v for k,v in paper.items() if k!='text'},'section':s,'heading':h,'output':t,'passage_id':pid})
ids=sorted({r['paper_id'] for r in unique}); rng.shuffle(ids)
if len(ids)<12 or len(unique)<60: raise ValueError('Too few usable documents/passages')
n=len(ids); mapping={x:('train' if i<int(.8*n) else 'validation' if i<int(.9*n) else 'test') for i,x in enumerate(ids)}
mix=['refine']*8+['coherence']*4+['restructure']*3+['redundancy']*2+['grammar']*2+['identity']
plan=[]
for split,fraction in [('train',.8),('validation',.1),('test',.1)]:
    items=[r for r in unique if mapping[r['paper_id']]==split]; rng.shuffle(items)
    for i,r in enumerate(items[:int(cfg['max_tasks']*fraction)]):
        task=mix[i%len(mix)]
        plan.append({**r,'split':split,'task':task,'sample_id':digest(r['passage_id']+'|'+task)})
rng.shuffle(plan)
write_json(root/'paper_splits.json',mapping); write_jsonl(root/'task_plan.jsonl',plan)
print('Frozen plan:',len(plan),Counter(r['split'] for r in plan),Counter(r['section'] for r in plan),file_hash(root/'task_plan.jsonl'))
