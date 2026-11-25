import os, json, hashlib, re, gc, time, random, unicodedata
from pathlib import Path
from collections import Counter
from decimal import Decimal
from difflib import SequenceMatcher

MODEL_ID = 'Qwen/Qwen3-4B-Instruct-2507'
SYSTEM = ('You are a technical academic editor. Improve the author’s prose, not the science. '
          'Preserve all scientific facts, component roles, numbers, units, citations, qualifiers '
          'and uncertainty. Never invent results, mechanisms or references. Return only revised prose.')
TASKS = {
 'refine':'Refine the draft into natural, precise technical academic prose.',
 'coherence':'Improve logical progression and transitions while preserving technical relationships.',
 'restructure':'Restructure awkward sentences without changing scientific content.',
 'redundancy':'Remove repeated ideas without removing any distinct scientific information.',
 'grammar':'Correct grammar, terminology usage and punctuation without changing claims.',
 'identity':'Edit only where needed; return clear and correct text unchanged.'
}
DRAFTS = {
 'refine':'Use less polished, simpler prose and slightly awkward phrasing.',
 'coherence':'Use weaker transitions, while retaining all dependencies and causal relationships.',
 'restructure':'Introduce awkward sentence boundaries or repetitive syntax.',
 'redundancy':'Repeat some existing ideas using prose, without adding facts.',
 'grammar':'Introduce a few realistic grammar or punctuation errors; retain technical terminology.'
}
def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def digest(s): return hashlib.sha256(s.encode()).hexdigest()
def file_hash(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8'); os.replace(q,p)
def write_jsonl(p,rows):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp')
    with q.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    os.replace(q,p)
def load_plan(directory):
    d=Path(directory); rows=read_jsonl(d/'task_plan.jsonl')
    splits=json.loads((d/'paper_splits.json').read_text())
    rev=json.loads((d/'revisions.json').read_text())
    cfg=json.loads((d/'config.json').read_text())
    if cfg['schema_version']!=3: raise ValueError('Expected V3 plan')
    if len(rows)!=len({r['sample_id'] for r in rows}) or any(splits[r['paper_id']]!=r['split'] for r in rows):
        raise ValueError('Invalid frozen plan')
    return rows,rev,cfg,file_hash(d/'task_plan.jsonl')
def instruction(r):
    return 'Section: '+r['section'].replace('_',' ')+'. '+TASKS[r['task']]+' Preserve scientific meaning, values, units, citations and uncertainty. Return only revised prose.'
def numbers(t):
    t=unicodedata.normalize('NFKC',t).replace('−','-')
    t=re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',t)
    return {str(Decimal(x).normalize()) for x in re.findall(r'(?<![\w.])[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?',t)}
def citations(t): return {re.sub(r'\s+','',x) for x in re.findall(r'\[\s*\d+(?:\s*[,;–-]\s*\d+)*\s*\]|\([^()]{0,160}\b(?:19|20)\d{2}[a-z]?[^()]{0,80}\)',t)}
def math_tokens(t): return {re.sub(r'\s+','',x) for x in re.findall(r'(?<!\$)\$[^$\n]+\$(?!\$)|\\\([^\n]+?\\\)',t)}
def flags(source,target,task):
    f=[]
    if numbers(source)!=numbers(target): f.append('numeric_mismatch')
    if citations(source)!=citations(target): f.append('citation_mismatch')
    if math_tokens(source)!=math_tokens(target): f.append('math_mismatch')
    sw,tw=len(source.split()),len(target.split())
    sim=SequenceMatcher(None,source.lower().split(),target.lower().split(),autojunk=False).ratio()
    if sw<max(20,.6*tw) or sw>2.1*tw: f.append('length')
    if task!='identity' and source==target: f.append('no_edit')
    if task not in ('identity','grammar') and sim>.96: f.append('near_copy')
    if task=='identity' and source!=target: f.append('identity_changed')
    if re.search(r'^\s*(?:[-*•]\s|\d+\.\s)',source,re.M): f.append('not_prose')
    if re.search(r'^(here is|certainly|sure[,!]|revised passage)',source,re.I): f.append('preamble')
    return f,sim
def load_base(revision):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    if not torch.cuda.is_available(): raise RuntimeError('Enable GPU')
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=revision)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.padding_side='left'
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=revision,quantization_config=quant,device_map={'':0},torch_dtype=dtype,attn_implementation='sdpa')
    model.eval(); model.config.use_cache=True
    return model,tok
def generate(model,tok,prompts,cap,system=SYSTEM):
    if not prompts: return []
    import torch
    tok.padding_side='left'
    texts=[tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':p}],tokenize=False,add_generation_prompt=True) for p in prompts]
    enc=tok(texts,padding=True,return_tensors='pt',add_special_tokens=False).to(model.get_input_embeddings().weight.device)
    if enc.input_ids.shape[1]+cap>8192: raise ValueError('Prompt too long')
    try:
        with torch.inference_mode():
            out=model.generate(**enc,max_new_tokens=cap,do_sample=False,pad_token_id=tok.pad_token_id,use_cache=True)
    except torch.cuda.OutOfMemoryError:
        if len(prompts)==1: raise
        out=None
    if out is None:
        del enc
        gc.collect(); torch.cuda.empty_cache()
        mid=len(prompts)//2
        return generate(model,tok,prompts[:mid],cap,system)+generate(model,tok,prompts[mid:],cap,system)
    tokens=out[:,enc.input_ids.shape[1]:].cpu()
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,(list,tuple)) else [eos])-{None}
    return [{'text':s.strip(),'finished':bool(eos.intersection(t.tolist()))} for s,t in zip(tok.batch_decode(tokens,skip_special_tokens=True),tokens)]
