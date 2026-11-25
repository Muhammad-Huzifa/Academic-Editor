import argparse, shutil
from pathlib import Path
from common import load_plan
p=argparse.ArgumentParser()
p.add_argument('--workspace',required=True)
p.add_argument('--out',default='/kaggle/working/shared_plan')
a=p.parse_args()
src=Path(a.workspace); dst=Path(a.out)
rows,rev,cfg,h=load_plan(src)
dst.mkdir(parents=True,exist_ok=True)
for name in ['task_plan.jsonl','paper_splits.json','config.json','revisions.json']:
    if (src/name).resolve()!=(dst/name).resolve(): shutil.copy2(src/name,dst/name)
print('Frozen tasks:',len(rows),'SHA256:',h)
print('Export:',shutil.make_archive(str(dst),'zip',dst))
