import argparse,json
from common import *
p=argparse.ArgumentParser(); p.add_argument('--plan-dir',required=True); p.add_argument('--shards-dir',required=True); p.add_argument('--num-shards',type=int,default=6); a=p.parse_args()
plan,rev,cfg,h=load_plan(a.plan_dir)
for i in range(a.num_shards):
    found=list(Path(a.shards_dir).rglob(f'shard_{i:02d}_manifest.json'))
    if len(found)!=1:
        print(i,'missing' if not found else 'ambiguous versions',len(plan[i::a.num_shards])); continue
    m=json.loads(found[0].read_text())
    if m['plan_sha256']!=h or m['num_shards']!=a.num_shards or m['shard_id']!=i: raise ValueError('Wrong plan or shard')
    if m['results_sha256']!=file_hash(found[0].with_name(f'shard_{i:02d}.jsonl')): raise ValueError('Corrupt checkpoint')
    print(i,m['attempted'],'/',len(plan[i::a.num_shards]),'auto-pass',m['auto_pass'],'complete',m['complete'])
