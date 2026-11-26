# Academic Editor — Technical Research Writing

Qwen3-4B-Instruct-2507 + QLoRA. Refine author-supplied abstracts, methodology and results without modifying scientific facts, component roles, numbers, citations or uncertainty. The goal is precise and natural scientific prose, not AI-detector evasion.

## Project workflow

`prepare_plan.py` → frozen plan → `run_shard.py` (6 independent partitions) → `merge_shards.py` → `train.py` → `evaluate.py`.

The repository contains source code; make it private before enabling same-repository checkpoint Release backups. Keep unpublished research, datasets and credentials out of Git history. Use authorized compute quotas. Six logical shards support reproducibility and resumability, not quota circumvention.

## 0. Setup (every new Kaggle notebook)

Enable Internet and GPU for generation, training and evaluation. CPU is sufficient for export and merge.

```python
!git clone https://github.com/Muhammad-Huzifa/Academic-Editor.git /kaggle/working/Academic-Editor
%cd /kaggle/working/Academic-Editor
!pip install -q -r requirements.txt
```

Use the same Git commit for all workers.

## 1. Prepare the plan once

If the original V3 notebook has already prepared a plan, interrupt its old generation loop and use:

```python
!find /kaggle/working -name task_plan.jsonl
!python -u export_plan.py --workspace /kaggle/working/editor_v3_dataset_workspace
```

Use the actual workspace found by find. If the original plan was lost, create a new one:

```python
!python -u prepare_plan.py
!python -u export_plan.py --workspace /kaggle/working/editor_v3_dataset_workspace
```

The output /kaggle/working/shared_plan.zip contains four files. Extract and upload them as a private Kaggle dataset, then attach it to the worker notebooks. Use !find /kaggle/input -name task_plan.jsonl to obtain the exact path.

## 2. Benchmark and generate one shard (GPU)

```python
!python -u run_shard.py --plan-dir /kaggle/input/academic-editor-plan --shard-id 0 --max-batches 3
```

Replace the path with the actual attached directory. Review seconds per batch and acceptance counts. Rerun without --max-batches to continue. Change --shard-id to 0, 1, 2, 3, 4 or 5. Keep the same plan, commit, shard count and worker protocol. Workers generate a draft and perform one short combined semantic/writing audit; they do not retry rejected pairs. This is faster in principle but may yield fewer accepted pairs and has not been benchmarked.

The worker uses one GPU and a conservative 9.5-hour wall-clock budget. It writes progress after each batch. Save both files from /kaggle/working/editor_shards before the session ends. For shard 0 they are shard_00.jsonl and shard_00_manifest.json.

In a fresh session, attach the previous output files and original plan:

```python
!python -u run_shard.py --plan-dir /kaggle/input/academic-editor-plan --shard-id 0 --resume-file /kaggle/input/academic-editor-shard-0/shard_00.jsonl
```

Keep the manifest next to the JSONL. Do not mix checkpoints from the earlier V3 notebooks with this worker. Each file is written atomically, but a forced shutdown between the two writes can produce a hash mismatch; keep the last consistent saved pair.

## 3. Merge (CPU)

Attach exactly one latest checkpoint pair per shard and the frozen plan.

```python
!python status.py --plan-dir /kaggle/input/academic-editor-plan --shards-dir /kaggle/input
!python merge_shards.py --plan-dir /kaggle/input/academic-editor-plan --shards-dir /kaggle/input --review-only
```

Complete means all planned attempts, not all accepted examples. Review /kaggle/working/editor_v3_merged/review.csv, verify source permissions, and mark human_decision as approve or reject. Export:

```python
!python merge_shards.py --plan-dir /kaggle/input/academic-editor-plan --shards-dir /kaggle/input --require-human-approval
```

For a preliminary automatic-screened experiment only, omit --require-human-approval after license review. Exported dataset: /kaggle/working/editor_v3_merged/academic_editor_v3_dataset.zip. Extract and upload the four files privately.

## 4. Fine-tune (GPU)

```python
!python -u train.py --dataset-dir /kaggle/input/academic-editor-dataset --epochs 20 --require-human-approval
```

20 is a ceiling, not a target. Response-only SFT, LoRA rank 16, learning rate 5e-5, cosine scheduler, effective batch size 8, best validation checkpoint and early stopping.

Save the full /kaggle/working/academic_editor_training_v3 directory, including optimizer and scheduler checkpoints. To resume, attach this directory and add --resume-from /kaggle/input/PREVIOUS_RUN/academic_editor_training_v3. Keep the same flags and default run path. Only evaluate once session_status.json reports training_complete.

## 5. Evaluate (GPU)

```python
!python -u evaluate.py --dataset-dir /kaggle/input/academic-editor-dataset --resume-from /kaggle/input/FINISHED_RUN/academic_editor_training_v3 --require-human-approval
```

Use matching approval flags. Compare original and adapted models with identical prompts. Outputs include reference overlap, copying diagnostics, numeric preservation and a blind review CSV. Save the evaluation directory if interrupted; attach/copy it to the same output location before resuming. Test independent author-written drafts as well.

## Limitations

Automatic audits use the same teacher model and may miss scientific errors. Human review and license verification remain necessary. Common Pile arXiv is a collection of openly licensed texts, not a guarantee of peer review or writing quality. Counts are targets. Full GPU execution and throughput remain unverified. The editor cannot establish human authorship.


## Remote checkpoint backup — existing repository

Use this same repository, **Muhammad-Huzifa/Academic-Editor**, for code and checkpoint Release assets. No second repository is required. Before backup, change repository visibility to Private in GitHub Settings > General > Danger Zone. The script refuses public repositories. Grant the Kaggle GITHUB_TOKEN Contents read/write access to this repository. Never print or commit the token.

In Kaggle, enable GPU and Internet, then load the secret and clone or update the repository with transient authentication:

```python
import os, base64, subprocess
from pathlib import Path
from kaggle_secrets import UserSecretsClient

os.environ["GITHUB_TOKEN"] = UserSecretsClient().get_secret("GITHUB_TOKEN")
BACKUP_REPO = "Muhammad-Huzifa/Academic-Editor"
RUN_TAG = "academic-editor-v3-final-training"
REPO_DIR = Path("/kaggle/working/Academic-Editor")

git_env = os.environ.copy()
git_env.update({
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
    "GIT_CONFIG_VALUE_0": "Authorization: Basic " + base64.b64encode(
        ("x-access-token:" + os.environ["GITHUB_TOKEN"]).encode()
    ).decode(),
    "GIT_TERMINAL_PROMPT": "0",
})
if (REPO_DIR / ".git").exists():
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], env=git_env, check=True)
else:
    subprocess.run(["git", "clone", "https://github.com/" + BACKUP_REPO + ".git", str(REPO_DIR)], env=git_env, check=True)
del git_env
os.chdir(REPO_DIR)
```

```python
!pip install -q -r requirements.txt gdown
!python github_backup.py check --repo "{BACKUP_REPO}" --tag "{RUN_TAG}"
```

The check creates a release if needed, uploads a small test file, downloads and checks its contents, and deletes only that test asset. It does not upload model weights. After it succeeds, prepare DATASET_DIR using the original four final dataset files and verify CUDA in PyTorch.

Fresh training (only if no prior checkpoint for this run):

```python
!python -u train.py --dataset-dir "{DATASET_DIR}" --epochs 20 --hours 9.5 --github-repo "{BACKUP_REPO}" --github-tag "{RUN_TAG}"
```

In a new Kaggle session, repeat the setup with identical data and dependency versions, then restore before running the same training command:

```python
!python -u github_backup.py restore --repo "{BACKUP_REPO}" --tag "{RUN_TAG}"
```

The restore destination defaults to /kaggle/working/academic_editor_training_v3 and must be empty. If training is already complete, evaluate instead. Restore only trusted checkpoints from your own repository. Use a unique run tag for a genuinely new experiment. To upload a saved local checkpoint after a failed backup (after training stops), run:

```python
!python -u github_backup.py upload --repo "{BACKUP_REPO}" --tag "{RUN_TAG}"
```

Add --final if training is complete. Checkpoints include the latest full Trainer state, best checkpoint and training configuration. Local checkpoint saves occur every 105 optimizer steps for the current 1,676-example dataset (half an epoch); remote backup happens after each save, at planned session pause, and on completion. If upload fails during training, training pauses at the local checkpoint. Abrupt termination may still lose work since the latest successful backup. Each GitHub asset must be under 2 GiB. Older verified assets are not automatically deleted; periodically retain the latest plus a fallback and preserve the final adapter elsewhere. Actual cloud upload and GPU resumption must be tested in Kaggle.
