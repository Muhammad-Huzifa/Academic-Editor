# Academic Editor — Technical Research Writing

Qwen3-4B-Instruct-2507 + QLoRA. Refine author-supplied abstracts, methodology and results without modifying scientific facts, component roles, numbers, citations or uncertainty. The goal is precise and natural scientific prose, not AI-detector evasion.

## V4 — scientific writing research pilot

The new [V4 workflow](v4/README.md) is isolated from the completed V3 baseline below. Begin with paper metadata discovery and rights review; do not launch GPU training before the reviewed dataset is frozen. V4 has its own dataset schema, trainer, evaluation and release tag.

## V3 project workflow

`prepare_plan.py` → frozen plan → `run_shard.py` (6 independent partitions) → `merge_shards.py` → `train.py` → `evaluate.py`.

The repository contains source code and optionally public checkpoint Release assets. Keep datasets, credentials, and confidential research out of Git history. Publish trained weights only after verifying source permissions and confidentiality. Use authorized compute quotas. Six logical shards support reproducibility and resumability, not quota circumvention.

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


## Remote checkpoint backup — existing public repository

Use **Muhammad-Huzifa/Academic-Editor** for code and checkpoint Release assets. No second repository or visibility change is required. Public Release assets can be downloaded by anyone, including the full training state and trained adapter. Publish only artifacts derived from authorized, non-confidential data; model weights may memorize training text. Never upload raw datasets or credentials.

Public cloning and restoring do not require authentication. Creating Releases and uploading assets require a GitHub personal access token with Contents read/write access to this repository. Store it as GITHUB_TOKEN in Kaggle Secrets and do not print it. For private repositories, authenticated Git clone and restore are required; omit --allow-public when writing.

In a fresh GPU notebook, enable Internet and use:

```python
!git clone https://github.com/Muhammad-Huzifa/Academic-Editor.git /kaggle/working/Academic-Editor
%cd /kaggle/working/Academic-Editor
!pip install -q -r requirements.txt gdown
```

For an existing checkout use git pull --ff-only instead of cloning.

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["GITHUB_TOKEN"] = UserSecretsClient().get_secret("GITHUB_TOKEN")
BACKUP_REPO = "Muhammad-Huzifa/Academic-Editor"
RUN_TAG = "academic-editor-v3-final-training"
```

```python
!python github_backup.py check --repo "{BACKUP_REPO}" --tag "{RUN_TAG}" --allow-public
```

The check creates a release if needed, uploads a small test asset, verifies its publicly downloaded bytes, and removes only that test asset. After it succeeds, verify PyTorch CUDA and set DATASET_DIR to the original four dataset files.

For a new run with no prior checkpoint (use a unique tag per experiment):

```python
!python -u train.py --dataset-dir "{DATASET_DIR}" --epochs 20 --hours 9.5 --github-repo "{BACKUP_REPO}" --github-tag "{RUN_TAG}" --allow-public
```

For subsequent Kaggle sessions, repeat the setup with the same data and dependency versions. Restore before executing the same training command:

```python
!python -u github_backup.py restore --repo "{BACKUP_REPO}" --tag "{RUN_TAG}"
```

Restore downloads the most recent uploaded archive and checks its size and checksum. Public restore can be used without a token, but continuing remote backups requires a token. The destination defaults to /kaggle/working/academic_editor_training_v3 and must be empty. Restore only trusted checkpoints from your own repository; optimizer and RNG states contain serialized objects. If training is complete, evaluate instead.

If an upload failed, after training stops retry with:

```python
!python -u github_backup.py upload --repo "{BACKUP_REPO}" --tag "{RUN_TAG}" --allow-public
```

Add --final if training is complete. Each backup contains the latest full Trainer checkpoint, best checkpoint and run configuration. Local checkpoints are saved every 105 optimizer steps for the 1,676-example training set, with uploads after saves, planned pause and completion. Failed upload pauses training at the local checkpoint. An unexpected termination can lose progress since the last successful upload. Each asset must be smaller than 2 GiB. Retain at least the latest verified backup and a fallback; older assets are not automatically deleted. Use another durable copy of important final adapters. Actual cloud uploads and GPU resumption still require Kaggle verification.
