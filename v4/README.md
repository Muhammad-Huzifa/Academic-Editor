# Academic Editor V4 — scientific writing pilot

V3 remains at repository root, including its release and checkpoint state. V4 starts from the original Qwen3-4B-Instruct-2507 base, not the V3 adapter. Objective: author-directed technical explanation, clear argument, meaningful revision, precise evidence, and preservation of scientific content. No detector scores or detector-evasion objectives.

## Workflow and gates

1. Discover candidate paper metadata (CPU); inspect papers for writing quality and applicable reuse permissions.
2. Inspect authentic revisions (arXivEdits, CASIMIR, ParaRev). Do not treat dataset-level licenses as permission for all underlying articles. Use supplied author-approved before/after examples; keep unpublished manuscripts outside public GitHub. No raw papers, datasets, reviews or secrets committed.
3. Normalize reviewed revision pairs with the schema in `dataset_schema.md`; collect at least 100 human-reviewed pilot pairs. Split by paper lineage, including across revision datasets. Keep any final independent manuscript examples outside training and tuning.
4. Build the pilot via `python v4/build_dataset.py --input /path/to/reviewed.jsonl --out /kaggle/working/editor_v4_pilot`. Audit rejected rows, leakage, task coverage and length. Do not train until review and baseline comparisons are adequate.
5. SFT from original base with assistant-only loss; reuse root backup mechanism under separate V4 tag/run directory. Pilot before scaling; preference optimization only after collecting verified preferred/rejected responses. No fixed promise of 12,000 examples or 20 days.
6. Compare copy baseline, base Qwen, V3 and V4 on same held-out examples. Blind judgments: factual/technical correctness (critical error veto), clarity, coherence, substantive improvement, author preference. Factual checks are diagnostics rather than semantic guarantees. Identity editing assessed separately. Report per-task results and uncertainty.
7. Deploy only after meaningful gains over prompted base and V3. Follow journal AI-use disclosure requirements.

## First Kaggle notebook (CPU, Internet enabled)

```python
!git clone https://github.com/Muhammad-Huzifa/Academic-Editor.git /kaggle/working/Academic-Editor
%cd /kaggle/working/Academic-Editor
!python -u v4/discover.py --out /kaggle/working/editor_v4_corpus --per-topic 30
```

If already cloned, use `git pull --ff-only origin main` instead. The discovery command writes candidate metadata, not training data. Each completed query is cached; preserve the output directory between sessions to resume. The generated `papers.csv` is a review queue. No GPU or token needed for discovery.

## References / candidate revision sources

- arXivEdits: https://github.com/chaojiang06/arXivEdits — paper versions and aligned edits, includes version-specific license metadata.
- CASIMIR: https://aclanthology.org/2024.lrec-main.257/ — multi-version OpenReview articles and revisions.
- ParaRev: https://arxiv.org/abs/2501.05222 — paragraph-level revision dataset derived from CASIMIR; check original document rights.
- EarlySciRev: https://arxiv.org/abs/2603.28515 — candidate early-stage revision pairs; check provenance, consent and license.
- arXiv API and text reuse: https://info.arxiv.org/help/api/tou.html and https://info.arxiv.org/help/license/reuse.html

## Checkpoints

Use a unique run directory and tag for each experiment, e.g. `academic-editor-v4-pilot-sft`. Root `github_backup.py` requires explicit --run-dir and --tag for V4. Never restore V3 state into V4. Public releases are only appropriate for weights from non-confidential authorized sources; private work requires private authorized storage. Use one sequential trainer per run and only authorized compute quota. Keep exact code revision, dataset hashes, dependency versions and configuration for resume.

Status: discovery, pilot intake, SFT and baseline evaluation scripts implemented; external dataset adapters, expert preference annotations, DPO and production deployment are gated on inspecting actual sources and pilot outcomes. No training data has been approved by creating these files.

## After pilot approval (do not run yet)

Use a fresh run directory and release tag for each experiment. For a fully authorized public-data pilot, the familiar four-cell Kaggle workflow will be:

1. Connect GitHub via Kaggle Secret `GITHUB_TOKEN`.
2. Clone the repository at the frozen commit; install `requirements.txt` and record resolved package versions.
3. Download the frozen reviewed V4 dataset from authorized storage; on subsequent sessions restore using the V4 tag and run directory below.
4. Start or resume training with the identical dataset, code, environment and training configuration.

```bash
# Storage preflight (public data and trained artifacts only)
python github_backup.py check --repo Muhammad-Huzifa/Academic-Editor --tag academic-editor-v4-pilot-sft --allow-public

# Subsequent session only; skip for the first run. Restore into an empty directory.
python -u github_backup.py restore --repo Muhammad-Huzifa/Academic-Editor --tag academic-editor-v4-pilot-sft --run-dir /kaggle/working/academic_editor_training_v4

# After human review, task coverage, leakage audit and frozen dataset approval
python -u v4/train.py --dataset-dir /kaggle/working/editor_v4_pilot --approve-pilot --epochs 8 --hours 9.5 --github-repo Muhammad-Huzifa/Academic-Editor --github-tag academic-editor-v4-pilot-sft --allow-public

# Completed run only
python -u v4/evaluate.py --dataset-dir /kaggle/working/editor_v4_pilot
# Optionally add --v3-adapter /path/to/v3/best_lora_adapter
```

The trainer refuses public checkpoint uploads unless each record carries a positive public-release attestation and the dataset manifest agrees. This is a procedural gate, not legal clearance. For confidential author examples use authorized private storage and omit public upload flags; never place private materials in public releases.

The SFT pilot uses rank 16, LR 2e-5, 4096-token maximum, gradient accumulation 8, early stopping and automatic full-state backup. These are starting settings, not selected optimal hyperparameters. The evaluation script includes an exact-copy baseline, original base, V4 and optional V3, task metrics and randomized blind review sheets. Evaluation on independent author drafts and stronger prompted baselines is still needed before accepting V4. No DPO trainer is provided until real preference annotations are available.

## Self-test

`python v4/self_test.py` checks the local record validation, a generated fixture dataset and paper-disjoint split. It requires no GPU or downloaded papers. Run it before the first discovery session. This does not validate actual corpus rights or run the GPU trainer.
