# Pilot revision record (one JSON object per line)

Required fields:
- sample_id: stable unique string.
- paper_id: canonical lineage across all sources (arXiv ID without version, DOI or curated canonical ID). Link duplicate publications/preprints manually before splitting.
- source: dataset name or author-owned.
- source_url: original revision/source reference (local author-owned IDs allowed).
- rights_basis: explanation of permission for this pair and intended use.
- license_status: manually_verified.
- quality_status: human_approved (review both draft and revision for factual correctness and actual improvement).
- public_checkpoint_ok: boolean; true only when the reviewer has verified non-confidentiality and permission for public trained artifacts.
- section: introduction, related_work, methodology, results, ablation, discussion, conclusion, abstract.
- task: technical_explanation, argument, notes_to_prose, restructure, coherence, experimental_analysis, redundancy, grammar, identity.
- input: author-supplied text or notes; include evidence required for the revision.
- output: improved prose that is supported by input. Do not insert invented results, references, mechanisms or causal claims.
- review_note: reviewer explanation of meaningful improvements, or why identity is correct.

Keep unpublished drafts and confidential reviews private. These fields are human attestations, not automatic legal verification. For authentic revisions that alter science, exclude or construct a new evidence-complete input with authorization. Do not treat paragraph versions with new findings as ordinary edits. Split by lineage before adding derived examples; audit cross-source near duplicates manually. Exact hashes and simple number/citation checks do not prove semantic correctness.

Pilot: at least 100 reviewed pairs spanning at least 10 lineages. Scale only after independent review and blind comparison. No detector scores used.
