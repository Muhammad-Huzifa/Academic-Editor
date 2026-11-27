"""Build reviewed V4 dataset from normalized JSONL, never auto-approve examples."""
import argparse, hashlib, json, random, re, sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import MODEL_ID, numbers, citations, math_tokens, read_jsonl, write_json, write_jsonl, file_hash

TASKS = {
    "technical_explanation": "Clarify mechanisms and technical relationships using only supplied information.",
    "argument": "Develop the scientific argument and connect limitations to the proposed approach using supplied evidence.",
    "notes_to_prose": "Convert the supplied technical notes into coherent scientific prose without adding claims.",
    "restructure": "Meaningfully restructure awkward prose while preserving scientific content.",
    "coherence": "Improve logical progression and transitions across the supplied text.",
    "experimental_analysis": "Explain supplied observations and comparisons with appropriately bounded interpretation.",
    "redundancy": "Remove repetition and generic language while retaining every distinct scientific claim.",
    "grammar": "Correct grammar and technical phrasing without changing claims.",
    "identity": "Leave clear and correct text unchanged."
}
SECTIONS = set("introduction related_work methodology results ablation discussion conclusion abstract".split())
REQUIRED = set("sample_id paper_id source source_url rights_basis license_status quality_status public_checkpoint_ok section task input output review_note".split())
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"

def normalized(s):
    return " ".join(s.lower().split())

def validate(r):
    if not isinstance(r, dict) or REQUIRED - r.keys():
        return "missing required fields"
    if any(not isinstance(r[k], str) or not r[k].strip() for k in REQUIRED - {"public_checkpoint_ok"}):
        return "empty or invalid field"
    if type(r["public_checkpoint_ok"]) is not bool:
        return "invalid public checkpoint attestation"
    if r["quality_status"] != "human_approved" or r["license_status"] != "manually_verified":
        return "missing human or rights approval"
    if r["task"] not in TASKS or r["section"] not in SECTIONS:
        return "invalid task/section"
    if r["task"] == "identity" and r["input"].strip() != r["output"].strip():
        return "identity changed"
    if r["task"] != "identity" and normalized(r["input"]) == normalized(r["output"]):
        return "unchanged non-identity example"
    if numbers(r["input"]) != numbers(r["output"]) or citations(r["input"]) != citations(r["output"]) or math_tokens(r["input"]) != math_tokens(r["output"]):
        return "number/citation/math check failed (review and normalize source)"
    if sorted(re.findall(r"https?://[^\s)]+",r["input"])) != sorted(re.findall(r"https?://[^\s)]+",r["output"])):
        return "URL mismatch"
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        p.error("use a new output directory for each frozen dataset")
    out.mkdir(parents=True, exist_ok=True)
    rows, rejected, seen_ids, seen_texts = [], [], set(), set()
    for i, r in enumerate(read_jsonl(a.input), 1):
        error = validate(r)
        if not error:
            texts = {normalized(r["input"]), normalized(r["output"])}
            if r["sample_id"] in seen_ids or texts & seen_texts:
                error = "exact duplicate text or sample ID"
            else:
                seen_ids.add(r["sample_id"]); seen_texts.update(texts)
                r = dict(r)
                r["instruction"] = "Section: " + r["section"].replace("_", " ") + ". " + TASKS[r["task"]] + " Preserve scientific facts, values, units, citations, qualifiers and uncertainty. Return only revised prose."
                r["auto_pass"] = True
                rows.append(r)
        if error:
            rejected.append({"row": i, "sample_id": r.get("sample_id") if isinstance(r, dict) else None, "reason": error})
    write_jsonl(out / "rejections.jsonl", rejected)
    groups = sorted({r["paper_id"] for r in rows})
    if len(rows) < 100 or len(groups) < 10:
        raise SystemExit(f"Pilot gate failed: {len(rows)} reviewed rows, {len(groups)} lineages. Need >=100 rows and >=10 lineages.")
    random.Random(42).shuffle(groups)
    n = max(1, round(len(groups)*.1))
    split = {g: ("test" if i<n else "validation" if i<2*n else "train") for i,g in enumerate(groups)}
    parts = {s: [] for s in ("train", "validation", "test")}
    for r in rows:
        r["split"] = split[r["paper_id"]]; parts[r["split"]].append(r)
    files = {}
    for s, subset in parts.items():
        path = out / (s + ".jsonl"); write_jsonl(path, subset); files[path.name] = file_hash(path)
    write_json(out / "lineage_splits.json", split)
    manifest = {"schema_version": 4, "model_id": MODEL_ID, "model_revision": REVISION,
                "files": files, "counts": {s: len(rs) for s, rs in parts.items()},
                "tasks": {s: dict(Counter(r["task"] for r in rs)) for s, rs in parts.items()},
                "public_checkpoint_ok": all(r["public_checkpoint_ok"] for r in rows),
                "human_review_required": True, "source_sha256": file_hash(a.input)}
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    print("Frozen pilot created. Audit task coverage, near duplicates and review decisions before training.")

if __name__ == "__main__":
    main()
