"""Offline smoke test; all example text is synthetic fixture data, not training data."""
import json, subprocess, sys, tempfile
from pathlib import Path
from build_dataset import validate

def fixture(i):
    return {
        "sample_id": f"fixture-{i}", "paper_id": f"fixture-lineage-{i//10}",
        "source": "synthetic-fixture", "source_url": "local-fixture",
        "rights_basis": "test fixture only", "license_status": "manually_verified",
        "quality_status": "human_approved", "public_checkpoint_ok": False,
        "section": "methodology", "task": "technical_explanation",
        "input": f"The fixture module {i} has 2 channels and processes the input.",
        "output": f"Fixture module {i} processes the input through 2 channels.",
        "review_note": "test fixture only; never use for model training"
    }

def main():
    assert validate(fixture(0)) is None
    assert validate({**fixture(0), "output": "The fixture module 0 has 3 channels."})
    assert validate({**fixture(0), "quality_status": "unreviewed"})
    assert validate({**fixture(0), "output": fixture(0)["input"]})
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); raw = root / "fixture.jsonl"; out = root / "built"
        rows = [fixture(i) for i in range(100)]
        raw.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
        subprocess.run([sys.executable, str(Path(__file__).with_name("build_dataset.py")),
                        "--input", str(raw), "--out", str(out)], check=True)
        m = json.loads((out/"manifest.json").read_text())
        assert m["counts"] == {"train":80,"validation":10,"test":10}
        assert m["public_checkpoint_ok"] is False
        groups = [{json.loads(line)["paper_id"] for line in (out/(s+".jsonl")).read_text().splitlines()}
                  for s in ("train","validation","test")]
        assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
    print("V4 offline smoke test passed.")

if __name__ == "__main__":
    main()
