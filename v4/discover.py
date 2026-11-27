"""Metadata-only candidate discovery. No paper text is downloaded or approved."""
import argparse, csv, json, re, time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

TOPICS = {
    "sign_language": 'ti:"sign language recognition" OR abs:"sign language recognition"',
    "skeleton_action": 'ti:"skeleton-based action recognition" OR ti:"skeleton action recognition"',
    "graph_learning": 'ti:"graph convolution" AND (abs:skeleton OR abs:pose)',
    "spatiotemporal": 'ti:spatiotemporal AND (abs:pose OR abs:motion)',
    "attention": 'ti:transformer AND (abs:skeleton OR abs:pose)',
    "efficient_vision": 'ti:efficient AND (abs:"action recognition" OR abs:"pose estimation")',
    "human_motion": 'ti:"human motion" AND (abs:prediction OR abs:generation)',
    "vision": 'ti:"human pose" AND (abs:representation OR abs:recognition)',
}
NS = {"a": "http://www.w3.org/2005/Atom"}

def query(term, count):
    params = urlencode({"search_query": "(" + term + ") AND (cat:cs.CV OR cat:cs.LG OR cat:cs.AI)",
                        "start": 0, "max_results": count, "sortBy": "relevance"})
    url = "https://export.arxiv.org/api/query?" + params
    request = Request(url, headers={"User-Agent": "AcademicEditorV4/1.0 (metadata research)"})
    with urlopen(request, timeout=90) as response:
        root = ET.fromstring(response.read())
    rows = []
    for entry in root.findall("a:entry", NS):
        url = entry.findtext("a:id", namespaces=NS)
        if not url:
            continue
        version = url.rstrip("/").split("/abs/", 1)[-1]
        lineage = re.sub(r"v\d+$", "", version)
        rows.append({"paper_id": lineage, "version_id": version,
                     "title": " ".join(entry.findtext("a:title", default="", namespaces=NS).split()),
                     "authors": "; ".join(x.findtext("a:name", default="", namespaces=NS) for x in entry.findall("a:author", NS)),
                     "published": entry.findtext("a:published", default="", namespaces=NS),
                     "url": "https://arxiv.org/abs/" + version,
                     "doi": entry.findtext("{http://arxiv.org/schemas/atom}doi", default="")})
    if not rows:
        raise RuntimeError("No entries returned; inspect query/API response before continuing")
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--per-topic", type=int, default=30)
    a = p.parse_args()
    if not 1 <= a.per_topic <= 100:
        p.error("per-topic must be between 1 and 100")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    by_id = {}
    for topic, term in TOPICS.items():
        cache = out / f"{topic}_{a.per_topic}.json"
        if cache.exists():
            rows = json.loads(cache.read_text(encoding="utf-8"))
        else:
            time.sleep(4)  # respect API request spacing
            rows = query(term, a.per_topic)
            tmp = cache.with_suffix(".tmp")
            tmp.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(cache)
        print(topic, len(rows), flush=True)
        for r in rows:
            key = r["paper_id"]
            if key not in by_id:
                by_id[key] = {**r, "topics": topic}
            else:
                by_id[key]["topics"] += ";" + topic
    fields = ["paper_id", "version_id", "title", "authors", "published", "url", "doi", "topics",
              "writing_review", "license_url", "permission_status", "notes"]
    dest = out / "papers.csv"
    if dest.exists():
        dest = out / f"papers_refresh_{int(time.time())}.csv" # never erase human review
    with dest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        writer.writerows(by_id.values())
    print(f"{len(by_id)} unique candidate papers: {dest}")
    print("Metadata only. All reuse permissions and writing quality remain unverified.")

if __name__ == "__main__":
    main()
