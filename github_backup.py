"""GitHub Release backups for full Trainer resume state.

Public uploads require explicit --allow-public acknowledgement.
Authenticate writes through GITHUB_TOKEN; public restores can be anonymous.
Never add weights or credentials to Git history.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import zipfile
import uuid
from pathlib import Path
from urllib.parse import quote

import requests

LIMIT = 2 * 1024**3
DEFAULT_REPO = "Muhammad-Huzifa/Academic-Editor"
DEFAULT_TAG = "academic-editor-v3-final-training"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class GitHubBackup:
    def __init__(self, repo, tag=DEFAULT_TAG, allow_public=False, read_only=False):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
            raise ValueError("Expected OWNER/REPO")
        self.tag = tag
        self.api = f"https://api.github.com/repos/{repo}"
        token = os.environ.get("GITHUB_TOKEN")
        if not token and not read_only:
            raise RuntimeError("Uploading checkpoints requires GITHUB_TOKEN in Kaggle Secrets, even for a public repository")
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        if token:
            self.session.headers["Authorization"] = "Bearer " + token
        info = self.request("GET", self.api).json()
        self.public = not info.get("private")
        if self.public and not read_only:
            if not allow_public:
                raise RuntimeError("This repository is public. Pass --allow-public only if you intend to publish the trained weights and resume state.")
            print("WARNING: checkpoint assets will be publicly downloadable. Upload only authorized, non-confidential artifacts.", flush=True)

    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", (20, 180))
        response = self.session.request(method, url, **kwargs)
        if not response.ok:
            hints = {
                401: "Token invalid or expired; update GITHUB_TOKEN in Kaggle Secrets.",
                403: "Token lacks access/write permission, or GitHub rate limit reached.",
                404: "Verify repository name and token access to this repository.",
                422: "GitHub rejected the request; check release or asset settings.",
            }
            raise RuntimeError(f"GitHub {method} returned HTTP {response.status_code}: {hints.get(response.status_code, 'Check network and GitHub status.')}")
        return response

    def release(self, create=False):
        url = self.api + "/releases/tags/" + quote(self.tag, safe="")
        r = self.session.get(url, timeout=(20, 60))
        if r.ok:
            return r.json()
        if r.status_code == 404 and create:
            return self.request("POST", self.api + "/releases", json={
                "tag_name": self.tag,
                "name": self.tag,
                "body": ("Public" if self.public else "Private") + " training state backups. Keep the newest verified checkpoint and a previous fallback.",
                "prerelease": True,
            }).json()
        raise RuntimeError(f"Release lookup HTTP {r.status_code}")

    def assets(self, release):
        page = 1
        result = []
        while True:
            items = self.request("GET", self.api + f"/releases/{release['id']}/assets",
                                 params={"per_page": 100, "page": page}).json()
            result.extend(items)
            if len(items) < 100:
                return result
            page += 1

    def upload(self, archive, release):
        path = Path(archive)
        if path.stat().st_size >= LIMIT:
            raise RuntimeError("Checkpoint exceeds the GitHub 2 GiB asset limit; use private model/object storage instead")
        expected = "sha256:" + sha256(path)
        for attempt in range(3):
            try:
                existing = [x for x in self.assets(release) if x["name"] == path.name]
                if existing:
                    x = existing[0]
                    if x.get("state") == "uploaded" and x["size"] == path.stat().st_size and x.get("digest") == expected:
                        print("Remote checkpoint already verified:", path.name, flush=True)
                        return
                    # Delete only a failed placeholder from our own failed upload.
                    if x.get("state") == "starter":
                        self.request("DELETE", self.api + f"/releases/assets/{x['id']}")
                    else:
                        raise RuntimeError("Existing asset differs; preserve it and choose a new run tag")
                with path.open("rb") as f:
                    response = self.request("POST", release["upload_url"].split("{")[0],
                        params={"name": path.name}, headers={"Content-Type": "application/zip"},
                        data=f, timeout=(30, 900)).json()
                if response.get("state") != "uploaded" or response.get("size") != path.stat().st_size:
                    raise RuntimeError("Incomplete remote asset")
                if response.get("digest") and response["digest"] != expected:
                    raise RuntimeError("Remote checksum mismatch")
                print("Checkpoint uploaded:", path.name, flush=True)
                return
            except (requests.RequestException, RuntimeError) as e:
                print(f"Backup attempt {attempt+1}/3 failed ({type(e).__name__})", flush=True)
                if attempt == 2:
                    raise RuntimeError("Remote backup failed; local checkpoint retained") from None
                time.sleep(5 * (attempt+1))

    def download(self, asset, stream=False):
        # Public assets can be downloaded without sending credentials.
        if self.public:
            response = requests.get(asset["browser_download_url"], stream=stream, timeout=(30, 900))
            if not response.ok:
                response.close()
                raise RuntimeError("Public release asset download failed")
            return response
        return self.request("GET", self.api + f"/releases/assets/{asset['id']}",
                            headers={"Accept": "application/octet-stream"},
                            stream=stream, timeout=(30, 900))

    def check(self):
        """Verify actual release-asset upload and download before training."""
        release = self.release(create=True)
        name = "preflight-" + uuid.uuid4().hex + ".txt"
        payload = b"Academic Editor checkpoint storage preflight"
        asset = None
        try:
            asset = self.request("POST", release["upload_url"].split("{")[0],
                                 params={"name": name},
                                 headers={"Content-Type": "text/plain"},
                                 data=payload).json()
            if asset.get("state") != "uploaded" or asset.get("size") != len(payload):
                raise RuntimeError("Preflight asset upload incomplete")
            with self.download(asset) as response:
                received = response.content
            if received != payload:
                raise RuntimeError("Preflight download did not match uploaded bytes")
        finally:
            if asset and asset.get("id"):
                self.request("DELETE", self.api + f"/releases/assets/{asset['id']}")
        print("GitHub checkpoint storage verified (upload + download). Release:", self.tag, flush=True)

    def save(self, run_dir, final=False):
        run = Path(run_dir).resolve()
        ckpts = sorted((p for p in (run/"checkpoints").glob("checkpoint-*")
                        if p.is_dir() and re.fullmatch(r"checkpoint-\d+", p.name)),
                       key=lambda p: int(p.name.split("-")[-1]))
        if not ckpts:
            raise RuntimeError("No saved Trainer checkpoint yet")
        latest = ckpts[-1]
        required = ["adapter_model.safetensors", "adapter_config.json", "trainer_state.json",
                    "optimizer.pt", "scheduler.pt", "rng_state.pth"]
        missing = [x for x in required if not (latest/x).is_file()]
        if missing:
            raise RuntimeError("Incomplete checkpoint: " + ", ".join(missing))
        state = json.loads((latest/"trainer_state.json").read_text())
        step = int(latest.name.split("-")[-1])
        if state["global_step"] != step:
            raise RuntimeError("Checkpoint step mismatch")
        dirs = {latest}
        best = state.get("best_model_checkpoint")
        if best:
            bestdir = run/"checkpoints"/Path(best).name
            if not (bestdir/"adapter_model.safetensors").exists():
                raise RuntimeError("Best checkpoint missing; preserve full run directory")
            dirs.add(bestdir)
        files = [run/"training_config.json"]
        for name in ["training_history.csv", "training_summary.json", "session_status.json"]:
            if (run/name).is_file():
                files.append(run/name)
        if final:
            if not (run/"best_lora_adapter"/"adapter_model.safetensors").exists():
                raise RuntimeError("Final adapter missing")
            dirs.add(run/"best_lora_adapter")
        if not files[0].is_file():
            raise RuntimeError("Training configuration missing")
        # Only archive the latest full checkpoint, best checkpoint and known metadata.
        for d in dirs:
            files.extend(p for p in d.rglob("*") if p.is_file())
        with tempfile.TemporaryDirectory(prefix="editor-backup-") as tmp:
            tmpfile = Path(tmp)/"backup.zip"
            with zipfile.ZipFile(tmpfile, "w", compression=zipfile.ZIP_STORED) as z:
                for f in sorted(set(files)):
                    if f.is_symlink() or not f.resolve().is_relative_to(run):
                        raise RuntimeError("Unexpected checkpoint path")
                    z.write(f, f.relative_to(run).as_posix())
            kind = "final" if final else "resume"
            archive = tmpfile.with_name(f"{kind}-{step:08d}-{sha256(tmpfile)[:16]}.zip")
            tmpfile.rename(archive)
            self.upload(archive, self.release(create=True))

    def restore(self, run_dir, asset_name=None):
        run = Path(run_dir).resolve()
        if run.exists() and any(run.iterdir()):
            raise RuntimeError("Restore into an empty directory; never overwrite local checkpoints")
        assets = [a for a in self.assets(self.release())
                  if a.get("state") == "uploaded"
                  and re.fullmatch(r"(resume|final)-\d{8}-[a-f0-9]{16}\.zip", a["name"])]
        if asset_name:
            assets = [a for a in assets if a["name"] == asset_name]
        if not assets:
            raise RuntimeError("No matching checkpoint asset")
        a = max(assets, key=lambda x: (int(x["name"].split("-")[1]),
                                       x["name"].startswith("final-"), x["created_at"]))
        with tempfile.TemporaryDirectory(prefix="editor-restore-") as tmp:
            archive = Path(tmp)/a["name"]
            with self.download(a, stream=True) as response:
                with archive.open("wb") as f:
                    for chunk in response.iter_content(1024*1024):
                        f.write(chunk)
            digest = sha256(archive)
            if archive.stat().st_size != a["size"] or not digest.startswith(a["name"].split("-")[-1][:-4]):
                raise RuntimeError("Downloaded checkpoint checksum mismatch")
            if a.get("digest") and a["digest"] != "sha256:"+digest:
                raise RuntimeError("GitHub digest mismatch")
            with zipfile.ZipFile(archive) as z:
                for member in z.infolist():
                    path = Path(member.filename)
                    if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                        raise RuntimeError("Unsafe archive member")
                z.extractall(run)
        print("Restored:", a["name"], "to", run, flush=True)
        if a["name"].startswith("final-"):
            print("Final adapter restored. Evaluate instead of resuming training.", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["check", "upload", "restore"])
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--tag", default=DEFAULT_TAG)
    p.add_argument("--run-dir", default="/kaggle/working/academic_editor_training_v3")
    p.add_argument("--asset")
    p.add_argument("--final", action="store_true")
    p.add_argument("--allow-public", action="store_true", help="Explicitly permit publicly downloadable training state")
    a = p.parse_args()
    backup = GitHubBackup(a.repo, a.tag, allow_public=a.allow_public, read_only=(a.action == "restore"))
    if a.action == "check":
        backup.check()
    elif a.action == "upload":
        backup.save(a.run_dir, final=a.final)
    else:
        backup.restore(a.run_dir, a.asset)


if __name__ == "__main__":
    main()
