"""
Model registry on S3 (the bucket comes from terraform/model_store.tf). Models are not kept in git.

Layout under a model prefix, e.g. s3://<bucket>/pulse-v3:
  <version>/predictor.pkl     the trained bundle
  <version>/manifest.json     version, sha256, size, scikit-learn version, created, notes, held-out metrics
  latest.json                 {"version": "<version>"}: what PULSE_MODEL_URI=s3://<bucket>/pulse-v3 loads
  _data/                      the training data (train.py): tau_bench/*.json, tau_bench/schemas/, embed_cache.npz

    python pulse_pred/model_store.py publish predictor.pkl s3://<bucket>/pulse-v3 --version 2026-11-03 --notes "..."
    python pulse_pred/model_store.py fetch s3://<bucket>/pulse-v3            # -> pulse_pred/data/models/<version>/
    python pulse_pred/model_store.py list s3://<bucket>/pulse-v3             # versions, metrics, which is latest
    python pulse_pred/model_store.py promote s3://<bucket>/pulse-v3 <version>
    python pulse_pred/model_store.py data push|pull s3://<bucket>/pulse-v3   # pulse_pred/data/ <-> _data/

fetch() downloads a version once and reuses the local copy, checking its sha256 against the manifest.
Everything stays under the model prefix, so the reader / publisher IAM policies (terraform/) cover it.
Credentials come from the usual AWS chain: a profile or env on a laptop, the pod's IAM role on EKS.
"""

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):                         # run as a script: make `pulse_pred` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pulse_pred import paths  # noqa: E402

FILE = "predictor.pkl"


def _split(uri):
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// URI: {uri!r}")
    bucket, _, prefix = uri[5:].partition("/")
    return bucket, prefix.rstrip("/")


def _s3():
    import boto3                                      # only needed when a model comes from S3
    return boto3.client("s3")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(uri, s3=None):
    """s3://bucket/prefix (-> latest.json) or s3://bucket/prefix/<version> -> (bucket, model prefix, version)."""
    s3 = s3 or _s3()
    bucket, prefix = _split(uri)
    try:
        latest = json.loads(s3.get_object(Bucket=bucket, Key=f"{prefix}/latest.json")["Body"].read())
        return bucket, prefix, latest["version"]
    except s3.exceptions.NoSuchKey:                   # the URI already names a version
        parent, _, version = prefix.rpartition("/")
        return bucket, parent, version


def fetch(uri, dest=None, s3=None):
    """Local path of the model at `uri`, downloaded on first use."""
    s3 = s3 or _s3()
    bucket, prefix, version = resolve(uri, s3)
    local = Path(dest or paths.DATA / "models") / version
    path = local / FILE
    manifest = json.loads(s3.get_object(Bucket=bucket, Key=f"{prefix}/{version}/manifest.json")["Body"].read())
    if not (path.exists() and _sha256(path) == manifest["sha256"]):
        local.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, f"{prefix}/{version}/{FILE}", str(path))
        if _sha256(path) != manifest["sha256"]:
            raise ValueError(f"{uri} version {version}: sha256 does not match its manifest")
    (local / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return path


def publish(model, uri, version, notes="", latest=True, s3=None, metrics=None):
    """Upload a trained bundle as a new version (never overwrites one) and point latest.json at it."""
    s3 = s3 or _s3()
    bucket, prefix = _split(uri)
    key = f"{prefix}/{version}/{FILE}"
    if s3.list_objects_v2(Bucket=bucket, Prefix=key).get("KeyCount"):
        raise ValueError(f"version {version} already exists under {uri}")
    import sklearn
    manifest = {"version": version, "file": FILE, "sha256": _sha256(model), "bytes": Path(model).stat().st_size,
                "sklearn": sklearn.__version__, "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "notes": notes, **({"metrics": metrics} if metrics else {})}
    s3.upload_file(str(model), bucket, key)
    s3.put_object(Bucket=bucket, Key=f"{prefix}/{version}/manifest.json", Body=json.dumps(manifest, indent=1).encode(),
                  ContentType="application/json")
    if latest:
        s3.put_object(Bucket=bucket, Key=f"{prefix}/latest.json", Body=json.dumps({"version": version}).encode(),
                      ContentType="application/json")
    return manifest


def versions(uri, s3=None) -> list[dict]:
    """Every version's manifest, newest first, with "latest": True on the one latest.json names."""
    s3 = s3 or _s3()
    bucket, prefix = _split(uri)
    try:
        latest = json.loads(s3.get_object(Bucket=bucket, Key=f"{prefix}/latest.json")["Body"].read())["version"]
    except s3.exceptions.NoSuchKey:
        latest = None
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix}/", Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            v = cp["Prefix"][len(prefix) + 1:].rstrip("/")
            if v.startswith("_"):                         # _data/ etc.
                continue
            try:
                m = json.loads(s3.get_object(Bucket=bucket, Key=f"{prefix}/{v}/manifest.json")["Body"].read())
            except s3.exceptions.NoSuchKey:
                continue                                  # an upload that never finished
            out.append({**m, "latest": v == latest})
    return sorted(out, key=lambda m: m.get("created", ""), reverse=True)


def promote(uri, version, s3=None):
    """Point latest.json at an existing version (rollback = promote an older one)."""
    s3 = s3 or _s3()
    bucket, prefix = _split(uri)
    s3.head_object(Bucket=bucket, Key=f"{prefix}/{version}/manifest.json")      # must exist
    s3.put_object(Bucket=bucket, Key=f"{prefix}/latest.json", Body=json.dumps({"version": version}).encode(),
                  ContentType="application/json")


DATA_FILES = ("tau_bench/*.json", "tau_bench/schemas/**/*", "embed_cache.npz")


def data(direction, uri, local=None, s3=None) -> list[str]:
    """Sync the training data between pulse_pred/data/ and <prefix>/_data/ (files whose size differs)."""
    s3 = s3 or _s3()
    bucket, prefix = _split(uri)
    local = Path(local or paths.DATA)
    moved = []
    if direction == "push":
        for pat in DATA_FILES:
            for f in sorted(p for p in local.glob(pat) if p.is_file()):
                key = f"{prefix}/_data/{f.relative_to(local).as_posix()}"
                try:
                    same = s3.head_object(Bucket=bucket, Key=key)["ContentLength"] == f.stat().st_size
                except Exception:
                    same = False
                if not same:
                    s3.upload_file(str(f), bucket, key)
                    moved.append(key)
    else:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix}/_data/"):
            for o in page.get("Contents", []):
                f = local / o["Key"][len(prefix) + 7:]
                if not (f.exists() and f.stat().st_size == o["Size"]):
                    f.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(bucket, o["Key"], str(f))
                    moved.append(str(f))
    return moved


def main():
    ap = argparse.ArgumentParser(description="pulse-v3 models on S3")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish", help="upload a trained bundle as a new version")
    p.add_argument("model")
    p.add_argument("uri", help="s3://<bucket>/pulse-v3")
    p.add_argument("--version", required=True)
    p.add_argument("--notes", default="")
    p.add_argument("--no-latest", action="store_true", help="upload without making it the latest")
    f = sub.add_parser("fetch", help="download a version (default: latest)")
    f.add_argument("uri")
    ls = sub.add_parser("list", help="every version, newest first")
    ls.add_argument("uri")
    pr = sub.add_parser("promote", help="make a version the latest (also: roll back)")
    pr.add_argument("uri")
    pr.add_argument("version")
    d = sub.add_parser("data", help="sync the training data: push (local -> S3) or pull")
    d.add_argument("direction", choices=["push", "pull"])
    d.add_argument("uri")
    a = ap.parse_args()
    if a.cmd == "publish":
        print(json.dumps(publish(a.model, a.uri, a.version, a.notes, latest=not a.no_latest), indent=1))
    elif a.cmd == "fetch":
        print(fetch(a.uri))
    elif a.cmd == "list":
        for m in versions(a.uri):
            met = (m.get("metrics") or {}).get("new") or {}
            print(f"{'*' if m['latest'] else ' '} {m['version']:<20} {m.get('created', '')[:16]}  "
                  f"warm@3 {met.get('warm@3', '-')}  hit@3 {met.get('hit@3', '-')}  {m.get('notes', '')[:60]}")
    elif a.cmd == "promote":
        promote(a.uri, a.version)
        print(f"latest -> {a.version}")
    else:
        moved = data(a.direction, a.uri)
        print(f"{len(moved)} file(s) {'uploaded' if a.direction == 'push' else 'downloaded'}")


if __name__ == "__main__":
    main()
