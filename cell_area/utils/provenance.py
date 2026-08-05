import json
import platform
import subprocess
import datetime
import pathlib
import os
import hashlib


def sha256_file(path) -> str:
    """Compute SHA-256 hex digest for a file (streamed, constant memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_provenance(
    out_dir,
    params: dict | None = None,
    script_path=None,
    filename: str = "run_meta.json",
):
    """Write a JSON provenance record identifying the code version that ran.

    Records a SHA-256 of the running script (``script_path``) plus the git
    commit, interpreter, platform, and any analysis ``params``. Provenance is
    best-effort: missing git or an unreadable script degrade to "unknown"
    rather than raising.
    """
    try:
        commit = subprocess.getoutput("git rev-parse --short HEAD") or "unknown"
    except Exception:
        commit = "unknown"

    script_name = "unknown"
    script_sha256 = "unknown"
    if script_path:
        script_name = os.path.basename(str(script_path))
        try:
            script_sha256 = sha256_file(script_path)
        except Exception:
            script_sha256 = "unknown"

    meta = {
        "time": datetime.datetime.now().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": commit,
        "script": script_name,
        "script_sha256": script_sha256,
        "cwd": os.getcwd(),
        "params": params or {},
    }
    out = pathlib.Path(out_dir) / filename
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return str(out)
