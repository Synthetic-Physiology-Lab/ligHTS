import datetime
import json
import os
import pathlib
import platform
import subprocess


def write_provenance(out_dir, params: dict | None = None):
    try:
        commit = subprocess.getoutput("git rev-parse --short HEAD") or "unknown"
    except Exception:
        commit = "unknown"
    meta = {
        "time": datetime.datetime.now().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": commit,
        "cwd": os.getcwd(),
        "params": params or {},
    }
    out = pathlib.Path(out_dir) / "run_meta.json"
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return str(out)
