"""
Rename files: <key>_<well>_<other> -> <key>_<well>_<40|60|80>, preserving extension(s).
Writes a CSV LUT of all actions (renamed, skipped, errors).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple


_WELL_RE = re.compile(r"^(?P<row>[A-Za-z])(?P<num>\d+)$")
_TMP_PREFIX = ".tmp_rename_"


@dataclass(frozen=True)
class Plan:
    src: Path
    dst: Path
    key: str
    well: str
    n: int
    suffix: int
    status: str
    message: str


def _all_suffixes(p: Path) -> str:
    return "".join(p.suffixes) if p.suffixes else ""


def _base_no_ext(p: Path) -> str:
    ext = _all_suffixes(p)
    return p.name[: -len(ext)] if ext else p.name


def _normcase(p: Path) -> str:
    return os.path.normcase(str(p))


def suffix_from_well(well: str) -> Tuple[Optional[int], Optional[int], str]:
    m = _WELL_RE.match(well)
    if not m:
        return None, None, f"Invalid well '{well}' (expected XN, e.g. A1)"
    n = int(m.group("num"))
    mod = n % 4
    if mod == 2:
        return 40, n, ""
    if mod == 3:
        return 60, n, ""
    if mod == 0:
        return 80, n, ""
    return None, n, f"Well number {n} does not match 2/3/4 + 4*i"


def build_plan(p: Path) -> Plan:
    base = _base_no_ext(p)
    ext = _all_suffixes(p)

    parts = base.split("_", 2)
    if len(parts) != 3 or not parts[0] or not parts[1]:
        return Plan(p, p, "", "", -1, -1, "SKIP", "Bad pattern: <key>_<well>_<other>")

    key, well = parts[0], parts[1]
    suffix, n, msg = suffix_from_well(well)
    if suffix is None or n is None:
        return Plan(p, p, key, well, n if n is not None else -1, -1, "SKIP", msg)

    dst = p.with_name(f"{key}_{well}_{suffix}_renamed{ext}")
    if dst.name == p.name:
        return Plan(p, p, key, well, n, suffix, "SKIP", "Already conforms")

    return Plan(p, dst, key, well, n, suffix, "PENDING", "")


def iter_files(folder: Path, recursive: bool) -> Iterable[Path]:
    it = folder.rglob("*") if recursive else folder.glob("*")
    for p in it:
        if not p.is_file():
            continue
        if p.name.startswith(_TMP_PREFIX):
            continue
        yield p


def resolve_conflicts(plans: list[Plan]) -> list[Plan]:
    pending = [p for p in plans if p.status == "PENDING"]
    by_dst: dict[str, list[Plan]] = {}
    for p in pending:
        by_dst.setdefault(_normcase(p.dst), []).append(p)

    colliding = {k for k, v in by_dst.items() if len(v) > 1}
    src_set = {_normcase(p.src) for p in pending}

    out: list[Plan] = []
    for p in plans:
        if p.status != "PENDING":
            out.append(p)
            continue

        if _normcase(p.dst) in colliding:
            out.append(
                Plan(
                    p.src,
                    p.dst,
                    p.key,
                    p.well,
                    p.n,
                    p.suffix,
                    "SKIP",
                    "Collision: multiple sources map to same destination",
                )
            )
            continue

        if p.dst.exists() and _normcase(p.dst) not in src_set:
            out.append(
                Plan(
                    p.src,
                    p.dst,
                    p.key,
                    p.well,
                    p.n,
                    p.suffix,
                    "SKIP",
                    "Conflict: destination already exists",
                )
            )
            continue

        out.append(p)

    return out


def rename_two_stage(plans: list[Plan], dry_run: bool) -> list[Plan]:
    todo = [p for p in plans if p.status == "PENDING"]
    done = [p for p in plans if p.status != "PENDING"]

    if dry_run:
        done.extend(
            Plan(p.src, p.dst, p.key, p.well, p.n, p.suffix, "DRY_RUN", "Would rename")
            for p in todo
        )
        return done

    tmp_for_src: dict[str, Path] = {}
    try:
        for p in todo:
            ext = _all_suffixes(p.src)
            tmp = p.src.with_name(
                f"{_TMP_PREFIX}{uuid.uuid4().hex}_{_base_no_ext(p.src)}{ext}"
            )
            while tmp.exists():
                tmp = p.src.with_name(
                    f"{_TMP_PREFIX}{uuid.uuid4().hex}_{_base_no_ext(p.src)}{ext}"
                )
            p.src.rename(tmp)
            tmp_for_src[_normcase(p.src)] = tmp

        for p in todo:
            tmp_for_src[_normcase(p.src)].rename(p.dst)

        done.extend(
            Plan(p.src, p.dst, p.key, p.well, p.n, p.suffix, "RENAMED", "OK")
            for p in todo
        )
        return done

    except Exception as exc:
        for p in todo:
            tmp = tmp_for_src.get(_normcase(p.src))
            if tmp and tmp.exists() and not p.src.exists():
                try:
                    tmp.rename(p.src)
                except Exception:
                    pass
        done.extend(
            Plan(
                p.src,
                p.dst,
                p.key,
                p.well,
                p.n,
                p.suffix,
                "ERROR",
                f"Exception during rename: {exc}",
            )
            for p in todo
        )
        return done


def default_csv_path(folder: Path) -> Path:
    return folder / f"LUT_{folder.name}.csv"


def write_lut(plans: list[Plan], csv_path: Path) -> None:
    fields = [
        "original_name",
        "new_name",
        "status",
        "message",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in sorted(plans, key=lambda x: x.src.name.lower()):
            w.writerow(
                {
                    "original_name": p.src.name,
                    "new_name": p.dst.name,
                    "status": p.status,
                    "message": p.message,
                }
            )


def pick_folder_gui() -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select folder containing files to rename")
    root.destroy()
    return Path(folder) if folder else None


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="", help="Folder to process (omit for GUI picker)")
    ap.add_argument("--recursive", action="store_true", help="Include subfolders")
    ap.add_argument("--dry-run", action="store_true", help="Do not rename; only write LUT")
    ap.add_argument("--csv", default="", help="Output LUT CSV path")
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    folder = Path(args.folder) if args.folder else pick_folder_gui()
    if not folder:
        print("No folder selected.", file=sys.stderr)
        return 2
    if not folder.is_dir():
        print(f"Not a folder: {folder}", file=sys.stderr)
        return 2

    plans = [build_plan(p) for p in iter_files(folder, args.recursive)]
    plans = resolve_conflicts(plans)
    plans = rename_two_stage(plans, args.dry_run)

    csv_path = Path(args.csv) if args.csv else default_csv_path(folder)
    write_lut(plans, csv_path)

    total = len(plans)
    renamed = sum(p.status == "RENAMED" for p in plans)
    dry = sum(p.status == "DRY_RUN" for p in plans)
    skipped = sum(p.status == "SKIP" for p in plans)
    errors = sum(p.status == "ERROR" for p in plans)

    print(f"Folder: {folder}")
    print(f"Files scanned: {total}")
    print(f"Renamed: {renamed} | Dry-run: {dry} | Skipped: {skipped} | Errors: {errors}")
    print(f"LUT CSV: {csv_path}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
