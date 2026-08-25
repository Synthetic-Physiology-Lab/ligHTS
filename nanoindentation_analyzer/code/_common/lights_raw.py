"""Reader for the Optics11 Chiaro text files.

Conventions
--------------
* forces in uN as recorded, displacements in nm as recorded, time in s;
* sample displacement ``s = piezo - cantilever`` with no baseline applied,
  so that different baseline models can be compared on identical inputs;
* no filtering and clipping

"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "CAMPAIGNS",
    "CURVE_RE",
    "RawCurve",
    "campaign_curve_paths",
    "iter_campaign",
    "parse_header",
    "read_chiaro",
]

CAMPAIGNS = ("D1_June2025", "D2_June2026", "D3_July2026")
#: Real files are ``S-1 X-nn Y-nn I-nn.txt``. Synthetic ones carry a
#: ``SYNTHETIC_`` prefix

CURVE_RE = re.compile(
    r"^(?P<synth>SYNTHETIC_)?S-(?P<scan>\d+) X-(?P<x>\d\d)"
    r" Y-(?P<y>\d\d) I-(?P<i>\d\d)\.txt$"
)

_TABLE_MARK = "Time (s)"
_BOM = "﻿"


@dataclass
class RawCurve:
    """Everything one Chiaro file contains """

    path: Path
    header: dict
    time_s: np.ndarray
    load_uN: np.ndarray
    vendor_indent_nm: np.ndarray
    cantilever_nm: np.ndarray
    piezo_nm: np.ndarray
    auxiliary: np.ndarray
    #: piezo minus cantilever deflection, no baseline removed
    s_nm: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.s_nm = self.piezo_nm - self.cantilever_nm

    # guarded header
    @property
    def radius_um(self) -> float:
        return _as_float(self.header.get("Tip radius (um)"))

    @property
    def spring_constant_n_m(self) -> float:
        return _as_float(self.header.get("k (N/m)"))

    @property
    def calibration_factor(self) -> float:
        return _as_float(self.header.get("Calibration factor"))

    @property
    def e_eff_pa(self) -> float:
        return _as_float(self.header.get("E[eff] (Pa)"))

    @property
    def status(self) -> str:
        return str(self.header.get("Status", "")).strip()

    @property
    def date(self) -> str:
        return str(self.header.get("Date", "")).strip()

    @property
    def time_of_day(self) -> str:
        return str(self.header.get("Time", "")).strip()

    @property
    def sm_duration_s(self) -> float:
        return _as_float(self.header.get("SMDuration (s)"))

    @property
    def dt_s(self) -> float:
        if self.time_s.size < 2:
            return float("nan")
        return float(np.median(np.diff(self.time_s)))

    @property
    def ramp_window_s(self) -> tuple[float, float]:
        """Commanded loading ramp, (start, end) in seconds.
        Falls back to (nan, nan)
        """
        starts = self.header.get("_step_start_s") or []
        ends = self.header.get("_step_end_s") or []
        if not starts or not ends:
            return (float("nan"), float("nan"))
        return (float(starts[0]), float(ends[0]))

    @property
    def profile_ramp_nm(self) -> float:
        return _as_float(self.header.get("D[Z1] (nm)"))

    @property
    def vendor_s0_nm(self) -> float:
        """The instrument's contact coordinate
        """
        contact = self.vendor_indent_nm > 0.0
        if int(contact.sum()) < 3:
            return float("nan")
        offsets = self.s_nm[contact] - self.vendor_indent_nm[contact]
        return float(np.median(offsets))

    @property
    def vendor_s0_spread_nm(self) -> float:
        """Robust spread of s0 """
        contact = self.vendor_indent_nm > 0.0
        if int(contact.sum()) < 3:
            return float("nan")
        offsets = self.s_nm[contact] - self.vendor_indent_nm[contact]
        return float(np.max(offsets) - np.min(offsets))


def _as_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(str(value).strip().replace(",", "."))
    except ValueError:
        return float("nan")


def _clean(line: str) -> str:
    return line.lstrip("\r").lstrip(_BOM)


def parse_header(lines: list[str]) -> dict:
    """Turn the free-form header block into a flat dictionary
    """
    head: dict = {}
    for raw in lines:
        line = _clean(raw)
        if not line.strip():
            continue
        if line.startswith(_TABLE_MARK):
            break
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) >= 2 and parts[0]:
            # Date/profile lines carry several key-value pairs in one row.
            for key, value in zip(parts[0::2], parts[1::2], strict=False):
                if key:
                    head.setdefault(key, value)
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            head.setdefault(key.strip(), value.strip())
            continue
   
        match = re.match(r"^(.*?)\s{1,}([-+0-9.,eE]+)\s*$", line)
        if match:
            head.setdefault(match.group(1).strip(), match.group(2).strip())

    head["_step_start_s"] = _time_list(
        head.get("Step absolute start times (s)")
    )
    head["_step_end_s"] = _time_list(head.get("Step absolute end times (s)"))
    return head


def _time_list(value) -> list[float]:
    if not value:
        return []
    out: list[float] = []
    for token in str(value).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError:
            return []
    return out


def read_chiaro(path: str | Path) -> RawCurve:
    """Parse each file """
    path = Path(path)
    text = path.read_bytes().decode("utf-8", "replace")
    lines = text.splitlines()
    stop = None
    for index, raw in enumerate(lines):
        if _clean(raw).startswith(_TABLE_MARK):
            stop = index
            break
    if stop is None:
        raise ValueError(f"no data table found in {path}")
    header = parse_header(lines[:stop])
    body = "\n".join(lines[stop + 1 :]).strip()
    if not body:
        raise ValueError(f"empty data table in {path}")
    arr = np.loadtxt(io.StringIO(body), delimiter="\t", ndmin=2)
    if arr.shape[1] < 5:
        raise ValueError(f"expected >=5 columns in {path}, got {arr.shape[1]}")
    aux = arr[:, 5] if arr.shape[1] > 5 else np.full(arr.shape[0], np.nan)
    return RawCurve(
        path=path,
        header=header,
        time_s=arr[:, 0],
        load_uN=arr[:, 1],
        vendor_indent_nm=arr[:, 2],
        cantilever_nm=arr[:, 3],
        piezo_nm=arr[:, 4],
        auxiliary=aux,
    )


def campaign_curve_paths(datasets_root: Path, campaign: str) -> list[Path]:
    """Every ``S-n X-nn Y-nn I-nn.txt`` in a campaign, in a stable order."""
    root = Path(datasets_root) / campaign
    out: list[Path] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        out.extend(
            sorted(p for p in folder.iterdir() if CURVE_RE.match(p.name))
        )
    return out


def curve_identity(path: Path, campaign: str) -> dict:
    """The identifying columns every downstream table shares."""
    match = CURVE_RE.match(path.name)
    if match is None:
        raise ValueError(f"not a curve file: {path.name}")
    sample_id = path.parent.name.replace("matrix_", "")
    condition, _, gel = sample_id.rpartition("_")
    try:
        condition_value: float = int(condition)
    except ValueError:
        # Synthetic folders are named for their scenario, not a
        # concentration; keep the label and leave the numeric field NaN.
        condition_value = float("nan")
    try:
        gel_number = int(gel)
    except ValueError:
        gel_number = 0
    return {
        "campaign": campaign,
        "sample_id": sample_id,
        "condition_mg_ml": condition_value,
        "condition_label": condition,
        "gel_number": gel_number,
        "is_synthetic": bool(match.group("synth")),
        "x_index": int(match.group("x")),
        "y_index": int(match.group("y")),
        "curve_id": f"{campaign}/{sample_id}/{path.stem}",
        "relative_path": f"{path.parent.name}/{path.name}",
    }


def iter_campaign(datasets_root: Path, campaign: str):
    """Yield ``(identity, RawCurve)`` for a whole campaign."""
    for path in campaign_curve_paths(datasets_root, campaign):
        yield curve_identity(path, campaign), read_chiaro(path)
