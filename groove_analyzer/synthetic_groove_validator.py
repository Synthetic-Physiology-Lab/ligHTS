#!/usr/bin/env python3
"""
Synthetic Groove Validator v2.1 - Automated validation of groove analyzer against synthetic datasets.

Validates analyzer outputs (pitch, depth, angle, gel height) against ground truth from synthetic
generator with GUM-compliant uncertainty thresholds and ISO-compliant acceptance criteria.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

LOG = logging.getLogger("synth_groove_validation")


@dataclass(frozen=True)
class ImagingConfig:
    """Microscope calibration parameters matching synthetic dataset generation."""

    xy_um: float = 1.34
    dz_um: float = 0.9


@dataclass(frozen=True)
class DatasetCalibration:
    """Extracted calibration from dataset_metadata.json for analyzer invocation."""

    xy_um: float
    dz_um: float
    z0_um_offset: float


@dataclass(frozen=True)
class ValidationConfig:
    """
    ISO 25178 and GUM inspired acceptance thresholds for metrology validation.
    """

    pitch_mape_thresh_pct: float = 5.0
    depth_mape_thresh_pct: float = 5.0
    height_mape_thresh_pct: float = 5.0
    angle_err_thresh_deg: float = 1.0
    height_rmse_thresh_um: float = 1.8
    gel_height_abs_tol_um: float = 1.8


@dataclass(frozen=True)
class SyntheticSample:
    """Single synthetic sample metadata from truth_index.csv."""

    stack_code: str
    file_path: Path
    folder_path: Path
    pitch_realized_mean_um: float
    depth_realized_mean_um: float
    groove_angle_deg: float
    scenario: str
    z_min_um: float
    hydrogel_height_um: float


@dataclass
class VerificationResult:
    """
    Per-sample validation result with truth/measured values and pass/fail flags.
    Includes error propagation metrics (MAPE, absolute error) for QC analysis.
    """

    stack_code: str
    scenario: str
    passed: bool
    pitch_truth_um: float
    depth_truth_um: float
    angle_truth_deg: float
    gel_height_truth_um: float
    pitch_meas_um: float
    depth_meas_um: float
    angle_meas_deg: float
    gel_height_meas_um: float
    pitch_mape_pct: float
    depth_mape_pct: float
    pitch_abs_err_um: float
    depth_abs_err_um: float
    angle_err_deg: float
    gel_height_abs_err_um: float
    height_mape_pct: float
    height_rmse_um: float
    pitch_pass: bool
    depth_pass: bool
    angle_pass: bool
    gel_height_pass: bool
    height_pass: bool
    recon_valid_frac: float
    recon_valid_pass: bool
    error: str = ""


def read_truth_index(truth_csv: Path) -> list[SyntheticSample]:
    """
    Parse truth_index.csv containing ground truth for all synthetic samples.
    Returns list of SyntheticSample with file paths resolved relative to truth_csv location.
    """
    base_dir = truth_csv.parent
    required = {
        "stack_code",
        "file_rel_path",
        "folder_rel_path",
        "pitch_realized_mean_um",
        "depth_realized_mean_um",
        "groove_angle_deg",
        "hydrogel_height_um",
    }

    out: list[SyntheticSample] = []
    with open(truth_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError("truth_index.csv has no header row.")
        missing = required - set(r.fieldnames)
        if missing:
            raise ValueError(f"truth_index.csv missing columns: {sorted(missing)}")

        for row in r:
            file_rel = str(row.get("file_rel_path", "")).strip()
            folder_rel = str(row.get("folder_rel_path", "")).strip()
            if not file_rel or not folder_rel:
                raise ValueError(
                    "truth_index.csv must contain file_rel_path and folder_rel_path"
                )

            out.append(
                SyntheticSample(
                    stack_code=str(row["stack_code"]).strip(),
                    file_path=(base_dir / file_rel),
                    folder_path=(base_dir / folder_rel),
                    pitch_realized_mean_um=float(row["pitch_realized_mean_um"]),
                    depth_realized_mean_um=float(row["depth_realized_mean_um"]),
                    groove_angle_deg=float(row["groove_angle_deg"]),
                    scenario=str(row.get("scenario", "")).strip(),
                    z_min_um=float(row.get("z_min_um", "nan")),
                    hydrogel_height_um=float(row["hydrogel_height_um"]),
                )
            )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write list of dicts to CSV with automatic field ordering from first row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("note\nno rows\n")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def read_dataset_metadata(dataset_dir: Path) -> DatasetCalibration:
    """
    Extract xy_um, dz_um, z0_um from dataset_metadata.json if present.
    Falls back to default ImagingConfig values if metadata missing.
    """
    metadata_path = dataset_dir / "dataset_metadata.json"
    if not metadata_path.exists():
        return DatasetCalibration(
            xy_um=ImagingConfig().xy_um,
            dz_um=ImagingConfig().dz_um,
            z0_um_offset=0.0,
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid dataset_metadata.json: {e}") from e

    xy_um = float(metadata.get("xy_um_per_px", ImagingConfig().xy_um))
    dz_um = float(metadata.get("dz_um_per_slice", ImagingConfig().dz_um))
    z0_um_offset = float(metadata.get("z0_um_offset", 0.0))
    return DatasetCalibration(xy_um=xy_um, dz_um=dz_um, z0_um_offset=z0_um_offset)


try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:
    tk = None
    filedialog = None
    messagebox = None


def tk_root() -> tk.Tk:
    """Create and return hidden Tkinter root window for dialogs."""
    if tk is None:
        raise RuntimeError(
            "tkinter is not available (install Tk or run without --gui)."
        )
    root = tk.Tk()
    root.withdraw()
    root.update()
    return root


def gui_info(title: str, msg: str, *, kind: str = "info") -> None:
    """Display modal info or error dialog via Tkinter messagebox."""
    if messagebox is None:
        raise RuntimeError("tkinter messagebox is not available.")
    root = tk_root()
    try:
        if kind == "error":
            messagebox.showerror(title, msg, parent=root)
        else:
            messagebox.showinfo(title, msg, parent=root)
    finally:
        root.destroy()


def gui_pick_file(title: str) -> Path:
    """Open file picker dialog and return selected .py file path."""
    if filedialog is None:
        raise RuntimeError("tkinter filedialog is not available.")
    root = tk_root()
    try:
        p = filedialog.askopenfilename(
            title=title,
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
            parent=root,
        )
    finally:
        root.destroy()
    if not p:
        raise SystemExit("No file selected")
    return Path(p)


def gui_pick_dir(title: str) -> Path:
    """Open directory picker dialog and return selected folder path."""
    if filedialog is None:
        raise RuntimeError("tkinter filedialog is not available.")
    root = tk_root()
    try:
        p = filedialog.askdirectory(title=title, parent=root)
    finally:
        root.destroy()
    if not p:
        raise SystemExit("No folder selected")
    return Path(p)


def load_analyzer_module(target: Path) -> tuple[Any, str]:
    """
    Dynamically import analyzer module and verify API compatibility.
    Requires analyze_file(path, xy_um, dz_um, z0_um, ...) interface.
    """
    target = target.resolve()

    if target.is_dir():
        analysis_py = target / "analysis.py"
        if not analysis_py.exists():
            raise RuntimeError(f"{target} does not contain analysis.py")

        sys.modules.pop("analysis", None)

        if str(target) not in sys.path:
            sys.path.insert(0, str(target))
        importlib.invalidate_caches()
        try:
            mod = importlib.import_module("analysis")
        except Exception as e:
            raise RuntimeError(
                f"Failed to import analyzer from folder: {target}\n{e}"
            ) from e
        label = f"refactor:{target.name}"

    elif target.is_file() and target.suffix.lower() == ".py":
        spec = importlib.util.spec_from_file_location("groove_analyzer", str(target))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import analyzer: {target}")

        sys.modules.pop(spec.name, None)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            raise RuntimeError(f"Failed to import analyzer file: {target}\n{e}") from e
        label = f"monolith:{target.name}"

    else:
        raise RuntimeError(
            "Analyzer path must be a .py file or a folder containing analysis.py"
        )

    if not (
        (hasattr(mod, "analyze_tiff") and hasattr(mod, "Calibration"))
        or hasattr(mod, "analyze_file")
    ):
        raise RuntimeError(
            "Analyzer must expose analyze_tiff(...) + Calibration or analyze_file(...)."
        )
    return mod, label


def analyze_one(
    mod: Any,
    tif_path: Path,
    imaging: ImagingConfig,
    *,
    z0_um_offset: float,
    outdir: Optional[Path],
) -> tuple[dict[str, Any], Optional[str]]:
    """
    Invoke analyzer on single TIFF with appropriate API (analyze_tiff or analyze_file).
    Returns (results_dict, error_msg) where error_msg is None on success.
    """
    if hasattr(mod, "analyze_tiff") and hasattr(mod, "Calibration"):
        cal = mod.Calibration(
            xy_um_per_px=float(imaging.xy_um), dz_um_per_slice=float(imaging.dz_um)
        )

        kwargs: dict[str, Any] = dict(
            z0_um_offset=float(z0_um_offset),
            outdir=outdir,
            save_csv=True,
            save_arrays=False,
        )

        if hasattr(mod, "AnalysisConfig") and hasattr(mod, "PlotConfig"):
            try:
                kwargs["config"] = mod.AnalysisConfig(
                    plots=mod.PlotConfig(enabled=False)
                )
            except Exception:
                pass

        try:
            res_obj, _dist = mod.analyze_tiff(Path(tif_path), cal, **kwargs)
        except Exception as e:
            return {}, str(e)

        try:
            res = asdict(res_obj)
        except Exception:
            res = dict(getattr(res_obj, "__dict__", {}))
    elif hasattr(mod, "analyze_file"):
        kwargs = {}
        if hasattr(mod, "AnalyzerConfig"):
            try:
                kwargs["config"] = mod.AnalyzerConfig(generate_plots=False)
            except Exception:
                pass
        try:
            res = mod.analyze_file(
                Path(tif_path),
                float(imaging.xy_um),
                float(imaging.dz_um),
                float(z0_um_offset),
                None,
                outdir,
                **kwargs,
            )
        except Exception as e:
            return {}, str(e)

    else:
        return (
            {},
            "Analyzer module does not expose analyze_tiff(...) or analyze_file(...).",
        )

    res["pitch_meas_um"] = float(res.get("pitch_mean_um", res.get("pitch_um", np.nan)))
    res["depth_meas_um"] = float(res.get("depth_mean_um", res.get("depth_um", np.nan)))
    res["angle_meas_deg"] = float(res.get("fft_angle_deg", np.nan))
    res["gel_height_meas_um"] = float(
        res.get("mean_height_um", res.get("gel_height_um", np.nan))
    )
    return res, None


def compute_height_rmse(
    mod: Any,
    tif_path: Path,
    imaging: ImagingConfig,
    truth_npz: Path,
    z0_um_offset: float,
) -> float:
    """
    Calculate pixel-wise RMSE between reconstructed and ground truth height maps.
    Returns RMSE in µm or nan if reconstruction unavailable or shape mismatch.
    """
    if not truth_npz.exists():
        return float("nan")

    if hasattr(mod, "load_volume") and hasattr(mod, "reconstruct_height_map"):
        vol = mod.load_volume(Path(tif_path))
        h_meas, _ = mod.reconstruct_height_map(
            vol, float(imaging.dz_um), float(z0_um_offset)
        )
        h_meas = h_meas.astype(np.float32)
    elif hasattr(mod, "load_tiff_volume") and hasattr(
        mod, "reconstruct_height_map_slices"
    ):
        vol = mod.load_tiff_volume(Path(tif_path))
        z_sub = mod.reconstruct_height_map_slices(vol)
        h_meas = z_sub.astype(np.float32) * float(imaging.dz_um) + float(z0_um_offset)
    else:
        return float("nan")

    td = np.load(truth_npz)
    if "height_um" in td:
        if np.isfinite(z0_um_offset) and abs(float(z0_um_offset)) > 1e-9:
            h_truth = td["height_um"].astype(np.float32) + float(z0_um_offset)
        else:
            h_truth = td["height_rel_um"].astype(np.float32)
    else:
        h_truth = td["height_rel_um"].astype(np.float32)

    if h_meas.shape != h_truth.shape:
        return float("nan")

    valid = np.isfinite(h_meas) & np.isfinite(h_truth)
    if not np.any(valid):
        return float("nan")

    diff = h_meas[valid] - h_truth[valid]
    return float(np.sqrt(np.mean(diff**2)))


def mape_pct(meas: float, truth: float) -> float:
    """Mean Absolute Percentage Error: |meas - truth| / |truth| × 100%."""
    if not (np.isfinite(meas) and np.isfinite(truth) and truth != 0):
        return float("nan")
    return float(abs((meas - truth) / truth) * 100.0)


def angle_error_deg(meas: float, truth: float) -> float:
    """
    Angular error with 180° periodicity: min(|Δ|, 180° - |Δ|).
    Handles groove orientation ambiguity (0° ≡ 180°).
    """
    if not np.isfinite(meas) or not np.isfinite(truth):
        return float("nan")
    diff = abs(meas - truth) % 180.0
    return float(min(diff, 180.0 - diff))


def verify_one(
    mod: Any,
    sample: SyntheticSample,
    imaging: ImagingConfig,
    z0_um_offset: float,
    cfg: ValidationConfig,
    output_root: Path,
) -> VerificationResult:
    """
    Run analyzer on single sample and compare against ground truth.
    Returns VerificationResult with pass/fail flags per metric and combined overall pass.
    """
    tif_path = sample.folder_path / f"{sample.stack_code}.tif"
    if not tif_path.exists():
        tif_path = sample.file_path

    per_out = output_root / sample.stack_code
    per_out.mkdir(parents=True, exist_ok=True)

    res, err = analyze_one(
        mod,
        tif_path,
        imaging,
        z0_um_offset=z0_um_offset,
        outdir=per_out,
    )

    if err is not None:
        return VerificationResult(
            stack_code=sample.stack_code,
            scenario=sample.scenario,
            passed=False,
            pitch_truth_um=sample.pitch_realized_mean_um,
            depth_truth_um=sample.depth_realized_mean_um,
            angle_truth_deg=sample.groove_angle_deg,
            gel_height_truth_um=sample.hydrogel_height_um,
            pitch_meas_um=float("nan"),
            depth_meas_um=float("nan"),
            angle_meas_deg=float("nan"),
            gel_height_meas_um=float("nan"),
            pitch_mape_pct=float("nan"),
            depth_mape_pct=float("nan"),
            pitch_abs_err_um=float("nan"),
            depth_abs_err_um=float("nan"),
            angle_err_deg=float("nan"),
            gel_height_abs_err_um=float("nan"),
            height_mape_pct=float("nan"),
            height_rmse_um=float("nan"),
            pitch_pass=False,
            depth_pass=False,
            angle_pass=False,
            gel_height_pass=False,
            height_pass=False,
            recon_valid_frac=float("nan"),
            recon_valid_pass=False,
            error=err,
        )

    pitch_meas = float(res.get("pitch_meas_um", np.nan))
    depth_meas = float(res.get("depth_meas_um", np.nan))
    angle_meas = float(res.get("angle_meas_deg", np.nan))
    gel_meas = float(res.get("gel_height_meas_um", np.nan))
    recon_valid_frac = float(res.get("recon_valid_frac", np.nan))

    truth_npz = sample.folder_path / f"{sample.stack_code}_truth_height_um.npz"
    height_rmse = compute_height_rmse(mod, tif_path, imaging, truth_npz, z0_um_offset)

    pitch_mape = mape_pct(pitch_meas, sample.pitch_realized_mean_um)
    depth_mape = mape_pct(depth_meas, sample.depth_realized_mean_um)
    height_mape = mape_pct(gel_meas, sample.hydrogel_height_um)
    angle_err = angle_error_deg(angle_meas, sample.groove_angle_deg)

    pitch_abs_err = (
        abs(pitch_meas - sample.pitch_realized_mean_um)
        if np.isfinite(pitch_meas)
        else float("nan")
    )
    depth_abs_err = (
        abs(depth_meas - sample.depth_realized_mean_um)
        if np.isfinite(depth_meas)
        else float("nan")
    )
    gel_abs_err = (
        abs(gel_meas - sample.hydrogel_height_um)
        if np.isfinite(gel_meas)
        else float("nan")
    )

    pitch_abs_tol_um = 2.0 * imaging.xy_um
    depth_abs_tol_um = 2.0 * imaging.dz_um

    pitch_pass = bool(
        np.isfinite(pitch_meas)
        and (
            (np.isfinite(pitch_mape) and pitch_mape <= cfg.pitch_mape_thresh_pct)
            or (np.isfinite(pitch_abs_err) and pitch_abs_err <= pitch_abs_tol_um)
        )
    )
    depth_pass = bool(
        np.isfinite(depth_meas)
        and (
            (np.isfinite(depth_mape) and depth_mape <= cfg.depth_mape_thresh_pct)
            or (np.isfinite(depth_abs_err) and depth_abs_err <= depth_abs_tol_um)
        )
    )
    angle_pass = bool(np.isfinite(angle_err) and angle_err <= cfg.angle_err_thresh_deg)
    gel_pass = bool(
        (np.isfinite(gel_abs_err) and gel_abs_err <= cfg.gel_height_abs_tol_um)
        or (np.isfinite(gel_abs_err) and height_mape <= cfg.height_mape_thresh_pct)
    )

    height_available = np.isfinite(height_rmse)
    height_pass = bool(
        (not height_available) or (height_rmse <= cfg.height_rmse_thresh_um)
    )

    if "recon_valid_frac" in res:
        recon_valid_pass = bool(
            np.isfinite(recon_valid_frac) and recon_valid_frac > 0.0
        )
    else:
        recon_valid_pass = True
    passed = bool(
        pitch_pass
        and depth_pass
        and angle_pass
        and gel_pass
        and height_pass
        and recon_valid_pass
    )

    return VerificationResult(
        stack_code=sample.stack_code,
        scenario=sample.scenario,
        passed=passed,
        pitch_truth_um=sample.pitch_realized_mean_um,
        depth_truth_um=sample.depth_realized_mean_um,
        angle_truth_deg=sample.groove_angle_deg,
        gel_height_truth_um=sample.hydrogel_height_um,
        pitch_meas_um=pitch_meas,
        depth_meas_um=depth_meas,
        angle_meas_deg=angle_meas,
        gel_height_meas_um=gel_meas,
        pitch_mape_pct=pitch_mape,
        depth_mape_pct=depth_mape,
        pitch_abs_err_um=float(pitch_abs_err),
        depth_abs_err_um=float(depth_abs_err),
        angle_err_deg=float(angle_err),
        gel_height_abs_err_um=gel_abs_err,
        height_mape_pct=height_mape,
        height_rmse_um=float(height_rmse),
        pitch_pass=pitch_pass,
        depth_pass=depth_pass,
        angle_pass=angle_pass,
        gel_height_pass=gel_pass,
        height_pass=height_pass,
        recon_valid_frac=recon_valid_frac,
        recon_valid_pass=recon_valid_pass,
        error="",
    )


def compute_statistics(values: list[float]) -> dict[str, float]:
    """Compute mean, std, max for finite values in list."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "max": float(np.max(a)),
    }


def summarize_results(
    results: list[VerificationResult], cfg: ValidationConfig
) -> dict[str, Any]:
    """
    Aggregate validation results across all samples: pass rates, error statistics, scenario breakdown.
    Returns summary dict for JSON export and report generation.
    """
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    errors = sum(1 for r in results if bool(r.error))
    valid = [r for r in results if not r.error]

    def col(attr: str) -> list[float]:
        return [float(getattr(r, attr)) for r in valid]

    scenarios = sorted({r.scenario for r in results})
    scenario_breakdown: dict[str, Any] = {}
    for sc in scenarios:
        rs = [r for r in results if r.scenario == sc]
        sp = sum(1 for r in rs if r.passed)
        scenario_breakdown[sc] = {
            "total": len(rs),
            "passed": sp,
            "pass_rate": (sp / len(rs)) if rs else 0.0,
        }

    return {
        "total_samples": total,
        "passed": passed,
        "failed": total - passed,
        "errors": errors,
        "overall_pass_rate": (passed / total) if total else 0.0,
        "pitch_mape": compute_statistics(col("pitch_mape_pct")),
        "depth_mape": compute_statistics(col("depth_mape_pct")),
        "pitch_abs_err_um": compute_statistics(col("pitch_abs_err_um")),
        "depth_abs_err_um": compute_statistics(col("depth_abs_err_um")),
        "angle_err_deg": compute_statistics(col("angle_err_deg")),
        "gel_height_abs_err_um": compute_statistics(col("gel_height_abs_err_um")),
        "height_mape": compute_statistics(col("height_mape_pct")),
        "height_rmse_um": compute_statistics(col("height_rmse_um")),
        "scenario_breakdown": scenario_breakdown,
        "thresholds": asdict(cfg),
    }


def write_report(
    dataset_dir: Path, summary: dict[str, Any], results: list[VerificationResult]
) -> Path:
    """
    Generate markdown validation report with executive summary and failed samples table.
    Returns path to generated VALIDATION_REPORT.md file.
    """
    report = dataset_dir / "VALIDATION_REPORT.md"
    failed = [r for r in results if not r.passed]
    th = summary["thresholds"]

    with open(report, "w", encoding="utf-8") as f:
        f.write("# Groove Analyzer Validation Report\n\n")
        f.write(f"**Generated:** {summary.get('timestamp','')}\n\n")
        f.write(f"**Analyzer:** {summary.get('analyzer','')}\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"- Total samples: {summary['total_samples']}\n")
        f.write(f"- Passed: {summary['passed']} ({summary['overall_pass_rate']:.1%})\n")
        f.write(f"- Failed: {summary['failed']}\n")
        f.write(f"- Analyzer errors: {summary['errors']}\n\n")

        f.write("## Acceptance Criteria\n\n")
        f.write("| Metric | Threshold |\n|---|---:|\n")
        f.write(f"| Pitch MAPE | ≤ {th['pitch_mape_thresh_pct']}% |\n")
        f.write(f"| Depth MAPE | ≤ {th['depth_mape_thresh_pct']}% |\n")
        f.write(f"| Angle error | ≤ {th['angle_err_thresh_deg']}° |\n")
        f.write(f"| Gel height |abs err| | ≤ {th['gel_height_abs_tol_um']} µm |\n")
        f.write(f"| Height MAPE | ≤ {th['height_mape_thresh_pct']}% |\n")
        f.write(f"| Height RMSE | ≤ {th['height_rmse_thresh_um']} µm |\n\n")

        f.write("## Failed Samples (first 25)\n\n")
        if not failed:
            f.write("No failed samples.\n")
        else:
            f.write(
                "| Stack | Scenario | Pitch MAPE | Depth MAPE | Angle err | "
                "Gel |abs err| | Height MAPE | Height RMSE | Reason |\n"
            )
            f.write("|---|---|---:|---:|---:|---:|---:|---:|---|\n")
            for r in failed[:25]:
                reason = (r.error or "threshold exceeded").replace("\n", " ")[:80]
                f.write(
                    f"| {r.stack_code} | {r.scenario} | {r.pitch_mape_pct:.2f} | "
                    f"{r.depth_mape_pct:.2f} | {r.angle_err_deg:.2f} | "
                    f"{r.gel_height_abs_err_um:.2f} | {r.height_mape_pct:.2f}|{r.height_rmse_um:.3f} | {reason} |\n"
                )

    return report


def log_summary(summary: dict[str, Any]) -> None:
    """Print validation summary to logger with formatted statistics tables."""
    LOG.info("=" * 60)
    LOG.info("VERIFICATION SUMMARY")
    LOG.info("=" * 60)
    LOG.info("Analyzer: %s", summary.get("analyzer", ""))
    LOG.info("Total samples: %d", summary.get("total_samples", 0))
    LOG.info(
        "Passed: %d (%.1f%%)",
        summary.get("passed", 0),
        100.0 * float(summary.get("overall_pass_rate", 0.0)),
    )
    LOG.info("Failed: %d", summary.get("failed", 0))
    LOG.info("Errors: %d", summary.get("errors", 0))
    LOG.info("-" * 60)

    def stat_dict(key: str) -> dict[str, float]:
        s = summary.get(key, {})
        return {
            "mean": float(s.get("mean", float("nan"))),
            "std": float(s.get("std", float("nan"))),
            "max": float(s.get("max", float("nan"))),
        }

    pitch_mape = stat_dict("pitch_mape")
    depth_mape = stat_dict("depth_mape")
    pitch_abs = stat_dict("pitch_abs_err_um")
    depth_abs = stat_dict("depth_abs_err_um")
    angle_err = stat_dict("angle_err_deg")
    gel_abs = stat_dict("gel_height_abs_err_um")
    height_mape = stat_dict("height_mape")
    height_rmse = stat_dict("height_rmse_um")

    LOG.info(
        "Pitch MAPE: %.1f%% ± %.1f%% (max: %.1f%%)",
        pitch_mape["mean"],
        pitch_mape["std"],
        pitch_mape["max"],
    )
    LOG.info(
        "Pitch |abs err|: %.1f ± %.1f µm (max: %.1f µm)",
        pitch_abs["mean"],
        pitch_abs["std"],
        pitch_abs["max"],
    )
    LOG.info(
        "Depth MAPE: %.1f%% ± %.1f%% (max: %.1f%%)",
        depth_mape["mean"],
        depth_mape["std"],
        depth_mape["max"],
    )
    LOG.info(
        "Depth |abs err|: %.1f ± %.1f µm (max: %.1f µm)",
        depth_abs["mean"],
        depth_abs["std"],
        depth_abs["max"],
    )
    LOG.info("Angle error: %.1f° ± %.1f°", angle_err["mean"], angle_err["std"])
    LOG.info(
        "Gel height |abs err|: %.1f ± %.1f µm (max: %.1f µm)",
        gel_abs["mean"],
        gel_abs["std"],
        gel_abs["max"],
    )
    LOG.info("Height RMSE: %.1f ± %.1f µm", height_rmse["mean"], height_rmse["std"])
    LOG.info(
        "Height MAPE: %.1f%% ± %.1f%% (max: %.1f%%)",
        height_mape["mean"],
        height_mape["std"],
        height_mape["max"],
    )
    LOG.info("-" * 60)
    LOG.info("Per-scenario pass rates:")
    sc = summary.get("scenario_breakdown", {}) or {}
    for name in sorted(sc.keys()):
        st = sc.get(name, {}) or {}
        total = int(st.get("total", 0))
        passed = int(st.get("passed", 0))
        rate = 100.0 * float(st.get("pass_rate", 0.0)) if total else 0.0
        label = name if str(name).strip() else "<unspecified>"
        LOG.info("  %s: %d/%d (%.1f%%)", label, passed, total, rate)
    LOG.info("=" * 60)
    report_path = summary.get("report_path")
    if report_path:
        LOG.info("Validation report written to: %s", report_path)


def run_validation(
    dataset_dir: Path, analyzer_path: Path
) -> tuple[list[VerificationResult], dict[str, Any]]:
    """
    Execute full validation workflow: load dataset, run analyzer on all samples, compare to truth.
    Returns (results_list, summary_dict) and writes verification_results.csv, summary.json, report.md.
    """
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {dataset_dir}")

    truth = dataset_dir / "truth_index.csv"
    if not truth.exists():
        raise FileNotFoundError(f"Could not find truth_index.csv in {dataset_dir}")

    samples = read_truth_index(truth)
    if not samples:
        raise RuntimeError("truth_index.csv contained 0 samples.")

    mod, label = load_analyzer_module(analyzer_path)

    calibration = read_dataset_metadata(dataset_dir)
    imaging = ImagingConfig(xy_um=calibration.xy_um, dz_um=calibration.dz_um)
    cfg = ValidationConfig()
    z0_um_offset = calibration.z0_um_offset

    out_root = dataset_dir / "analysis_outputs"
    out_root.mkdir(parents=True, exist_ok=True)

    LOG.info("Running validation with analyzer: %s", label)

    results: list[VerificationResult] = []
    for i, s in enumerate(samples, 1):
        results.append(verify_one(mod, s, imaging, z0_um_offset, cfg, out_root))
        if i % 20 == 0 or i == len(samples):
            n_pass = sum(1 for x in results if x.passed)
            LOG.info("Verified %d/%d samples (%d passed)", i, len(samples), n_pass)

    summary = summarize_results(results, cfg)
    summary["analyzer"] = label
    summary["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary.setdefault("thresholds", {})
    summary["thresholds"]["pitch_abs_err_thresh_um"] = 2.0 * float(imaging.xy_um)
    summary["thresholds"]["depth_abs_err_thresh_um"] = 2.0 * float(imaging.dz_um)
    summary["thresholds"]["gel_abs_err_thresh_um"] = float(cfg.gel_height_abs_tol_um)
    summary["thresholds"]["height_rmse_thresh_um"] = float(cfg.height_rmse_thresh_um)
    summary["thresholds"]["angle_err_thresh_deg"] = float(cfg.angle_err_thresh_deg)

    write_csv(dataset_dir / "verification_results.csv", [asdict(r) for r in results])
    with open(dataset_dir / "verification_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    report = write_report(dataset_dir, summary, results)
    summary["report_path"] = str(report.resolve())

    log_summary(summary)
    return results, summary


def setup_logging(level: str) -> None:
    """Configure logging with specified level and stdout handler."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main_gui() -> None:
    """Launch GUI workflow: prompt for analyzer and dataset selection via dialogs."""
    setup_logging("INFO")
    gui_info(
        "Synthetic Groove Validation",
        "You will be asked to:\n"
        "1) Select the analyzer (.py)\n"
        "2) Select the dataset folder (must contain truth_index.csv)\n",
    )
    analyzer = gui_pick_file("Select analyzer (.py)")
    ds = gui_pick_dir("Select dataset folder (containing truth_index.csv)")
    try:
        run_validation(ds, analyzer)
    except Exception as e:
        gui_info("Validation error", str(e), kind="error")


def main_cli(dataset_dir: str, analyzer_path: str, *, log_level: str) -> None:
    """CLI entry point: run validation with specified dataset and analyzer paths."""
    setup_logging(log_level)
    try:
        run_validation(Path(dataset_dir), Path(analyzer_path))
    except Exception:
        LOG.exception("Validation failed")
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Validate a groove analyzer against an existing synthetic dataset",
        epilog=f"Example (CLI): python {Path(__file__).name} -a groove_analyzer.py -d C:\\path\\to\\dataset",
    )
    p.add_argument("--gui", action="store_true", help="Use GUI dialogs")
    p.add_argument(
        "-a",
        "--analyzer-path",
        type=str,
        help="Path to analyzer (.py) or folder containing analysis.py",
    )
    p.add_argument(
        "-d",
        "--dataset-dir",
        type=str,
        help="Dataset directory containing truth_index.csv",
    )
    p.add_argument(
        "--log", type=str, default="INFO", help="Logging level (INFO, WARNING, ERROR)"
    )
    args = p.parse_args()

    if args.gui:
        main_gui()
    else:
        if not args.analyzer_path or not args.dataset_dir:
            raise SystemExit(
                "Must specify --analyzer-path and --dataset-dir (or use --gui)"
            )
        main_cli(args.dataset_dir, args.analyzer_path, log_level=args.log)
