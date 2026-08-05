"""
Synthetic Flat Gel Validator - Geometric Parameter Validation.

Validates gel surface analyzer against synthetic datasets by comparing:
  - Gel height (TopZ_LFilteredMean_um)
  - X-tilt (Plane_a_um_per_mm)
  - Y-tilt (Plane_b_um_per_mm)
  - Combined tilt angle (TiltAngle_deg)

Usage:
  CLI mode:
    python synthetic_flat_gel_validator.py -a analyzer.py -d ./dataset

  GUI mode:
    python synthetic_flat_gel_validator.py --gui
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib
import importlib.util
import inspect
import json
import logging
import os
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger("synth_flat_gel_validator")


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass(frozen=True)
class FieldSpec:
    """
    Specification for a single validation field.

    Defines field name, tolerances, and data type for comparison.
    """

    name: str
    abs_tol: float | None = None
    rel_tol: float | None = None
    kind: str = "float"


@dataclass(frozen=True)
class ValidationConfig:
    """
    Validation configuration for geometric parameters.

    Defines which fields to validate and their acceptance tolerances.
    Focuses on core geometric parameters critical for gel characterization.
    """

    fields: tuple[FieldSpec, ...] = (
        FieldSpec("Plane_a_um_per_mm", abs_tol=3.0, kind="float"),
        FieldSpec("Plane_b_um_per_mm", abs_tol=3.0, kind="float"),
        FieldSpec("TiltAngle_deg", abs_tol=0.5, kind="float"),
        FieldSpec("TopZ_LFilteredMean_um", abs_tol=2.0, kind="float"),
    )
    optional_fields: tuple[FieldSpec, ...] = tuple()


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class TruthRow:
    """
    Single row from ground truth CSV.

    Contains filename and all metadata/ground truth values for one sample.
    """

    file: str
    row: dict[str, Any]

    @property
    def stem(self) -> str:
        """Extract filename stem without extension."""
        return Path(self.file).stem


@dataclass
class FieldResult:
    """
    Validation result for a single field.

    Stores ground truth, measured value, errors, and pass/fail status.
    """

    name: str
    truth: Any
    meas: Any
    abs_err: float | None
    rel_err: float | None
    passed: bool
    skipped: bool
    note: str = ""


@dataclass
class SampleResult:
    """
    Complete validation result for one sample.

    Aggregates all field results and overall pass/fail status.
    """

    file: str
    scenario: str
    passed: bool
    error: str
    fields: list[FieldResult]


# =============================================================================
# File I/O Functions
# =============================================================================


def read_truth_metrics_csv(csv_path: Path) -> list[TruthRow]:
    """
    Load ground truth CSV file.

    Reads CSV with 'File' column and ground truth parameters.
    Returns list of TruthRow objects, one per sample.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing truth metrics CSV: {csv_path}")

    rows: list[TruthRow] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"{csv_path.name} has no header row")
        if "File" not in reader.fieldnames:
            raise ValueError(f"{csv_path.name} must contain a 'File' column")

        for rec in reader:
            file_name = str(rec.get("File", "")).strip()
            if file_name:
                rows.append(TruthRow(file=file_name, row=dict(rec)))

    if not rows:
        raise ValueError(f"{csv_path.name} contained 0 usable rows")

    return rows


def write_validation_csv(output_path: Path, results: list[SampleResult]) -> None:
    """
    Write detailed validation results to CSV.

    Each row contains per-field comparisons for one sample.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = flatten_results(results)
    if not rows:
        output_path.write_text("note\nno rows\n", encoding="utf-8")
        return

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(
    output_path: Path, summary: dict[str, Any], results: list[SampleResult]
) -> None:
    """
    Generate human-readable validation report in Markdown.

    Includes summary statistics, error analysis, and failed samples.
    """
    failed = [r for r in results if not r.passed]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Synthetic Flat Gel Validation Report\n\n")
        f.write(f"- Timestamp: {summary.get('timestamp', '')}\n")
        f.write(f"- Analyzer: {summary.get('analyzer', '')}\n")
        f.write(f"- Truth CSV: {summary.get('truth_csv', '')}\n\n")

        f.write("## Summary\n\n")
        f.write(f"- Total: {summary['total']}\n")
        f.write(f"- Passed: {summary['passed']} ({summary['pass_rate']:.1%})\n")
        f.write(f"- Failed: {summary['failed']}\n")
        f.write(f"- Analyzer errors: {summary['errors']}\n\n")

        f.write("## Criteria\n\n")
        f.write("| Field | abs_tol | rel_tol | kind |\n")
        f.write("|---|---:|---:|---|\n")
        for spec in summary.get("criteria", []):
            f.write(
                f"| {spec.get('name', '')} | "
                f"{spec.get('abs_tol', '')} | "
                f"{spec.get('rel_tol', '')} | "
                f"{spec.get('kind', '')} |\n"
            )
        f.write("\n")

        f.write("## Field Error Stats (Non-Skipped)\n\n")
        f.write("| Field | abs mean | abs max | rel mean | rel max |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for field_name, stats in (summary.get("field_stats", {}) or {}).items():
            f.write(
                f"| {field_name} | "
                f"{stats['abs_err_mean']:.4g} | "
                f"{stats['abs_err_max']:.4g} | "
                f"{stats['rel_err_mean']:.4g} | "
                f"{stats['rel_err_max']:.4g} |\n"
            )
        f.write("\n")

        f.write("## Per-Scenario Pass Rate\n\n")
        f.write("| Scenario | Total | Passed | Pass rate |\n")
        f.write("|---|---:|---:|---:|\n")
        for scenario, stats in (summary.get("scenario_breakdown", {}) or {}).items():
            f.write(
                f"| {scenario} | "
                f"{stats['total']} | "
                f"{stats['passed']} | "
                f"{stats['pass_rate']:.1%} |\n"
            )
        f.write("\n")

        f.write("## Failed Samples (First 25)\n\n")
        if not failed:
            f.write("None.\n")
        else:
            f.write("| File | Scenario | Error | First failing field |\n")
            f.write("|---|---|---|---|\n")
            for result in failed[:25]:
                first_fail = ""
                for field_res in result.fields:
                    if not field_res.skipped and not field_res.passed:
                        first_fail = f"{field_res.name} ({field_res.note})"
                        break
                error_msg = (result.error or "threshold exceeded").replace("\n", " ")
                error_msg = error_msg[:120]
                f.write(f"| {result.file} | {result.scenario} | " f"{error_msg} | {first_fail} |\n")


def write_validation_outputs(
    dataset_dir: Path,
    results: list[SampleResult],
    summary: dict[str, Any],
    out_root: Path,
) -> None:
    """
    Write all validation output files.

    Creates CSV, JSON, and Markdown reports in dataset directory.
    """
    write_validation_csv(dataset_dir / "validation_results.csv", results)

    with open(dataset_dir / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = dataset_dir / "VALIDATION_REPORT.md"
    write_markdown_report(report_path, summary, results)

    LOG.info("Wrote: %s", report_path)
    LOG.info("Wrote: %s", dataset_dir / "validation_results.csv")
    LOG.info("Wrote: %s", dataset_dir / "validation_summary.json")
    LOG.info("Per-stack analyzer outputs under: %s", out_root)


# =============================================================================
# Type Conversion Helpers
# =============================================================================


def convert_to_float(value: Any) -> float:
    """
    Convert value to float, returning NaN for invalid values.

    Handles None, empty strings, and invalid conversions gracefully.
    """
    if value is None:
        return float("nan")
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)

    string_val = str(value).strip()
    if string_val == "" or string_val.lower() in {"nan", "none", "null"}:
        return float("nan")

    try:
        return float(string_val)
    except (ValueError, TypeError):
        return float("nan")


def convert_to_bool(value: Any) -> bool | None:
    """
    Convert value to bool, returning None for invalid values.

    Recognizes common boolean representations (true/false, 1/0, yes/no).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value

    string_val = str(value).strip().lower()
    if string_val in {"true", "1", "yes", "y"}:
        return True
    if string_val in {"false", "0", "no", "n"}:
        return False
    return None


def convert_to_str(value: Any) -> str:
    """Convert value to string, returning empty string for None."""
    return "" if value is None else str(value)


# =============================================================================
# Analyzer Loading and Execution
# =============================================================================


def get_utc_timestamp_iso() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def compute_file_sha256(file_path: Path) -> str:
    """
    Compute SHA256 hash of file.

    Reads file in chunks to handle large files efficiently.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_analyzer_module(analyzer_path: Path, *, disable_pngs: bool) -> tuple[Any, str, str]:
    """
    Load analyzer module from .py file or package directory.

    Sets environment variables to disable PNG output if requested.
    Returns (module, label, sha256_hash).
    """
    analyzer_path = analyzer_path.resolve()

    if disable_pngs:
        os.environ.setdefault("GEL_SURF_SAVE_PNGS", "0")

    if analyzer_path.is_dir():
        return load_analyzer_from_directory(analyzer_path, disable_pngs)
    elif analyzer_path.is_file() and analyzer_path.suffix.lower() == ".py":
        return load_analyzer_from_file(analyzer_path, disable_pngs)
    else:
        raise RuntimeError("Analyzer path must be a .py file or a directory containing analysis.py")


def load_analyzer_from_directory(directory: Path, disable_pngs: bool) -> tuple[Any, str, str]:
    """Load analyzer from package directory containing analysis.py."""
    analysis_py = directory / "analysis.py"
    if not analysis_py.exists():
        raise RuntimeError(f"{directory} does not contain analysis.py")

    sys.modules.pop("analysis", None)
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    importlib.invalidate_caches()

    try:
        module = importlib.import_module("analysis")
    except Exception as e:
        raise RuntimeError(f"Failed to import analyzer from folder: {directory}\n{e}") from e

    validate_and_configure_analyzer(module, disable_pngs)

    label = f"pkg:{directory.name}"
    sha = compute_file_sha256(analysis_py)
    return module, label, sha


def load_analyzer_from_file(file_path: Path, disable_pngs: bool) -> tuple[Any, str, str]:
    """Load analyzer from standalone .py file."""
    spec = importlib.util.spec_from_file_location("gel_analyzer", str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import analyzer: {file_path}")

    sys.modules.pop(spec.name, None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Failed to import analyzer file: {file_path}\n{e}") from e

    validate_and_configure_analyzer(module, disable_pngs)

    label = f"file:{file_path.name}"
    sha = compute_file_sha256(file_path)
    return module, label, sha


def validate_and_configure_analyzer(module: Any, disable_pngs: bool) -> None:
    """
    Validate analyzer module has required functions and configure settings.

    Checks for process_stack function and optionally disables PNG outputs.
    """
    if not hasattr(module, "process_stack"):
        raise RuntimeError("Analyzer must expose process_stack() function")

    if disable_pngs:
        png_flags = ["SAVE_PNGS", "SAVE_HEIGHT_PNG", "SAVE_BANDPASS_RESIDUAL_PNG"]
        for flag_name in png_flags:
            if hasattr(module, flag_name):
                try:
                    setattr(module, flag_name, False)
                except Exception:
                    pass


def call_analyzer_process_stack(
    module: Any,
    tif_path: Path,
    *,
    xy_um_per_px: float,
    z_step_um: float,
    z0_um: float,
    out_dir: Path,
    analysis_sha256: str,
) -> tuple[dict[str, Any], str | None]:
    """
    Call analyzer's process_stack function.

    Handles optional parameters and converts result to dict.
    Returns (result_dict, error_message).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        signature = inspect.signature(module.process_stack)
        kwargs: dict[str, Any] = {}

        if "analysis_sha256" in signature.parameters:
            kwargs["analysis_sha256"] = analysis_sha256
        if "analysis_timestamp_utc" in signature.parameters:
            kwargs["analysis_timestamp_utc"] = get_utc_timestamp_iso()

        result = module.process_stack(
            str(tif_path),
            float(xy_um_per_px),
            float(z_step_um),
            float(z0_um),
            str(out_dir),
            **kwargs,
        )
    except Exception as e:
        return {}, str(e)

    if isinstance(result, dict):
        return dict(result), None
    if dataclasses.is_dataclass(result):
        return dataclasses.asdict(result), None
    return dict(getattr(result, "__dict__", {})), None


# =============================================================================
# Field Comparison Functions
# =============================================================================


def compare_float_field(
    field_name: str, truth_val: Any, measured_val: Any, spec: FieldSpec
) -> FieldResult:
    """
    Compare float field values with absolute and relative tolerances.

    Returns FieldResult with errors and pass/fail status.
    """
    truth = convert_to_float(truth_val)
    measured = convert_to_float(measured_val)

    if not np.isfinite(truth):
        return FieldResult(
            field_name,
            truth_val,
            measured_val,
            None,
            None,
            passed=True,
            skipped=True,
            note="truth=NaN (skipped)",
        )

    if not np.isfinite(measured):
        return FieldResult(
            field_name,
            truth_val,
            measured_val,
            None,
            None,
            passed=False,
            skipped=False,
            note="meas=NaN",
        )

    abs_error = float(abs(measured - truth))
    rel_error = float(abs_error / max(abs(truth), 1e-12))

    abs_ok = True if spec.abs_tol is None else (abs_error <= spec.abs_tol)
    rel_ok = True if spec.rel_tol is None else (rel_error <= spec.rel_tol)
    passed = bool(abs_ok and rel_ok)

    note_parts = []
    if spec.abs_tol is not None:
        note_parts.append(f"abs_tol={spec.abs_tol}")
    if spec.rel_tol is not None:
        note_parts.append(f"rel_tol={spec.rel_tol}")

    return FieldResult(
        name=field_name,
        truth=float(truth),
        meas=float(measured),
        abs_err=abs_error,
        rel_err=rel_error,
        passed=passed,
        skipped=False,
        note=",".join(note_parts),
    )


def compare_bool_field(
    field_name: str, truth_val: Any, measured_val: Any, spec: FieldSpec
) -> FieldResult:
    """
    Compare boolean field values.

    Returns FieldResult with pass/fail status.
    """
    truth = convert_to_bool(truth_val)
    measured = convert_to_bool(measured_val)

    if truth is None:
        return FieldResult(
            field_name,
            truth_val,
            measured_val,
            None,
            None,
            passed=True,
            skipped=True,
            note="truth=unset (skipped)",
        )

    if measured is None:
        return FieldResult(
            field_name,
            truth_val,
            measured_val,
            None,
            None,
            passed=False,
            skipped=False,
            note="meas=unset",
        )

    return FieldResult(
        field_name,
        truth,
        measured,
        None,
        None,
        passed=(truth == measured),
        skipped=False,
        note="",
    )


def compare_str_field(
    field_name: str, truth_val: Any, measured_val: Any, spec: FieldSpec
) -> FieldResult:
    """
    Compare string field values.

    Returns FieldResult with pass/fail status.
    """
    truth = convert_to_str(truth_val).strip()
    measured = convert_to_str(measured_val).strip()

    if truth == "":
        return FieldResult(
            field_name,
            truth_val,
            measured_val,
            None,
            None,
            passed=True,
            skipped=True,
            note="truth=empty (skipped)",
        )

    return FieldResult(
        field_name,
        truth,
        measured,
        None,
        None,
        passed=(truth == measured),
        skipped=False,
        note="",
    )


def compare_fields(
    truth_dict: dict[str, Any],
    measured_dict: dict[str, Any],
    field_specs: Iterable[FieldSpec],
) -> list[FieldResult]:
    """
    Compare all specified fields between truth and measured values.

    Returns list of FieldResult objects, one per field.
    """
    results: list[FieldResult] = []
    for spec in field_specs:
        truth_val = truth_dict.get(spec.name)
        measured_val = measured_dict.get(spec.name)

        if spec.kind == "float":
            results.append(compare_float_field(spec.name, truth_val, measured_val, spec))
        elif spec.kind == "bool":
            results.append(compare_bool_field(spec.name, truth_val, measured_val, spec))
        else:
            results.append(compare_str_field(spec.name, truth_val, measured_val, spec))

    return results


# =============================================================================
# Result Processing Functions
# =============================================================================


def extract_scenario_from_row(row_dict: dict[str, Any]) -> str:
    """
    Extract scenario name from truth row.

    Checks multiple possible column names (case-insensitive).
    """
    for key in ("scenario", "Scenario", "SCENARIO"):
        if key in row_dict and str(row_dict.get(key, "")).strip():
            return str(row_dict[key]).strip()
    return ""


def summarize_results(
    results: list[SampleResult],
    field_specs: Iterable[FieldSpec],
    analyzer_label: str,
    truth_csv_name: str,
) -> dict[str, Any]:
    """
    Generate summary statistics from validation results.

    Computes pass rates, error statistics, and per-scenario breakdowns.
    """
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    errors = sum(1 for r in results if r.error)

    field_stats = compute_field_statistics(results, field_specs)
    scenario_breakdown = compute_scenario_breakdown(results)

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "analyzer": analyzer_label,
        "truth_csv": truth_csv_name,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "errors": errors,
        "pass_rate": (passed / total) if total else 0.0,
        "field_stats": field_stats,
        "scenario_breakdown": scenario_breakdown,
        "criteria": [dataclasses.asdict(s) for s in field_specs],
    }


def compute_field_statistics(
    results: list[SampleResult], field_specs: Iterable[FieldSpec]
) -> dict[str, dict[str, float]]:
    """
    Compute error statistics for each validated field.

    Returns dict mapping field names to error statistics.
    """
    field_names = [spec.name for spec in field_specs]
    stats: dict[str, dict[str, float]] = {}

    for name in field_names:
        abs_errors = []
        rel_errors = []

        for result in results:
            for field_res in result.fields:
                if (
                    field_res.name == name
                    and not field_res.skipped
                    and field_res.abs_err is not None
                ):
                    abs_errors.append(field_res.abs_err)
                    if field_res.rel_err is not None:
                        rel_errors.append(field_res.rel_err)

        abs_arr = np.asarray(abs_errors, dtype=float)
        abs_arr = abs_arr[np.isfinite(abs_arr)]
        rel_arr = np.asarray(rel_errors, dtype=float)
        rel_arr = rel_arr[np.isfinite(rel_arr)]

        stats[name] = {
            "abs_err_mean": float(np.mean(abs_arr)) if abs_arr.size else float("nan"),
            "abs_err_max": float(np.max(abs_arr)) if abs_arr.size else float("nan"),
            "rel_err_mean": float(np.mean(rel_arr)) if rel_arr.size else float("nan"),
            "rel_err_max": float(np.max(rel_arr)) if rel_arr.size else float("nan"),
        }

    return stats


def compute_scenario_breakdown(
    results: list[SampleResult],
) -> dict[str, dict[str, Any]]:
    """
    Compute pass rates grouped by test scenario.

    Returns dict mapping scenario names to statistics.
    """
    scenarios = sorted({(r.scenario or "<unspecified>") for r in results})
    breakdown: dict[str, dict[str, Any]] = {}

    for scenario in scenarios:
        scenario_results = [r for r in results if (r.scenario or "<unspecified>") == scenario]
        passed_count = sum(1 for r in scenario_results if r.passed)

        breakdown[scenario] = {
            "total": len(scenario_results),
            "passed": passed_count,
            "pass_rate": ((passed_count / len(scenario_results)) if scenario_results else 0.0),
        }

    return breakdown


def flatten_results(results: list[SampleResult]) -> list[dict[str, Any]]:
    """
    Convert SampleResult objects to flat dictionaries for CSV export.

    Each result becomes one row with per-field columns.
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        row = {
            "File": result.file,
            "Scenario": result.scenario,
            "Passed": result.passed,
            "Error": result.error,
        }

        for field_res in result.fields:
            prefix = field_res.name
            row[f"{prefix}__truth"] = field_res.truth
            row[f"{prefix}__meas"] = field_res.meas
            row[f"{prefix}__abs_err"] = field_res.abs_err
            row[f"{prefix}__rel_err"] = field_res.rel_err
            row[f"{prefix}__passed"] = field_res.passed
            row[f"{prefix}__skipped"] = field_res.skipped

        rows.append(row)

    return rows


def print_validation_summary(summary: dict[str, Any]) -> None:
    """
    Print validation summary to console.

    Displays key metrics and per-scenario results in readable format.
    """
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Timestamp:     {summary.get('timestamp', 'N/A')}")
    print(f"Analyzer:      {summary.get('analyzer', 'N/A')}")
    print(f"Truth CSV:     {summary.get('truth_csv', 'N/A')}")
    print()
    print(f"Total samples: {summary['total']}")
    print(f"Passed:        {summary['passed']} ({summary['pass_rate']:.1%})")
    print(f"Failed:        {summary['failed']}")
    print(f"Errors:        {summary['errors']}")
    print()

    print("Field Error Statistics:")
    print("-" * 80)
    field_stats = summary.get("field_stats", {})
    if field_stats:
        print(f"{'Field':<30} {'Mean Error':<15} {'Max Error':<15}")
        print("-" * 80)
        for field_name, stats in field_stats.items():
            mean_err = stats["abs_err_mean"]
            max_err = stats["abs_err_max"]
            mean_str = f"{mean_err:.4g}" if np.isfinite(mean_err) else "N/A"
            max_str = f"{max_err:.4g}" if np.isfinite(max_err) else "N/A"
            print(f"{field_name:<30} {mean_str:<15} {max_str:<15}")
    else:
        print("No field statistics available")
    print()

    print("Per-Scenario Results:")
    print("-" * 80)
    scenario_breakdown = summary.get("scenario_breakdown", {})
    if scenario_breakdown:
        print(f"{'Scenario':<20} {'Total':<10} {'Passed':<10} {'Pass Rate':<15}")
        print("-" * 80)
        for scenario, stats in scenario_breakdown.items():
            print(
                f"{scenario:<20} {stats['total']:<10} "
                f"{stats['passed']:<10} {stats['pass_rate']:<14.1%}"
            )
    else:
        print("No scenario breakdown available")

    print("=" * 80)
    print()


# =============================================================================
# Main Validation Function
# =============================================================================


def run_validation(
    dataset_dir: Path,
    analyzer_path: Path,
    *,
    truth_csv_name: str = "truth_metrics.csv",
    output_dir_name: str = "validation_outputs",
    include_optional_fields: bool = False,
    disable_pngs: bool = True,
) -> tuple[list[SampleResult], dict[str, Any]]:
    """
    Execute complete validation workflow.

    Loads analyzer, processes all samples, compares results, and generates reports.
    Returns (results_list, summary_dict).
    """
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {dataset_dir}")

    truth_csv = dataset_dir / truth_csv_name
    truth_rows = read_truth_metrics_csv(truth_csv)

    module, analyzer_label, analyzer_sha = load_analyzer_module(
        analyzer_path, disable_pngs=disable_pngs
    )

    config = ValidationConfig()
    field_specs = config.fields + (config.optional_fields if include_optional_fields else tuple())

    out_root = dataset_dir / output_dir_name
    out_root.mkdir(parents=True, exist_ok=True)

    results: list[SampleResult] = []
    for idx, truth_row in enumerate(truth_rows, 1):
        result = process_single_sample(
            truth_row,
            dataset_dir,
            truth_csv,
            module,
            analyzer_sha,
            field_specs,
            out_root,
            idx,
            len(truth_rows),
        )
        results.append(result)

    summary = summarize_results(results, field_specs, analyzer_label, truth_csv.name)
    write_validation_outputs(dataset_dir, results, summary, out_root)

    return results, summary


def process_single_sample(
    truth_row: TruthRow,
    dataset_dir: Path,
    truth_csv: Path,
    analyzer_module: Any,
    analyzer_sha: str,
    field_specs: tuple[FieldSpec, ...],
    out_root: Path,
    sample_num: int,
    total_samples: int,
) -> SampleResult:
    """
    Process and validate a single sample.

    Calls analyzer, compares results to ground truth, returns SampleResult.
    """
    tif_path = dataset_dir / truth_row.file
    if not tif_path.exists():
        tif_path = (truth_csv.parent / truth_row.file).resolve()

    per_out = out_root / truth_row.stem
    per_out.mkdir(parents=True, exist_ok=True)

    xy = convert_to_float(truth_row.row.get("XY_um_per_px", np.nan))
    dz = convert_to_float(truth_row.row.get("Z_step_um", np.nan))
    z0 = convert_to_float(truth_row.row.get("Z0_um", np.nan))

    if not (np.isfinite(xy) and np.isfinite(dz) and np.isfinite(z0)):
        return SampleResult(
            file=truth_row.file,
            scenario=extract_scenario_from_row(truth_row.row),
            passed=False,
            error="Missing voxel params (need XY_um_per_px, Z_step_um, Z0_um)",
            fields=[],
        )

    LOG.info("[%d/%d] %s", sample_num, total_samples, truth_row.file)

    measured, error = call_analyzer_process_stack(
        analyzer_module,
        tif_path,
        xy_um_per_px=xy,
        z_step_um=dz,
        z0_um=z0,
        out_dir=per_out,
        analysis_sha256=analyzer_sha,
    )

    if error is not None:
        return SampleResult(
            file=truth_row.file,
            scenario=extract_scenario_from_row(truth_row.row),
            passed=False,
            error=error,
            fields=[],
        )

    field_results = compare_fields(truth_row.row, measured, field_specs)
    passed = all(fr.passed for fr in field_results if not fr.skipped)

    return SampleResult(
        file=truth_row.file,
        scenario=extract_scenario_from_row(truth_row.row),
        passed=passed,
        error="",
        fields=field_results,
    )


# =============================================================================
# GUI Functions
# =============================================================================


def select_file_gui(title: str, filetypes: list[tuple[str, str]]) -> Path | None:
    """
    Open file selection dialog.

    Returns selected Path or None if cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        LOG.error("tkinter not available - cannot use GUI mode")
        return None

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()

    return Path(file_path) if file_path else None


def select_directory_gui(title: str) -> Path | None:
    """
    Open directory selection dialog.

    Returns selected Path or None if cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        LOG.error("tkinter not available - cannot use GUI mode")
        return None

    root = tk.Tk()
    root.withdraw()
    dir_path = filedialog.askdirectory(title=title)
    root.destroy()

    return Path(dir_path) if dir_path else None


def run_gui_mode() -> None:
    """
    Run validator in GUI mode with file/directory pickers.

    Prompts user to select analyzer and dataset directory.
    """
    print("\n=== Synthetic Flat Gel Validator - GUI Mode ===\n")

    print("Step 1: Select analyzer file (.py)")
    analyzer_path = select_file_gui(
        "Select Analyzer File", [("Python files", "*.py"), ("All files", "*.*")]
    )

    if not analyzer_path:
        print("Cancelled - no analyzer selected")
        return

    print(f"Selected analyzer: {analyzer_path}")

    print("\nStep 2: Select dataset directory")
    dataset_dir = select_directory_gui("Select Dataset Directory")

    if not dataset_dir:
        print("Cancelled - no dataset directory selected")
        return

    print(f"Selected dataset: {dataset_dir}")

    print("\nRunning validation...\n")

    try:
        results, summary = run_validation(
            dataset_dir,
            analyzer_path,
            truth_csv_name="truth_metrics.csv",
            output_dir_name="validation_outputs",
            include_optional_fields=False,
            disable_pngs=True,
        )

        print_validation_summary(summary)

        print("\nValidation complete!")
        print(f"Results saved in: {dataset_dir}")

    except Exception as e:
        LOG.error("Validation failed: %s", e, exc_info=True)
        print(f"\nError: {e}")


# =============================================================================
# CLI and Entry Point
# =============================================================================


def setup_logging(level: str) -> None:
    """Configure logging with specified level."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    """Parse arguments and run validation in CLI or GUI mode."""
    parser = argparse.ArgumentParser(
        description="Validate gel analyzer against synthetic data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--gui", action="store_true", help="Run in GUI mode with file pickers")
    parser.add_argument(
        "-a",
        "--analyzer-path",
        type=str,
        help="Path to analyzer (.py) or folder with analysis.py",
    )
    parser.add_argument(
        "-d", "--dataset-dir", type=str, help="Dataset directory with truth_metrics.csv"
    )
    parser.add_argument("--truth-csv", default="truth_metrics.csv", help="Truth CSV filename")
    parser.add_argument(
        "--out-dir-name",
        default="validation_outputs",
        help="Subfolder for analyzer outputs",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Compare optional fields (currently none)",
    )
    parser.add_argument("--keep-pngs", action="store_true", help="Keep analyzer PNG outputs")
    parser.add_argument(
        "--log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    setup_logging(args.log)

    if args.gui:
        run_gui_mode()
    else:
        if not args.analyzer_path or not args.dataset_dir:
            parser.error(
                "CLI mode requires both --analyzer-path and --dataset-dir "
                "(or use --gui for interactive mode)"
            )

        results, summary = run_validation(
            Path(args.dataset_dir),
            Path(args.analyzer_path),
            truth_csv_name=args.truth_csv,
            output_dir_name=args.out_dir_name,
            include_optional_fields=args.include_optional,
            disable_pngs=not args.keep_pngs,
        )

        print_validation_summary(summary)


if __name__ == "__main__":
    main()
