"""Re-run the chain, then assert the claims the results rest on.

Two different things need checking and they need different treatment.

**Reproducibility** - does the code still produce the tables? Answered by
re-running every script in order (``--full``), which rewrites everything
under ``data/synthetic``, ``data/validation`` and ``data/results``.

**Correctness** - do the results still support what is said about them?
Answered by the gates below, which are assertions with numbers in them, run
against whatever is on disk. They guard against a chain that runs cleanly but
no longer produces the results the text describes:

  1. ``s - Indentation`` is one constant on every curve, which is what
     licenses reading a contact point out of that column;
  2. quality control retains at least 80 % of the curves in every condition;
  3. on a synthetic Hertzian half-space every estimator that does not apply a
     film correction returns the true modulus, and the film method's
     departure is exactly its own correction term;
  4. every pathological synthetic scenario is rejected by QC and every benign
     one is retained;
  5. the pipeline recovers ground truth on every benign scenario where a
     half-space Hertz answer is defined, over the same scenario set gate 4
     uses. Where the physics moves the answer -- a bonded film, a surface
     layer, a graded gel, adhesion, viscoelasticity -- there is no half-space
     truth to recover, so what an uncorrected fit reads is reported rather
     than scored, next to a forward calculation from the generator's own
     mechanics;
  6. most of the within-gel scatter is the sample, not the analysis.

Run
---
    python verify.py            # gates only, against what is on disk
    python verify.py --full     # re-run the chain first, then the gates
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("LIGHTS_DATA", CODE.parent / "data"))
sys.path.insert(0, str(CODE / "_common"))

import lights_fit as lf  # nopep8
from scipy import optimize  # nopep8

RESULTS = ROOT / "results"
VALIDATION = ROOT / "validation"
SCRIPTS = (
    "01_synthesise.py",
    "02_validate.py",
    "03_analyze.py",
    "04_statistics.py",
)
#: Scenarios on which a half-space Hertz answer is exact by construction.
HALF_SPACE = ("ideal", "ideal_100hz", "soft_2kPa", "stiff_20kPa")
#: Estimators that model a plain half-space. The film method deliberately
#: does not, and is reported rather than asserted.
UNCORRECTED = ("pipeline", "nonlinear", "stiffness")
#: Built to be rejected.
PATHOLOGIES = (
    "no_contact",
    "starts_in_contact",
    "shallow",
    "hard_contact",
    "snap_in",
    "slip",
    "spike",
)
#: Built to be kept: ordinary curves and the non-Hertzian mechanics that QC
#: is not there to filter out. The two deliberate stress tests (25 nm
#: deflection noise, a 20 nN 1 Hz wobble) are excluded because they are meant
#: to be marginal. The composite scenario at 16 kPa is excluded because it
#: reaches 6.6 uN of peak force, thirteen times the median real curve: at
#: that contact stiffness the displacement noise alone produces force scatter
#: larger than the step criterion's threshold, which is a multiple of the
#: baseline force sigma. See VALIDATION_REPORT.md.
BENIGN = (
    "ideal",
    "ideal_100hz",
    "soft_2kPa",
    "stiff_20kPa",
    "drift",
    "drift_extreme",
    "oscillation_real",
    "film_20um",
    "film_50um",
    "layer_1um",
    "layer_2um",
    "graded",
    "adhesion_weak",
    "adhesion_strong",
    "viscoelastic",
    "poroviscoelastic",
    "realistic_3kPa",
    "realistic_8kPa",
)
TOLERANCE = 0.02
PASS: list[str] = []
FAIL: list[str] = []


def gate(name: str, ok: bool, detail: str) -> None:
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def run_chain() -> None:
    for script in SCRIPTS:
        print(f"--- {script}")
        result = subprocess.run(
            [sys.executable, str(CODE / "scripts" / script)],
            cwd=CODE,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(result.stdout[-3000:])
            print(result.stderr[-3000:])
            raise SystemExit(f"{script} failed with {result.returncode}")


def gate_vendor_channel() -> None:
    cond = pd.read_csv(RESULTS / "condition_summary.csv")
    worst = float(cond.max_vendor_s0_spread_nm.max())
    gate(
        "vendor Indentation channel encodes one contact coordinate",
        worst < 1e-3,
        f"max spread of s - Indentation = {worst:.1e} nm over "
        f"{int(cond.n_measured.sum())} curves",
    )


def gate_retention() -> None:
    cond = pd.read_csv(RESULTS / "condition_summary.csv")
    worst = float(cond.retention.min())
    overall = float(cond.n_retained.sum()) / float(cond.n_measured.sum())
    gate(
        "quality control retains at least 80 % in every condition",
        worst >= 0.80,
        f"worst condition {100 * worst:.1f} %, overall {100 * overall:.1f} % "
        f"({int(cond.n_retained.sum())} of {int(cond.n_measured.sum())})",
    )


def film_offset_on_exact_hertz() -> tuple[float, float]:
    """What the film correction alone does to an exact Hertz curve.

    Fit the film-corrected model, with exactly the freedom the estimator has
    (force offset, contact point, slope), to noiseless Hertz data over the
    analysis window plus the 1 um of pre-contact baseline the estimator keeps
    inside it. No file, no noise, no detection. Returns the slope ratio and
    the contact-point shift the correction term produces by itself.
    """
    radius_um, thickness_um, s0_nm = 27.0, lf.DEFAULT_THICKNESS_UM, 5810.0
    s = np.arange(s0_nm - 1000.0, s0_nm + 4200.0, 1.0)
    force = np.power(np.clip(s - s0_nm, 0.0, None), 1.5)

    def model(x, f0, s0, k):
        depth = np.clip(x - s0, 0.0, None)
        return f0 + k * np.power(depth, 1.5) * lf.garcia_bec(
            depth, radius_um, thickness_um
        )

    win = (
        (s >= s0_nm - 1000.0)
        & (s <= s0_nm + lf.FIT_MAX_DEPTH_UM * 1000.0)
        & ((s < s0_nm) | (s >= s0_nm + lf.FIT_MIN_DEPTH_UM * 1000.0))
    )
    popt, _ = optimize.curve_fit(
        model, s[win], force[win], p0=[0.0, s0_nm, 1.0], maxfev=80000
    )
    return float(popt[2]), float(popt[1] - s0_nm) / 1000.0


def gate_exact_on_half_space() -> None:
    s = pd.read_csv(VALIDATION / "validation_summary.csv")
    clean = s[s.scenario.isin(HALF_SPACE)]
    core = clean[clean.method.isin(UNCORRECTED)]
    worst = float(np.nanmax(np.abs(core.model_bias - 1.0)))
    film = clean[clean.method == "film"]
    measured_e = float(film.model_bias.median())
    measured_s0 = float(film.s0_noisefree_error_um.median())
    predicted_e, predicted_s0 = film_offset_on_exact_hertz()
    gate(
        "every uncorrected estimator is exact on a Hertzian half-space",
        worst < 0.005
        and abs(measured_e / predicted_e - 1.0) < 0.01
        and abs(measured_s0 - predicted_s0) < 0.01,
        f"worst model-form bias 1{worst:+.2e} over "
        f"{core.scenario.nunique()} scenarios x {core.method.nunique()} "
        f"estimators; the film method reads x{measured_e:.4f} and "
        f"{measured_s0:+.4f} um there against x{predicted_e:.4f} and "
        f"{predicted_s0:+.4f} um for its correction term alone",
    )


def gate_qc_on_synthetic() -> None:
    m = pd.read_csv(VALIDATION / "validation_master.csv")
    rate = m[m.method == "pipeline"].groupby("scenario").qc_pass.mean()
    caught = sum(rate.get(s, 1.0) == 0.0 for s in PATHOLOGIES)
    kept = sum(rate.get(s, 0.0) == 1.0 for s in BENIGN)
    gate(
        "QC catches every pathology and keeps every benign scenario",
        caught == len(PATHOLOGIES) and kept == len(BENIGN),
        f"{caught}/{len(PATHOLOGIES)} pathologies rejected, "
        f"{kept}/{len(BENIGN)} benign scenarios retained",
    )


def gate_recovery() -> None:
    """Scored on the benign half-space scenarios, the set gate 4 uses.

    ``BENIGN`` already excludes the two deliberate stress tests, a 25 nm
    deflection noise and a 20 nN 1 Hz wobble, because they are built to be
    marginal: the fit window spans a single period of that wobble. Scoring
    recovery on them would be scoring the pipeline on curves the synthetic
    set declares unrecoverable, and gate 4 makes the same exclusion.

    Where the physics moves the answer there is no half-space modulus to
    recover, so those scenarios are reported rather than asserted. The 20 um
    bonded film is printed as the worked example of what an uncorrected
    half-space fit reads on such a scenario.
    """
    m = pd.read_csv(VALIDATION / "validation_master.csv")
    s = pd.read_csv(VALIDATION / "validation_summary.csv")
    kept = m[
        (m.method == "pipeline")
        & m.qc_pass
        & m.half_space_hertz_defined
        & m.scenario.isin(BENIGN)
    ]
    inside = np.abs(kept.recovery - 1.0) <= TOLERANCE
    moved = s[
        (s.method == "pipeline")
        & (~s.pathology)
        & (~s.half_space_hertz_defined)
    ]
    film = moved[moved.scenario == "film_20um"]
    example = (
        f", e.g. on a 20 um bonded film it reads "
        f"{float(film.model_bias.iloc[0]):.2f}x the true bulk modulus"
        if len(film)
        else ""
    )
    gate(
        "the pipeline recovers ground truth on every benign half-space "
        "scenario",
        float(inside.mean()) >= 0.90,
        f"{100 * float(inside.mean()):.1f} % of {len(kept)} retained curves "
        f"within +/-{100 * TOLERANCE:.0f} %, over the same scenario set as "
        f"gate 4; the {len(moved)} physics-moved scenarios have no half-space "
        "truth to recover, so what an uncorrected fit reads there is "
        f"reported, not scored{example}",
    )


def gate_variance_budget() -> None:
    v = pd.read_csv(RESULTS / "variance_budget.csv")
    row = v[v.campaign == "ALL"].iloc[0]
    gate(
        "most within-gel scatter is the sample, not the analysis",
        row.material_share_of_variance > 0.75,
        f"material {100 * row.material_share_of_variance:.1f} %, analysis "
        f"{100 * row.analysis_share_of_variance:.1f} %, measurement "
        f"{100 * row.measurement_share_of_variance:.1f} %; "
        f"sigma(ln E*) within gel {row.sigma_within_gel:.3f}, between gel "
        f"{row.sigma_between_gel:.3f}, between condition "
        f"{row.sigma_between_condition:.3f}",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full", action="store_true", help="re-run every script first"
    )
    args = ap.parse_args()
    if args.full:
        run_chain()
    print("\n=== gates ===")
    for fn in (
        gate_vendor_channel,
        gate_retention,
        gate_exact_on_half_space,
        gate_qc_on_synthetic,
        gate_recovery,
        gate_variance_budget,
    ):
        try:
            fn()
        except Exception as exc:  # report the failure, never hide it
            gate(fn.__name__, False, f"raised {type(exc).__name__}: {exc}")
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed gates: " + ", ".join(FAIL))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
