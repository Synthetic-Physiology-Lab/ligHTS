"""Step 2: run the pipeline on the synthetic set and compare against truth.

Three quantities are reported per curve:

**Model-form bias** = E*(noise-free twin) / E*(true bulk).
**Robustness** = E*(noisy curve) / E*(noise-free twin). 
**Predicted bias** = take the generator's own noise-free curve, measure indentation
from the recorded true contact point, and take the least-squares Hertz slope
over the indentation interval the estimator analysed, with the apparent
contact pinned at the offset the estimator found. 

The three supporting methods are run alongside the pipeline on every curve.
A fifth column, ``film_known_thickness``, is the film method told the true
film thickness, it checks that the bonded-film
correction is exact when the thickness is known.

QC runs on every synthetic curve, so the pathological scenarios test whether
the filter catches what it is for and the benign ones test whether it rejects
them wrongly.

Outputs
-------
data/validation/validation_master.csv   one row per curve per method
data/validation/validation_summary.csv  one row per scenario per method
data/validation/figures/<scenario>.png  the per-scenario check panel
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("LIGHTS_DATA", CODE.parent / "data"))
sys.path.insert(0, str(CODE / "_common"))

import lights_fit as lf  # nopep8
import lights_qc as lq  # nopep8
import lights_raw as lr  # nopep8
import matplotlib  # nopep8

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # nopep8

OUT = ROOT / "validation"
METHODS = (
    "pipeline",
    "nonlinear",
    "stiffness",
    "film",
    "film_known_thickness",
)
#: Scenarios built to be rejected
PATHOLOGIES = (
    "no_contact",
    "starts_in_contact",
    "shallow",
    "hard_contact",
    "snap_in",
    "slip",
    "spike",
)
#: Recovery is counted at this relative tolerance on E*.
TOLERANCE = 0.02


def fit_all(rc, true_thickness_um: float = float("nan")) -> tuple:
    """The pipeline and its three cross-checks
    """
    p = lf.prepare(rc)
    if p is None:
        return None, {}, {}
    fit = lf.fit_pipeline(p)
    seed = fit.extra.get("seed_s0_um", np.nan) * 1000.0
    if not np.isfinite(seed):
        line = lf.linearised_line(p, 0.20, 1.00)
        seed = line[1] if line else np.nan
    fits = {"pipeline": fit}
    if np.isfinite(seed):
        slope = fit.extra.get("hertz_slope_nN_per_nm32")
        fits["nonlinear"] = lf.fit_nonlinear(
            p, seed_s0_nm=seed, seed_slope=slope
        )
        fits["stiffness"] = lf.fit_stiffness(p, seed_s0_nm=seed)
        fits["film"] = lf.fit_film(p, seed_s0_nm=seed, seed_slope=slope)
        if np.isfinite(true_thickness_um):
            fits["film_known_thickness"] = lf.fit_film(
                p,
                seed_s0_nm=seed,
                seed_slope=slope,
                thickness_um=true_thickness_um,
            )
    return p, fits, lq.qc_curve(p, fit)


def predicted_bias(rc, truth_row: pd.Series, s0_fit_um: float) -> float:
    """Forward model-form term
    """
    shift_nm = s0_fit_um * 1000.0 - truth_row.s0_true_nm
    if not np.isfinite(shift_nm):
        return float("nan")
    f = rc.load_uN * 1000.0
    f = f - float(np.median(f[rc.time_s <= rc.time_s[0] + 0.10]))
    peak_i = int(np.nanargmax(f))
    depth = rc.s_nm - truth_row.s0_true_nm - shift_nm
    win = (
        (np.arange(f.size) <= peak_i)
        & (depth >= lf.FIT_MIN_DEPTH_UM * 1000.0)
        & (depth <= lf.FIT_MAX_DEPTH_UM * 1000.0)
    )
    if int(win.sum()) < 25:
        return float("nan")
    x = np.power(depth[win], 1.5)
    slope = float((x @ f[win]) / (x @ x))
    e_pred = lf.e_star_from_slope(slope, truth_row.radius_um)
    return e_pred / (truth_row.e_star_true_Pa / 1000.0)


def noise_free_pass(truth: pd.DataFrame) -> dict:
    """Every method on noise-free twins and the forward prediction."""
    out: dict = {}
    for _, tr in truth[truth.noise_free].iterrows():
        rc = lr.read_chiaro(ROOT / tr.path)
        _p, fits, _qc = fit_all(rc, tr.thickness_um)
        for name in METHODS:
            r = fits.get(name)
            ok = r is not None and r.ok
            out[(tr.scenario, name, "e")] = r.e_star_kPa if ok else np.nan
            out[(tr.scenario, name, "s0")] = r.contact_s0_um if ok else np.nan
        ref = fits.get("pipeline")
        out[(tr.scenario, "pipeline", "pred")] = (
            predicted_bias(rc, tr, ref.contact_s0_um)
            if ref is not None and ref.ok
            else np.nan
        )
    return out


def check_panel(scenario: str, path: Path, tr: pd.Series) -> None:
    """One self-contained panel per scenario."""
    rc = lr.read_chiaro(path)
    p, fits, qc = fit_all(rc)
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.0))
    s0_true = tr.s0_true_nm / 1000.0

    ax = axes[0]
    ax.plot(rc.s_nm / 1000.0, rc.load_uN * 1000.0, color="0.7", lw=0.6)
    ax.axvline(s0_true, color="#009E73", lw=1.0, label="true $s_0$")
    ax.set_title(f"raw file: {path.name}", fontsize=7)
    ax.set_xlabel("s (um)")
    ax.set_ylabel("Load (nN), raw")
    ax.legend(fontsize=6)

    ax = axes[1]
    if p is not None:
        ax.plot(
            p.s_nm / 1000.0,
            p.force_nN,
            color="#0072B2",
            lw=0.8,
            label="processed loading branch",
        )
        ax.axhline(0, color="k", lw=0.4)
        if p.onset_index >= 0:
            ax.axvline(
                p.s_onset_nm / 1000.0,
                color="#56B4E9",
                lw=0.8,
                ls=":",
                label="detected onset",
            )
    ax.axvline(s0_true, color="#009E73", lw=1.0)
    fit = fits.get("pipeline")
    if fit is not None and fit.ok:
        ax.axvline(
            fit.contact_s0_um, color="#D55E00", lw=1.0, label="fitted contact"
        )
        ax.axvspan(
            fit.contact_s0_um + lf.FIT_MIN_DEPTH_UM,
            fit.contact_s0_um + lf.FIT_MAX_DEPTH_UM,
            color="#D55E00",
            alpha=0.10,
            lw=0,
            label="analysis window",
        )
        d = np.linspace(0, lf.FIT_MAX_DEPTH_UM, 80)
        ax.plot(
            fit.contact_s0_um + d,
            fit.extra["hertz_slope_nN_per_nm32"] * (d * 1000) ** 1.5,
            "k--",
            lw=0.9,
            label="fitted Hertz model",
        )
    ax.axvline(
        s0_true + lf.FIT_MAX_DEPTH_UM,
        color="#009E73",
        lw=0.6,
        ls="--",
        label=f"true $s_0$ + {lf.FIT_MAX_DEPTH_UM:g} um",
    )
    ax.set_xlabel("s (um)")
    ax.set_ylabel("force (nN), baseline removed")
    ax.set_title("processed, contact and window", fontsize=7)
    ax.legend(fontsize=5.5)

    ax = axes[2]
    if p is not None:
        pos = p.force_nN > 0
        ax.plot(p.s_nm[pos] / 1000.0, p.force_nN[pos] ** (2 / 3), ".", ms=1)
        ax.axvline(s0_true, color="#009E73", lw=1.0)
        if fit is not None and fit.ok:
            ax.axvline(fit.contact_s0_um, color="#D55E00", lw=1.0)
    ax.set_xlabel("s (um)")
    ax.set_ylabel("$F^{2/3}$")
    ax.set_title("Hertz linearisation (straight iff Hertz)", fontsize=7)

    ax = axes[3]
    ax.axis("off")
    e_true = tr.e_star_true_Pa / 1000.0
    lines = [
        f"scenario: {scenario}",
        f"{tr.notes}",
        "",
        f"TRUE   E* = {e_true:.3f} kPa",
        f"TRUE   s0 = {s0_true:.3f} um",
        f"half-space Hertz defined: {bool(tr.expect_recovers_e_star)}",
        "",
        f"QC: {'PASS' if qc.get('qc_pass') else 'FAIL'}"
        f"  {qc.get('qc_reasons', '')}",
        "",
        f"{'method':<22}{'E* kPa':>9}{'/true':>8}{'s0 um':>9}",
    ]
    for name in METHODS:
        r = fits.get(name)
        if r is None or not r.ok:
            lines.append(f"{name:<22}{'-':>9}{'-':>8}{'-':>9}")
            continue
        lines.append(
            f"{name:<22}{r.e_star_kPa:>9.3f}"
            f"{r.e_star_kPa / e_true:>8.3f}{r.contact_s0_um:>9.3f}"
        )
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=7,
        family="monospace",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"{scenario}.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(ROOT / "synthetic" / "truth.csv")
    pd.set_option("display.width", 200)

    print("noise-free twins ...")
    noise_free = noise_free_pass(truth)

    rows = []
    for _, tr in truth[~truth.noise_free].iterrows():
        rc = lr.read_chiaro(ROOT / tr.path)
        _p, fits, qc = fit_all(rc)
        e_true = tr.e_star_true_Pa / 1000.0
        for name in METHODS:
            r = fits.get(name)
            e_nf = noise_free.get((tr.scenario, name, "e"), np.nan)
            rows.append(
                {
                    "scenario": tr.scenario,
                    "curve_id": tr.curve_id,
                    "method": name,
                    "ok": bool(r.ok) if r else False,
                    "e_star_kPa": r.e_star_kPa if r else np.nan,
                    "contact_s0_um": r.contact_s0_um if r else np.nan,
                    "reason": r.reason if r else "no_fit",
                    "e_true_kPa": e_true,
                    "s0_true_um": tr.s0_true_nm / 1000.0,
                    "half_space_hertz_defined": tr.expect_recovers_e_star,
                    "pathology": tr.scenario in PATHOLOGIES,
                    "qc_pass": qc.get("qc_pass", False),
                    "qc_reasons": qc.get("qc_reasons", "no_loading_branch"),
                    "e_noisefree_kPa": e_nf,
                    "s0_noisefree_um": noise_free.get(
                        (tr.scenario, name, "s0"), np.nan
                    ),
                    "predicted_bias": (
                        noise_free.get(
                            (tr.scenario, "pipeline", "pred"), np.nan
                        )
                        if name == "pipeline"
                        else np.nan
                    ),
                }
            )
    df = pd.DataFrame(rows)
    df["model_bias"] = df.e_noisefree_kPa / df.e_true_kPa
    df["robustness"] = df.e_star_kPa / df.e_noisefree_kPa
    df["ratio_to_true"] = df.e_star_kPa / df.e_true_kPa
    df["s0_error_um"] = df.contact_s0_um - df.s0_true_um
    df["s0_noisefree_error_um"] = df.s0_noisefree_um - df.s0_true_um
    # Recovery target: the true bulk modulus where a half-space Hertz answer
    # is defined, and the forward-predicted value where the physics moves it.
    df["e_target_kPa"] = np.where(
        df.half_space_hertz_defined,
        df.e_true_kPa,
        df.predicted_bias * df.e_true_kPa,
    )
    df["recovery"] = df.e_star_kPa / df.e_target_kPa
    df.to_csv(OUT / "validation_master.csv", index=False)

    summary = (
        df.groupby(["scenario", "method"])
        .agg(
            n=("ok", "size"),
            n_ok=("ok", "sum"),
            n_qc_pass=("qc_pass", "sum"),
            half_space_hertz_defined=("half_space_hertz_defined", "first"),
            pathology=("pathology", "first"),
            model_bias=("model_bias", "median"),
            predicted_bias=("predicted_bias", "median"),
            robustness=("robustness", "median"),
            ratio_to_true=("ratio_to_true", "median"),
            recovery=("recovery", "median"),
            cv_percent=(
                "e_star_kPa",
                lambda x: 100 * np.nanstd(x, ddof=1) / np.nanmean(x),
            ),
            s0_error_um=("s0_error_um", "median"),
            s0_noisefree_error_um=("s0_noisefree_error_um", "median"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / "validation_summary.csv", index=False)

    core = summary[summary.method == "pipeline"]
    print("\nmodel-form bias, forward prediction, and robustness (pipeline):")
    print(
        core[
            [
                "scenario",
                "half_space_hertz_defined",
                "pathology",
                "model_bias",
                "predicted_bias",
                "robustness",
                "recovery",
                "n_qc_pass",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print("\nmodel-form bias by method:")
    print(
        summary.pivot(index="scenario", columns="method", values="model_bias")
        .round(4)
        .to_string()
    )
    print("\ncontact position error, noise-free (fitted - true), um:")
    print(
        summary.pivot(
            index="scenario", columns="method", values="s0_noisefree_error_um"
        )
        .round(4)
        .to_string()
    )

    kept = df[(df.method == "pipeline") & (~df.pathology) & df.qc_pass]
    inside = np.abs(kept.recovery - 1.0) <= TOLERANCE
    grp = kept.half_space_hertz_defined
    print(
        f"\nrecovery within +/-{100 * TOLERANCE:.0f} %: "
        f"{100 * float(inside.mean()):.1f} % of {len(kept)} retained curves "
        f"(half-space {100 * float(inside[grp].mean()):.1f} % of "
        f"{int(grp.sum())}, physics-moved "
        f"{100 * float(inside[~grp].mean()):.1f} % of {int((~grp).sum())})"
    )
    worst = (
        core[core.pathology.eq(False) & ~core.half_space_hertz_defined]
        .assign(gap=lambda d: (d.model_bias / d.predicted_bias - 1).abs())
        .gap.max()
    )
    print(
        "worst |model_bias / predicted_bias - 1| over the physics-moved "
        f"scenarios: {100 * float(worst):.2f} %"
    )
    pathologies = df[(df.method == "pipeline") & df.pathology]
    print(
        f"pathologies rejected by QC: "
        f"{int((~pathologies.qc_pass).sum())}/{len(pathologies)}"
    )

    if not args.no_figures:
        for scenario in truth.scenario.unique():
            tr = truth[
                (truth.scenario == scenario) & (~truth.noise_free)
            ].iloc[0]
            check_panel(scenario, ROOT / tr.path, tr)
        print(f"wrote {truth.scenario.nunique()} check panels")


if __name__ == "__main__":
    main()
