"""Step 3: quality control
For each curve it prepares the loading
branch, fits the pipeline estimator, and runs the sixteen mechanical
criteria. For each curve the criteria retain it also measures:

* three contact-point-free measurements of whether the apparent modulus
  depends on indentation depth, the window ladder, the pointwise modulus,
  and the curvature of the stiffness plot;
* the analysis sweep: the same curve refitted under every defensible setting
  of the pipeline and of the three cross-check methods.

The vendor's own contact coordinate is recovered from the ``Indentation``
channel on every curve and its spread is recorded per condition.

Outputs
-------
data/results/per_curve.csv           one row per retained curve
data/results/condition_summary.csv   per campaign x condition
data/results/figures/map_modulus_<campaign>.png
data/results/figures/map_contact_<campaign>.png
data/results/figures/map_contact_residual_<campaign>.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time
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
from matplotlib.colors import LogNorm  # nopep8

RAW = ROOT / "raw"
OUT = ROOT / "results"
CAMPAIGN_LABEL = {
    "D1_June2025": "batch 1 (17 Jul 2025)",
    "D2_June2026": "batch 2, fresh (5 Jun 2026)",
    "D3_July2026": "batch 2, +1 month (6-9 Jul 2026)",
}
#: Lateral pitch of the indentation grid, in micrometres.
PITCH_UM = {"D1_June2025": 1000.0, "D2_June2026": 300.0, "D3_July2026": 300.0}

SETTINGS = (
    ("pipeline", "seed_force_lo", (0.10, 0.15, 0.20, 0.30, 0.40)),
    ("pipeline", "depth_hi_um", (2.5, 3.0, 3.5, 4.0)),
    ("pipeline", "depth_lo_um", (0.0, 0.25, 0.50, 1.00)),
    (
        "pipeline",
        "baseline_model",
        ("initial_constant", "constant", "linear", "quadratic"),
    ),
    ("pipeline", "ramp_guard_s", (0.01, 0.02, 0.05)),
    ("pipeline", "pre_gap_s", (0.15, 0.30, 0.60)),
    ("nonlinear", "depth_hi_um", (2.5, 3.0, 3.5, 4.0)),
    ("nonlinear", "include_precontact_um", (0.0, 0.5, 1.0, 2.0)),
    ("stiffness", "window_s", (0.02, 0.05, 0.10)),
    ("stiffness", "subtract_noise", (True, False)),
    ("stiffness", "weighted", (True, False)),
    ("film", "thickness_um", (30.0, 50.0, 100.0, 200.0, 1000.0)),
)

KEEP_SETTINGS = {
    ("film", "thickness_um", 30.0): "E_star_film_h30_kPa",
    ("film", "thickness_um", 50.0): "E_star_film_h50_kPa",
    ("film", "thickness_um", 1000.0): "E_star_film_h1000_kPa",
    ("pipeline", "depth_hi_um", 2.5): "E_star_cap2.5um_kPa",
    ("pipeline", "depth_hi_um", 4.0): "E_star_cap4.0um_kPa",
}


def fit_setting(rc, base_p, base_seed_nm, method: str, key: str, value):
    """One estimator at one setting. ``base_p`` is the default preparation.
    """
    if key in PREPARE_KEYS:
        p = lf.prepare(rc, **{key: value})
        if p is None:
            return np.nan
        seed_nm = np.nan
        if method != "pipeline":
            seed = lf.fit_pipeline(p).extra.get("seed_s0_um", np.nan)
            seed_nm = seed * 1000.0
    else:
        p, seed_nm = base_p, base_seed_nm
    kw = {} if key in PREPARE_KEYS else {key: value}
    if method == "pipeline":
        out = lf.fit_pipeline(p, **kw)
    elif not np.isfinite(seed_nm):
        return np.nan
    elif method == "nonlinear":
        out = lf.fit_nonlinear(p, seed_s0_nm=seed_nm, **kw)
    elif method == "stiffness":
        out = lf.fit_stiffness(p, seed_s0_nm=seed_nm, **kw)
    else:
        out = lf.fit_film(p, seed_s0_nm=seed_nm, **kw)
    return out.e_star_kPa if out.ok else np.nan


def analysis_sweep(rc, base_p, base_seed_nm) -> tuple[dict, float]:
    """Every defensible setting on one curve.

    Returns the settings kept as columns, and sigma(ln E*) over the whole
    sweep.
    """
    kept, values = {}, []
    for method, key, options in SETTINGS:
        for value in options:
            e = fit_setting(rc, base_p, base_seed_nm, method, key, value)
            if np.isfinite(e) and e > 0:
                values.append(e)
            name = KEEP_SETTINGS.get((method, key, value))
            if name is not None:
                kept[name] = e
    sigma = (
        float(np.log(np.asarray(values)).std(ddof=1))
        if len(values) > 1
        else np.nan
    )
    return kept, sigma


def one_curve(ident: dict, rc, sweep: bool) -> dict:
    """QC every curve, then measure the retained ones"""
    row = dict(ident)
    row.update(
        acquisition_date=rc.date,
        tip_radius_um=rc.radius_um,
        spring_constant_n_m=rc.spring_constant_n_m,
        calibration_factor=rc.calibration_factor,
        vendor_e_eff_kPa=rc.e_eff_pa / 1000.0,
        vendor_s0_um=rc.vendor_s0_nm / 1000.0,
        vendor_s0_spread_nm=rc.vendor_s0_spread_nm,
        dt_s=rc.dt_s,
    )
    p = lf.prepare(rc)
    if p is None:
        row.update(lq.qc_curve(None, lf.FitOut("pipeline")))
        return row
    fit = lf.fit_pipeline(p)
    row.update(lq.qc_curve(p, fit))
    if not (row["qc_pass"] and fit.ok):
        return row
    row.update(
        E_star_kPa=fit.e_star_kPa,
        contact_s0_um=fit.contact_s0_um,
        depth_lo_um=fit.depth_lo_um,
        depth_hi_um=fit.depth_hi_um,
        n_points=fit.n_points,
        nrmse_percent=fit.extra["nrmse_percent"],
        depth_reached_um=fit.extra["depth_reached_um"],
        onset_before_ramp_s=(
            p.ramp_start_s - p.time_s[p.onset_index]
            if p.onset_index >= 0
            else np.nan
        ),
        ramp_end_header_s=p.ramp_end_s,
        ramp_stop_detected_s=p.extras["ramp_stop_s"],
    )
    row.update(lf.depth_diagnostics(p))
    if sweep:
        seed_nm = fit.extra.get("seed_s0_um", np.nan) * 1000.0
        kept, sigma = analysis_sweep(rc, p, seed_nm)
        row.update(kept)
        row["sigma_analysis"] = sigma
    return row


def maps(df: pd.DataFrame, column: str, label: str, log: bool) -> None:
    """One panel per gel, one colour per campaign.

    """
    for campaign, sub in df.groupby("campaign"):
        gels = list(
            sub[["condition_mg_ml", "gel_number", "sample_id"]]
            .drop_duplicates()
            .sort_values(["condition_mg_ml", "gel_number"])
            .sample_id
        )
        ncol = 4
        nrow = int(np.ceil(len(gels) / ncol))
        fig, axes = plt.subplots(
            nrow,
            ncol,
            figsize=(2.6 * ncol, 2.75 * nrow),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        vmin = sub[column].quantile(0.02)
        vmax = sub[column].quantile(0.98)
        norm = (
            LogNorm(vmin=max(vmin, 1e-2), vmax=vmax)
            if log
            else matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        )
        pitch = PITCH_UM[campaign]
        im = None
        for ax, gel in zip(axes.ravel(), gels, strict=False):
            g = sub[sub.sample_id == gel]
            nx, ny = int(g.x_index.max()), int(g.y_index.max())
            grid = np.full((ny, nx), np.nan)
            for _, r in g.iterrows():
                grid[int(r.y_index) - 1, int(r.x_index) - 1] = r[column]
            im = ax.imshow(
                grid,
                origin="lower",
                cmap="viridis",
                norm=norm,
                interpolation="nearest",
                extent=(0, nx * pitch, 0, ny * pitch),
            )
            ax.set_title(
                gel.replace("_", " mg/mL, gel "),
                fontsize=6.5,
            )
            ax.tick_params(labelsize=5.5)
        for ax in axes.ravel()[len(gels) :]:
            ax.axis("off")
        for ax in axes[-1, :]:
            ax.set_xlabel("x (um)", fontsize=6)
        for ax in axes[:, 0]:
            ax.set_ylabel("y (um)", fontsize=6)
        fig.suptitle(
            f"{CAMPAIGN_LABEL.get(campaign, campaign)}: {label} maps",
            fontsize=9,
        )
        cb = fig.colorbar(
            im, ax=axes.ravel().tolist(), fraction=0.022, pad=0.02
        )
        cb.set_label(label, fontsize=7)
        cb.ax.tick_params(labelsize=6)
        if log:
            candidate = np.array([1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
            ticks = candidate[(candidate >= vmin) & (candidate <= vmax)]
            cb.set_ticks(ticks)
            cb.ax.yaxis.set_major_formatter(
                matplotlib.ticker.ScalarFormatter()
            )
            cb.ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            cb.minorticks_off()
        stem = "modulus" if column == "E_star_kPa" else "contact"
        fig.savefig(OUT / "figures" / f"map_{stem}_{campaign}.png", dpi=220)
        plt.close(fig)


def residual_maps(df: pd.DataFrame, column: str, label: str) -> None:
    """Per-gel detilted residual of ``column``.

    The absolute-contact map does not remove any reference level, so a common
    offset and any stage or sample tilt dominate the colour scale and hide the
    finer spatial structure. Here each gel has a least-squares tilt plane
    z = a*x + b*y + c fitted to its retained points and subtracted, and the
    median of that detrended field removed, so what is drawn is the deviation
    from the gel's own tilted contact plane.
    """
    norm = matplotlib.colors.Normalize(vmin=-5.0, vmax=5.0)

    cmap = matplotlib.colormaps["RdBu_r"].with_extremes(bad="black")
    for campaign, sub in df.groupby("campaign"):
        gels = list(
            sub[["condition_mg_ml", "gel_number", "sample_id"]]
            .drop_duplicates()
            .sort_values(["condition_mg_ml", "gel_number"])
            .sample_id
        )
        ncol = 4
        nrow = int(np.ceil(len(gels) / ncol))
        fig, axes = plt.subplots(
            nrow,
            ncol,
            figsize=(2.6 * ncol, 2.75 * nrow),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        pitch = PITCH_UM[campaign]
        im = None
        for ax, gel in zip(axes.ravel(), gels, strict=False):
            g = sub[sub.sample_id == gel]
            nx, ny = int(g.x_index.max()), int(g.y_index.max())
            x = g.x_index.to_numpy(float)
            y = g.y_index.to_numpy(float)
            z = g[column].to_numpy(float)

            detr = z - np.nanmedian(z)
            if np.isfinite(z).sum() >= 3:
                a = np.column_stack([x, y, np.ones_like(x)])
                if np.linalg.matrix_rank(a) == 3:
                    coeffs, *_ = np.linalg.lstsq(a, z, rcond=None)
                    detr = z - a @ coeffs
            # Residual = detrended value minus the median of the detrended
            # value for the gel.
            resid = detr - np.nanmedian(detr)
            grid = np.full((ny, nx), np.nan)
            for xi, yi, ri in zip(x, y, resid):
                grid[int(yi) - 1, int(xi) - 1] = ri
            im = ax.imshow(
                grid,
                origin="lower",
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                extent=(0, nx * pitch, 0, ny * pitch),
            )
            ax.set_title(
                gel.replace("_", " mg/mL, gel "),
                fontsize=6.5,
            )
            ax.tick_params(labelsize=5.5)
        for ax in axes.ravel()[len(gels) :]:
            ax.axis("off")
        for ax in axes[-1, :]:
            ax.set_xlabel("x (um)", fontsize=6)
        for ax in axes[:, 0]:
            ax.set_ylabel("y (um)", fontsize=6)
        fig.suptitle(
            f"{CAMPAIGN_LABEL.get(campaign, campaign)}: "
            f"{label} detilted residual (per-gel plane removed) maps",
            fontsize=9,
        )
        cb = fig.colorbar(
            im, ax=axes.ravel().tolist(), fraction=0.022, pad=0.02,
        )
        cb.set_label("residual from median contact position [um]", fontsize=7)
        cb.ax.tick_params(labelsize=6)
        fig.savefig(
            OUT / "figures" / f"map_contact_residual_{campaign}.png", dpi=220
        )
        plt.close(fig)


def condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Retention, level, and why curves were dropped, per condition."""
    rows = []
    for (campaign, cond), sub in df.groupby(["campaign", "condition_mg_ml"]):
        kept = sub[sub.qc_pass]
        entry = {
            "campaign": campaign,
            "condition_mg_ml": cond,
            "n_measured": len(sub),
            "n_retained": len(kept),
            "retention": len(kept) / len(sub),
            "n_gels": int(sub.sample_id.nunique()),
            "max_vendor_s0_spread_nm": float(sub.vendor_s0_spread_nm.max()),
            "median_E_star_kPa": float(kept.E_star_kPa.median()),
            "q25_E_star_kPa": float(kept.E_star_kPa.quantile(0.25)),
            "q75_E_star_kPa": float(kept.E_star_kPa.quantile(0.75)),
            "median_contact_s0_um": float(kept.contact_s0_um.median()),
            "median_depth_reached_um": float(kept.depth_reached_um.median()),
        }
        reasons = sub.qc_reasons.fillna("").str.split(";")
        for code in lq.QC_ORDER:
            entry[f"reason_{code}"] = int(
                reasons.apply(lambda parts, c=code: c in parts).sum()
            )
        rows.append(entry)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--no-sweep",
        action="store_true",
        help="skip the analysis sweep (leaves the variance split undefined)",
    )
    args = ap.parse_args()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 210)

    rows = []
    started = time.time()
    for campaign in lr.CAMPAIGNS:
        t0 = time.time()
        for ident, rc in lr.iter_campaign(RAW, campaign):
            rows.append(one_curve(ident, rc, not args.no_sweep))
        print(f"{campaign}: {len(rows)} curves, {time.time() - t0:.0f} s")
    everything = pd.DataFrame(rows)
    kept = everything[everything.qc_pass].copy()

    kept["sigma_measurement"] = kept["e_star_rel_sigma"]
    columns = [
        "campaign",
        "sample_id",
        "condition_mg_ml",
        "gel_number",
        "x_index",
        "y_index",
        "curve_id",
        "relative_path",
        "acquisition_date",
        "tip_radius_um",
        "spring_constant_n_m",
        "calibration_factor",
        "dt_s",
        "E_star_kPa",
        "contact_s0_um",
        "depth_lo_um",
        "depth_hi_um",
        "depth_reached_um",
        "n_points",
        "nrmse_percent",
        "onset_before_ramp_s",
        "ramp_end_header_s",
        "ramp_stop_detected_s",
        "vendor_e_eff_kPa",
        "vendor_s0_um",
        "sigma_analysis",
        "sigma_measurement",
        *KEEP_SETTINGS.values(),
        *(c for c in kept.columns if c.startswith(("ladder_E_", "Ept_"))),
        "k2_slope_ratio",
    ]
    columns = [c for c in columns if c in kept.columns]
    kept[columns].to_csv(OUT / "per_curve.csv", index=False)
    summary = condition_summary(everything)
    summary.to_csv(OUT / "condition_summary.csv", index=False)

    print(
        f"\nretained {len(kept)} of {len(everything)} curves "
        f"({100 * len(kept) / len(everything):.1f} %) in "
        f"{time.time() - started:.0f} s"
    )
    print(
        summary[
            [
                "campaign",
                "condition_mg_ml",
                "n_measured",
                "n_retained",
                "retention",
                "median_E_star_kPa",
                "median_depth_reached_um",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    counts = {
        code: int(summary[f"reason_{code}"].sum()) for code in lq.QC_ORDER
    }
    print("\nfailures by criterion:")
    for code, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        if n:
            print(f"  {code:<24}{n:4d}")
    print(
        "\nvendor Indentation channel: max spread of s - Indentation = "
        f"{summary.max_vendor_s0_spread_nm.max():.1e} nm over "
        f"{len(everything)} curves"
    )

    maps(kept, "E_star_kPa", "$E^*$ (kPa)", log=True)
    maps(kept, "contact_s0_um", "contact position $s_0$ (um)", log=False)
    residual_maps(kept, "contact_s0_um", "contact position $s_0$")
    print(f"wrote {OUT / 'per_curve.csv'} and 9 map figures")


if __name__ == "__main__":
    main()
