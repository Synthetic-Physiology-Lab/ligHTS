#!/usr/bin/env python3
"""
gelma_indentation_suite.py
============================================================================
Publication-ready analysis suite for Optics11 Chiaro spherical
nanoindentation of GelMA hydrogels.

Point it at ONE session folder (all gels of one characterisation session, e.g.
12 gels = 4 replicates x 3 concentrations, each gel a sub-folder of raw .txt
force curves) and it produces, with no manual steps:

  PER MATRIX (one sub-folder per gel)
    - per_curve_metrics.csv          instrument-extracted + recalculated values,
                                     per-QC-gate pass/fail, thresholds, outlier flag
    - <sample_id>_curve_fits.eps/.jpg   every force curve with contact point + Hertz fit
    - <sample_id>_spatial_maps.eps/.jpg 2D maps over the X-Y grid: apparent modulus |
                                     contact point | QC / FOV technical success
    - matrix_report.txt

  PER SESSION (top-level / general folder)
    - master_per_curve.csv           every curve of the session, one row each
    - per_matrix_summary.csv         one row per gel
    - per_condition_summary.csv      one row per concentration
    - statistics.csv                 ANOVA + pairwise (Welch/Holm, exact perm, Hedges g)
    - superplot.eps/.jpg             per-curve + gel-median + condition-mean figure
    - modulus_map_atlas.eps/.jpg     every gel's modulus map, tiled by condition
    - contactpoint_map_atlas.eps/.jpg every gel's contact-point map
    - sensitivity_analysis.eps/.jpg + sensitivity_analysis.csv   stability of the
                                     result to contact rule, fit window, processing
    - session_report.txt

  Every figure is written as vector EPS and raster JPEG and follows scientific-
  illustration conventions: no on-figure titles, units in square brackets,
  capitalised axis quantities, perceptually-uniform / colourblind-safe palettes.
  Maps draw only the measured grid points; un-sampled lattice cells stay blank.

METHOD (apparent reduced indentation modulus, soft-matter Hertz sphere)
------------------------------------------------------------------
  * depth axis zeta = piezo - cantilever (removes cantilever compliance);
  * robust pre-contact baseline (line + MAD scatter), subtracted;
  * OBJECTIVE R^2-maximising contact point over a fixed 0-4 um window
    (coarse 40 nm scan, refined to 5 nm) - no hand-picked contact;
  * zero-intercept Hertz sphere  F = (4/3) E_app sqrt(R) delta^1.5  over 0-4 um,
    slope -> E_app (apparent reduced modulus; reported endpoint);
  * late-window (1.5-4 um) self-consistency diagnostic;
  * pre-registered QC gates + within-gel outlier screen (modified z on log10 E);
  * gel = experimental unit; gel median -> per-condition summary + statistics.

This is a single, dependency-light file (numpy / pandas / scipy / matplotlib;
tkinter only for the optional folder chooser).

USAGE
------------------------------------------------------------------
  # GUI: a window asks you to pick the session folder
  python gelma_indentation_suite.py

  # headless / scripted
  python gelma_indentation_suite.py --raw <session_folder> --out <output_folder>

Input folders: one sub-folder per gel, named  GelMA_<conc>mgmL_gel<n>[_...]
(concentration in mg/mL). Percent-w/v labels (7.5/10/15) and the raw
matrix_scan.._<pct>_<rep> instrument names are also recognised. Curve files
keep the instrument name  <label> S-1 X-<col> Y-<row> I-01.txt .
============================================================================
"""
from __future__ import annotations

import argparse
import io as _io
import math
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
CFG = {
    "probe": {"tip_radius_m": 27.0e-6, "cantilever_k_n_per_m": 0.48},
    "contact": {
        "smoothing_window": 15, "min_branch_points": 80,
        "baseline_fraction": 0.25, "baseline_min_points": 40,
        "sigma_threshold": 5.0, "threshold_run_fraction": 0.8,
        "threshold_window_divisor": 100, "threshold_window_floor": 15,
        "search_low_offset_nm": 1200.0, "search_high_margin_nm": 1500.0,
        "search_high_floor_offset_nm": 1200.0, "coarse_step_nm": 40.0,
        "refine_half_width_nm": 40.0, "refine_step_nm": 5.0,
        "min_fitted_depth_nm": 3500.0,
    },
    "hertz": {"cap_nm": 4000.0, "fit_min_points": 10, "late_window_low_nm": 1500.0},
    "qc": {"snr_min": 30.0, "depth_min_nm": 3900.0, "depth_max_nm": 6500.0,
           "r2_min": 0.97, "nrmse_max": 0.05, "late_ratio_min": 0.85,
           "late_ratio_max": 1.15, "e_min_kpa": 0.2, "e_max_kpa": 50.0,
           "outlier_mod_z_threshold": 3.0},
    "topography": {"default_pitch_um": 300.0, "min_points_per_gel": 6},
    "sensitivity": {
        "caps_nm": [2000, 2500, 3000, 3500, 4000],
        "z0_shifts_nm": [-150, -100, -50, 0, 50, 100, 150],
        "smoothing_windows": [11, 15, 19],
        "baseline_fractions": [0.20, 0.25, 0.30],
        "sigma_thresholds": [3.0, 5.0, 7.0],
    },
    "stats": {"ci_level": 0.95},
    "palette": {75: "#E69F00", 100: "#009E73", 150: "#0072B2",
                125: "#CC79A7", 200: "#D55E00", 50: "#56B4E9"},
}

# ---- folder-name -> (concentration mg/mL, replicate) -----------------------
_PCT_TO_MGML = {"7.5": 75, "10": 100, "15": 150, "5": 50, "20": 200, "12.5": 125}


def parse_folder_identity(name: str):
    """Return (conc_mgml, gel, tag) from a matrix folder name, or None."""
    m = re.search(r"GelMA_(\d+)mgmL_gel(\d+)(?:_(\w+))?", name, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2)), (m.group(3) or "")
    # instrument name: matrix_scan<NN>_<pct>_<rep>
    m = re.search(r"scan(\d+)_(\d+(?:\.\d+)?)_(\d+)", name)
    if m:
        pct, rep = m.group(2), int(m.group(3))
        return _PCT_TO_MGML.get(pct, int(round(float(pct)))), rep, f"scan{m.group(1)}"
    # bare <pct>_<rep>
    m = re.fullmatch(r"(\d+(?:\.\d+)?)_(\d+)", name)
    if m:
        pct, rep = m.group(1), int(m.group(2))
        return _PCT_TO_MGML.get(pct, int(round(float(pct)))), rep, ""
    return None


XY_RE = re.compile(r"X-(\d+)\s+Y-(\d+)")


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def _grab(text: str, key: str) -> float:
    """Return the first number following ``key`` in the file header, or NaN if absent."""
    m = re.search(re.escape(key) + r"\s+([-\d.eE+]+)", text)
    return float(m.group(1)) if m else np.nan


def parse_curve(path: Path):
    """Read one Chiaro .txt curve: instrument metadata plus the time / load /
    cantilever / piezo columns of the data block."""
    text = path.read_text(encoding="latin-1")
    lines = text.splitlines()
    meta = {
        "instr_k_Nm": _grab(text, "k (N/m)"),
        "instr_R_um": _grab(text, "Tip radius (um)"),
        "instr_Eeff_kPa": _grab(text, "E[eff] (Pa)") / 1e3,
        "instr_Zsurface_um": _grab(text, "Z surface (um)"),
        "instr_Zposition_um": _grab(text, "Z-position (um)"),
        "instr_Pmax_uN": _grab(text, "P[max] (uN)"),
        "instr_status": "OK" if re.search(r"Status\s+OK", text) else "NOK",
    }
    hdr = [i for i, l in enumerate(lines) if l.startswith("Time (s)")]
    if not hdr:
        raise ValueError(f"No 'Time (s)' data header in {path.name}")
    arr = np.loadtxt(_io.StringIO("\n".join(lines[hdr[0] + 1:])))
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError(f"Unexpected data shape in {path.name}")
    t, load, ind, cant, piezo = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    return meta, {"t": t, "load": load, "cant": cant, "piezo": piezo}


def discover_matrices(raw_root: Path):
    """List gel sub-folders that have a parseable identity and at least one X-/Y- curve file."""
    out = []
    for d in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        ident = parse_folder_identity(d.name)
        if ident is None:
            continue
        if not any(XY_RE.search(f.name) for f in d.glob("*.txt")):
            continue
        conc, gel, tag = ident
        out.append({"path": d, "name": d.name, "conc": conc, "gel": gel, "tag": tag})
    return out


# ----------------------------------------------------------------------------
# PER-CURVE ANALYSIS
# ----------------------------------------------------------------------------
def loading_branch(data, w):
    """Extract the loading branch: depth axis zeta = piezo - cantilever and force,
    truncated at the smoothed force peak and sorted by increasing depth."""
    load = data["load"]
    smoothed = np.convolve(load, np.ones(w) / w, mode="same")
    i_peak = int(np.argmax(smoothed))
    zeta = (data["piezo"] - data["cant"])[: i_peak + 1]
    force = load[: i_peak + 1]
    order = np.argsort(zeta)
    return zeta[order], force[order]


def fit_baseline(zeta, force, frac, min_pts):
    """Fit a line to the pre-contact region; return (slope, intercept, MAD scatter,
    baseline-subtracted force)."""
    n = len(zeta)
    n_pre = max(min_pts, int(frac * n))
    z, f = zeta[:n_pre], force[:n_pre]
    coef, *_ = np.linalg.lstsq(np.vstack([z, np.ones_like(z)]).T, f, rcond=None)
    resid = f - (coef[0] * z + coef[1])
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.std(resid) or 1e-4
    return coef[0], coef[1], float(sigma), force - (coef[0] * zeta + coef[1])


def threshold_contact(zeta, force_corr, sigma, p):
    """Seed contact point: first depth where the corrected force stays above
    sigma_threshold x baseline scatter for a run of consecutive points."""
    n = len(zeta)
    above = force_corr > p["sigma_threshold"] * sigma
    run = max(p["threshold_window_floor"], n // p["threshold_window_divisor"])
    for i in range(n):
        if above[i] and above[i: i + run].mean() > p["threshold_run_fraction"]:
            return float(zeta[i])
    return float(zeta[int(0.5 * n)])


def _s2m(R):
    """Slope-to-modulus factor for the Hertz sphere: E = slope / ((4/3) sqrt(R))."""
    return 1.0 / ((4.0 / 3.0) * math.sqrt(R))


def fit_window(fcorr, zeta, z0, R, cap_nm, min_pts, lo_nm=0.0):
    """Fit the zero-intercept Hertz sphere over the depth window (lo_nm, cap_nm].
    Return slope, apparent modulus [kPa], R2, NRMSE, point count and max depth,
    or None if the window holds fewer than ``min_pts`` points."""
    delta = (zeta - z0) * 1e-9
    mask = (delta > lo_nm * 1e-9) & (delta <= cap_nm * 1e-9) & (fcorr > 0)
    if mask.sum() < min_pts:
        return None
    x = delta[mask] ** 1.5
    y = fcorr[mask] * 1e-6
    slope = np.sum(x * y) / np.sum(x * x)
    sse = np.sum((y - slope * x) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - sse / sst if sst > 0 else np.nan
    nrmse = math.sqrt(sse / len(y)) / (y.max() - y.min()) if y.max() > y.min() else np.nan
    return {"slope": float(slope), "E_kPa": float(slope * _s2m(R)) / 1e3,
            "R2": float(r2), "nrmse": float(nrmse), "n": int(mask.sum()),
            "dmax_nm": float(delta[mask].max() * 1e9)}


def analyse_curve(data, cfg, sensitivity=True):
    """Full per-curve analysis: baseline subtraction, objective R2-maximising
    contact point (coarse then refined scan), Hertz fit over 0-cap, self-consistency
    diagnostic and, when ``sensitivity`` is set, the alternative-setting re-fits."""
    R = cfg["probe"]["tip_radius_m"]
    c, h = cfg["contact"], cfg["hertz"]
    cap, mp = h["cap_nm"], h["fit_min_points"]
    zeta, force = loading_branch(data, c["smoothing_window"])
    if len(zeta) < c["min_branch_points"]:
        return {"fail": "short_branch"}
    bs, bi, sigma, fcorr = fit_baseline(zeta, force, c["baseline_fraction"], c["baseline_min_points"])
    z0_thr = threshold_contact(zeta, fcorr, sigma, c)
    peak = float(fcorr.max())
    lo = z0_thr - c["search_low_offset_nm"]
    hi = max(zeta[-1] - c["search_high_margin_nm"], z0_thr + c["search_high_floor_offset_nm"])
    grid = np.arange(lo, hi, c["coarse_step_nm"])

    def score(res):
        if res is None or not np.isfinite(res["R2"]) or res["dmax_nm"] < c["min_fitted_depth_nm"]:
            return -np.inf
        return res["R2"]

    best = None
    for z0 in grid:
        r = fit_window(fcorr, zeta, z0, R, cap, mp)
        s = score(r)
        if best is None or s > best[1]:
            best = (z0, s, r)
    if best is None or best[2] is None or not np.isfinite(best[1]):
        return {"fail": "no_valid_contact_fit"}
    for z0 in np.arange(best[0] - c["refine_half_width_nm"], best[0] + c["refine_half_width_nm"], c["refine_step_nm"]):
        r = fit_window(fcorr, zeta, z0, R, cap, mp)
        s = score(r)
        if s > best[1]:
            best = (z0, s, r)
    z0, fit = best[0], best[2]
    late = fit_window(fcorr, zeta, z0, R, cap, mp, lo_nm=h["late_window_low_nm"])
    out = {
        "z0_zeta_nm": float(z0), "z0_thr_nm": float(z0_thr),
        "baseline_slope": bs, "baseline_int": bi, "baseline_sigma_uN": sigma,
        "E_app_kPa": fit["E_kPa"], "R2": fit["R2"], "nrmse": fit["nrmse"],
        "slope_cap4": fit["slope"], "n_fit": fit["n"], "dmax_nm": fit["dmax_nm"],
        "a_contact_um": math.sqrt(R * fit["dmax_nm"] * 1e-9) * 1e6,
        "late_ratio": (late["E_kPa"] / fit["E_kPa"]) if (late and fit["E_kPa"]) else np.nan,
        "peak_force_uN": peak, "snr": peak / sigma,
        "depth_nm": float(zeta[-1] - z0),
    }
    if sensitivity:
        for cap_nm in cfg["sensitivity"]["caps_nm"]:
            cf = fit_window(fcorr, zeta, z0, R, cap_nm, mp)
            out[f"E_cap{cap_nm}"] = cf["E_kPa"] if cf else np.nan
        for sh in cfg["sensitivity"]["z0_shifts_nm"]:
            sf = fit_window(fcorr, zeta, z0 + sh, R, cap, mp)
            out[f"E_shift{sh}"] = sf["E_kPa"] if sf else np.nan
        tf = fit_window(fcorr, zeta, z0_thr, R, cap, mp)
        out["E_thresholdcontact"] = tf["E_kPa"] if tf else np.nan
        devs = [abs(out[f"E_shift{s}"] - fit["E_kPa"]) / fit["E_kPa"]
                for s in (-100, 100) if np.isfinite(out.get(f"E_shift{s}", np.nan))]
        out["z0_sensitivity_frac"] = float(max(devs)) if devs else np.nan
    return out


# ----------------------------------------------------------------------------
# QC + outliers + roughness
# ----------------------------------------------------------------------------
QC_GATES = ["status", "snr", "depth", "R2", "nrmse", "late_ratio", "E_range"]


def qc_gate(row, q):
    """Apply the pre-registered QC gates to one curve row; return
    (passed, '|'-joined failed-gate names, per-gate boolean dict)."""
    fail = row.get("fail")
    fail = fail if isinstance(fail, str) and fail else ""
    if fail or not np.isfinite(row.get("E_app_kPa", np.nan)):
        g = {f"qc_{k}": False for k in QC_GATES}
        return False, (fail or "no_fit"), g
    g = {
        "qc_status": row["instr_status"] == "OK",
        "qc_snr": row["snr"] >= q["snr_min"],
        "qc_depth": q["depth_min_nm"] <= row["depth_nm"] <= q["depth_max_nm"],
        "qc_R2": row["R2"] >= q["r2_min"],
        "qc_nrmse": row["nrmse"] <= q["nrmse_max"],
        "qc_late_ratio": q["late_ratio_min"] <= row["late_ratio"] <= q["late_ratio_max"],
        "qc_E_range": q["e_min_kpa"] <= row["E_app_kPa"] <= q["e_max_kpa"],
    }
    reasons = [k.replace("qc_", "") for k, v in g.items() if not v]
    return (len(reasons) == 0), "|".join(reasons), g


def flag_outliers(df, thr):
    """Within a gel, flag QC-passing curves whose log10 E is a modified-z outlier,
    then set the ``included`` column (QC pass and not an outlier)."""
    df = df.copy()
    df["is_outlier"] = False
    passing = df[df["QC_pass"]]
    if len(passing) >= 3:
        logE = np.log10(passing["E_app_kPa"])
        med = logE.median()
        mad = 1.4826 * (logE - med).abs().median()
        if mad > 0:
            z = (logE - med).abs() / mad
            df.loc[passing.index[z > thr], "is_outlier"] = True
    df["included"] = df["QC_pass"] & ~df["is_outlier"]
    return df


def detrended_roughness(df, pitch, min_pts):
    """Plane-detrended surface roughness Rq [nm] and tilt [nm/um] from the
    per-point contact heights of the included curves."""
    sub = df[df["included"] & df["z0_zeta_nm"].notna()]
    if len(sub) < min_pts:
        return np.nan, np.nan
    x, y = sub["X"].to_numpy() * pitch, sub["Y"].to_numpy() * pitch
    hgt = sub["z0_zeta_nm"].to_numpy()
    A = np.vstack([x, y, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, hgt, rcond=None)
    resid = hgt - A @ coef
    return float(np.std(resid)), float(math.hypot(coef[0], coef[1]))


# ----------------------------------------------------------------------------
# PER-MATRIX
# ----------------------------------------------------------------------------
def analyse_matrix(mat, cfg, out_dir):
    """Analyse one gel folder: fit every curve, run QC and the outlier screen,
    then write per_curve_metrics.csv, the curve-fit montage, the spatial maps and
    the matrix report; return the per-curve frame and the one-row gel summary."""
    files = sorted(f for f in mat["path"].glob("*.txt") if XY_RE.search(f.name))
    rows, cache = [], {}
    for f in files:
        xy = XY_RE.search(f.name)
        rec = {"sample_id": f"GelMA_{mat['conc']}mgmL_gel{mat['gel']}",
               "conc": mat["conc"], "gel": mat["gel"], "source": mat["tag"],
               "file": f.name, "X": int(xy.group(1)), "Y": int(xy.group(2))}
        try:
            meta, data = parse_curve(f)
            rec.update(meta)
            res = analyse_curve(data, cfg)
            rec.update(res)
            cache[f.name] = (data, res)
        except Exception as exc:
            rec["instr_status"] = rec.get("instr_status", "NOK")
            rec["fail"] = f"parse_err:{exc}"
        rows.append(rec)
    df = pd.DataFrame(rows)
    # Guarantee the per-curve analysis columns exist even when every curve in
    # this gel failed to fit; without this, QC, spatial maps and the summary
    # below would KeyError on an all-fail matrix and abort the whole session.
    for col in ("E_app_kPa", "R2", "nrmse", "snr", "depth_nm", "late_ratio",
                "z0_zeta_nm", "z0_thr_nm", "a_contact_um", "slope_cap4",
                "baseline_slope", "baseline_int", "z0_sensitivity_frac"):
        if col not in df.columns:
            df[col] = np.nan

    q = cfg["qc"]
    passes, reasons, gates = [], [], []
    for _, r in df.iterrows():
        p, why, g = qc_gate(r, q)
        passes.append(p); reasons.append(why); gates.append(g)
    df["QC_pass"] = passes
    df["QC_fail_reason"] = reasons
    for k in [f"qc_{g}" for g in QC_GATES]:
        df[k] = [gg[k] for gg in gates]
    df = flag_outliers(df, q["outlier_mod_z_threshold"])

    mat_out = out_dir / mat["name"]
    mat_out.mkdir(parents=True, exist_ok=True)
    # master per-curve for the gel: instrument + calculated + QC + thresholds
    df["depth_min_thr_nm"] = q["depth_min_nm"]; df["depth_max_thr_nm"] = q["depth_max_nm"]
    df["R2_min_thr"] = q["r2_min"]; df["snr_min_thr"] = q["snr_min"]
    keep = (["sample_id", "conc", "gel", "source", "file", "X", "Y",
             "instr_status", "instr_k_Nm", "instr_R_um", "instr_Eeff_kPa",
             "instr_Zsurface_um", "instr_Zposition_um", "instr_Pmax_uN",
             "z0_zeta_nm", "z0_thr_nm", "E_app_kPa", "R2", "nrmse", "snr",
             "depth_nm", "a_contact_um", "late_ratio", "z0_sensitivity_frac"]
            + [f"qc_{g}" for g in QC_GATES]
            + ["QC_pass", "is_outlier", "included", "QC_fail_reason",
               "snr_min_thr", "R2_min_thr", "depth_min_thr_nm", "depth_max_thr_nm"])
    df.reindex(columns=[c for c in keep if c in df.columns]).to_csv(
        mat_out / "per_curve_metrics.csv", index=False)

    inc = df[df["included"]]
    e = inc["E_app_kPa"].dropna()
    rq, tilt = detrended_roughness(df, cfg["topography"]["default_pitch_um"],
                                   cfg["topography"]["min_points_per_gel"])
    summ = {"sample_id": f"GelMA_{mat['conc']}mgmL_gel{mat['gel']}", "conc_mgml": mat["conc"],
            "gel": mat["gel"], "source": mat["tag"], "n_curves": len(df),
            "n_included": len(inc), "n_qc_fail": int((~df["QC_pass"]).sum()),
            "n_outlier": int(df["is_outlier"].sum()),
            "technical_success_pct": round(100 * df["QC_pass"].mean(), 1),
            "median_E_kPa": round(e.median(), 3) if len(e) else np.nan,
            "geomean_E_kPa": round(float(np.exp(np.log(e).mean())), 3) if len(e) else np.nan,
            "cv_pct": round(100 * e.std(ddof=1) / e.mean(), 1) if len(e) > 1 else np.nan,
            "median_R2": round(inc["R2"].median(), 4) if len(inc) else np.nan,
            "Rq_detrended_nm": round(rq, 1) if np.isfinite(rq) else np.nan,
            "tilt_nm_per_um": round(tilt, 4) if np.isfinite(tilt) else np.nan}

    sid = summ["sample_id"]
    _montage(df, cache, mat, summ, mat_out / f"{sid}_curve_fits", cfg)
    _spatial_maps(df, mat, summ, mat_out / f"{sid}_spatial_maps")
    with open(mat_out / "matrix_report.txt", "w") as fh:
        fh.write(f"Matrix: {mat['name']}\nConcentration: {mat['conc']} mg/mL   "
                 f"Gel: {mat['gel']}   Source: {mat['tag']}\n"
                 f"Curves: {summ['n_curves']}  included: {summ['n_included']}  "
                 f"QC-fail: {summ['n_qc_fail']}  outliers: {summ['n_outlier']}  "
                 f"(technical success {summ['technical_success_pct']} %)\n"
                 f"Gel median E_app: {summ['median_E_kPa']} kPa "
                 f"(geometric mean {summ['geomean_E_kPa']} kPa, CV {summ['cv_pct']} %)\n"
                 f"Median fit R^2: {summ['median_R2']}   "
                 f"Detrended roughness Rq: {summ['Rq_detrended_nm']} nm   "
                 f"tilt: {summ['tilt_nm_per_um']} nm/um\n")
    return df, summ


# ----------------------------------------------------------------------------
# FIGURES
# ----------------------------------------------------------------------------
def save_fig(fig, stem, dpi=300):
    """Write a figure as vector EPS and raster JPEG (publication formats)."""
    for ext in ("eps", "jpg"):
        fig.savefig(f"{stem}.{ext}", dpi=dpi)
    plt.close(fig)


def _grid_array(df, value_col, mask_col=None):
    """Build a 2D (Y,X) array of value_col; cells absent/failed -> NaN."""
    xs, ys = sorted(df["X"].unique()), sorted(df["Y"].unique())
    xi = {x: i for i, x in enumerate(xs)}
    yi = {y: i for i, y in enumerate(ys)}
    grid = np.full((len(ys), len(xs)), np.nan)
    for _, r in df.iterrows():
        if mask_col is not None and not r.get(mask_col, False):
            continue
        v = r.get(value_col, np.nan)
        if np.isfinite(v):
            grid[yi[r["Y"]], xi[r["X"]]] = v
    return grid, xs, ys


def _spatial_maps(df, mat, summ, stem):
    """Three X-Y grid maps for one gel: apparent modulus, contact point, and a
    QC / FOV category (included, outlier, QC-fail, no fit). Un-sampled cells blank."""
    n_acq = len(df)
    n_fit = int(df["E_app_kPa"].notna().sum())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    gE, xs, ys = _grid_array(df[df["E_app_kPa"].notna()], "E_app_kPa")
    im0 = axes[0].imshow(gE, origin="lower", cmap="viridis", aspect="equal")
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label("Apparent modulus [kPa]")
    hi = np.nanmax(gE) if np.isfinite(np.nanmax(gE)) else 1.0
    for (j, i), v in np.ndenumerate(gE):
        if np.isfinite(v):
            axes[0].text(i, j, f"{v:.1f}", ha="center", va="center", fontsize=6,
                         color="w" if v < hi * 0.55 else "k")
    gZ, _, _ = _grid_array(df[df["z0_zeta_nm"].notna()], "z0_zeta_nm")
    im1 = axes[1].imshow(gZ / 1000, origin="lower", cmap="cividis", aspect="equal")
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.set_label(r"Contact point $z_0$ [$\mu$m]")
    # QC / FOV: un-measured lattice cells stay blank; only measured points get a category
    xi = {x: i for i, x in enumerate(xs)}
    yi = {y: i for i, y in enumerate(ys)}
    cat = np.full((len(ys), len(xs)), np.nan)
    for _, r in df.iterrows():
        if r["X"] in xi and r["Y"] in yi:
            if r.get("included"):
                c = 3
            elif r.get("is_outlier"):
                c = 2
            elif np.isfinite(r.get("E_app_kPa", np.nan)):
                c = 1
            else:
                c = 0
            cat[yi[r["Y"]], xi[r["X"]]] = c
    cmap = ListedColormap(["#999999", "#E69F00", "#CC79A7", "#009E73"])
    cmap.set_bad("white")
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    axes[2].imshow(np.ma.masked_invalid(cat), origin="lower", cmap=cmap, norm=norm, aspect="equal")
    from matplotlib.patches import Patch
    axes[2].legend(handles=[Patch(color="#009E73", label="Included"),
                            Patch(color="#CC79A7", label="Outlier"),
                            Patch(color="#E69F00", label="QC-fail"),
                            Patch(color="#999999", label="No fit")],
                   loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False)
    for ax in axes:
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, fontsize=8)
        ax.set_yticks(range(len(ys))); ax.set_yticklabels(ys, fontsize=8)
        ax.set_xlabel("Grid position X"); ax.set_ylabel("Grid position Y")
    lbl = f"n = {n_acq} acquired" + (f" ({n_fit} fitted)" if n_fit != n_acq else "")
    axes[0].text(0.0, 1.02, lbl, transform=axes[0].transAxes, fontsize=9, va="bottom")
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.16, top=0.93, wspace=0.3)
    save_fig(fig, stem)


def _montage(df, cache, mat, summ, stem, cfg):
    """Tile every curve of a gel (baseline-corrected force vs depth) with its Hertz
    fit and contact point; panel colour marks included / QC-fail / outlier."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    sw = cfg["contact"]["smoothing_window"]
    sub = df.sort_values(["X", "Y"])
    ncol = int(np.ceil(np.sqrt(len(sub)))); nrow = int(np.ceil(len(sub) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    color = {"inc": "#009E73", "qc": "#E69F00", "out": "#CC79A7"}
    for ax, (_, r) in zip(axes, sub.iterrows()):
        c = color["inc"] if r["included"] else (color["out"] if r.get("is_outlier") else color["qc"])
        rec = cache.get(r["file"])
        if rec is not None and np.isfinite(r.get("z0_zeta_nm", np.nan)):
            data, _ = rec
            zeta, force = loading_branch(data, sw)
            fcorr = force - (r["baseline_slope"] * zeta + r["baseline_int"])
            dl = zeta - r["z0_zeta_nm"]
            ax.plot(dl / 1000, fcorr, color="0.4", lw=0.6)
            if np.isfinite(r.get("slope_cap4", np.nan)):
                m = (dl > 0) & (dl <= cfg["hertz"]["cap_nm"])
                ax.plot(dl[m] / 1000, r["slope_cap4"] * (np.clip(dl, 0, None) * 1e-9)[m] ** 1.5 * 1e6,
                        color="#D55E00", lw=0.9)
            ax.scatter([0], [0], color="k", s=6, zorder=5)
        lbl = (f"X{int(r.X)}Y{int(r.Y)}  {r['E_app_kPa']:.1f}" if np.isfinite(r.get("E_app_kPa", np.nan))
               else f"X{int(r.X)}Y{int(r.Y)}  fail")
        ax.text(0.04, 0.93, lbl, transform=ax.transAxes, fontsize=6, color=c, va="top")
        ax.tick_params(labelsize=4)
        for s in ax.spines.values():
            s.set_color(c); s.set_linewidth(1.2)
    for ax in axes[len(sub):]:
        ax.set_visible(False)
    fig.legend(handles=[Patch(color="#009E73", label="Included"), Patch(color="#E69F00", label="QC-fail"),
                        Patch(color="#CC79A7", label="Outlier"),
                        Line2D([], [], color="0.4", label="Loading branch"),
                        Line2D([], [], color="#D55E00", label=r"Hertz fit (0-4 $\mu$m)")],
               loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.supxlabel(r"Indentation depth [$\mu$m]"); fig.supylabel(r"Force [$\mu$N]")
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.98, wspace=0.35, hspace=0.42)
    save_fig(fig, stem, dpi=200)


def _map_atlas(master, per_matrix, value_col, cbar_label, stem, cmap):
    """Tile every gel's map of ``value_col`` on a shared colour scale, one row per
    concentration, for a session-wide overview."""
    concs = sorted(per_matrix["conc_mgml"].unique())
    ncol = max(per_matrix.groupby("conc_mgml").size())
    fig, axes = plt.subplots(len(concs), ncol, figsize=(2.6 * ncol, 2.6 * len(concs)), squeeze=False)
    vmin = np.nanpercentile(master[value_col], 2)
    vmax = np.nanpercentile(master[value_col], 98)
    im = None
    for ri, conc in enumerate(concs):
        gels = per_matrix[per_matrix["conc_mgml"] == conc].sort_values("gel")
        for ci in range(ncol):
            ax = axes[ri][ci]
            if ci < len(gels):
                sid = gels.iloc[ci]["sample_id"]
                sub = master[master["sample_id"] == sid]
                g, xs, ys = _grid_array(sub[sub[value_col].notna()], value_col)
                im = ax.imshow(g, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
                ax.text(0.5, 1.04, f"{conc} mg mL$^{{-1}}$ gel {int(gels.iloc[ci]['gel'])}",
                        transform=ax.transAxes, ha="center", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                ax.set_visible(False)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cax).set_label(cbar_label)
    fig.subplots_adjust(left=0.03, right=0.9, bottom=0.05, top=0.93, wspace=0.15, hspace=0.3)
    save_fig(fig, stem)


def superplot(master, per_matrix, stem):
    """Superplot per concentration: light per-curve points, gel-median markers and
    the condition mean, on a log modulus axis."""
    concs = sorted(per_matrix["conc_mgml"].unique())
    fig, ax = plt.subplots(figsize=(1.9 * len(concs) + 2, 5.6))
    for i, c in enumerate(concs):
        col = CFG["palette"].get(c, "#555555")
        pts = master[(master["conc"] == c) & master["included"]]
        sc = ax.scatter(np.random.default_rng(c).normal(i, 0.06, len(pts)), pts["E_app_kPa"],
                        s=8, color=col, alpha=0.25, zorder=2)
        sc.set_rasterized(True)
        gm = per_matrix[per_matrix["conc_mgml"] == c]["median_E_kPa"].dropna()
        ax.scatter([i] * len(gm), gm, s=90, color=col, edgecolors="k", lw=1.0, zorder=4)
        ax.hlines(gm.mean(), i - 0.25, i + 0.25, color="k", lw=2, zorder=5)
    ax.set_xticks(range(len(concs))); ax.set_xticklabels([str(c) for c in concs])
    ax.set_xlabel(r"Concentration [mg mL$^{-1}$]")
    ax.set_yscale("log"); ax.set_ylabel(r"Apparent modulus $E_\mathrm{app}$ [kPa]")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="0.93")
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.1, top=0.97)
    save_fig(fig, stem)


# ----------------------------------------------------------------------------
# STATISTICS
# ----------------------------------------------------------------------------
def _hedges_g(a, b):
    """Hedges' g effect size (bias-corrected standardised mean difference) for b vs a."""
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    d = (b.mean() - a.mean()) / sp if sp > 0 else np.nan
    return d * (1 - 3 / (4 * (na + nb) - 9))


def _perm_two(a, b):
    """Exact two-sided permutation p-value for a Welch t statistic between a and b."""
    from scipy import stats
    obs = abs(stats.ttest_ind(b, a, equal_var=False).statistic)
    pool = np.concatenate([a, b]); n = len(a); cnt = tot = 0
    for idx in combinations(range(len(pool)), n):
        mask = np.zeros(len(pool), bool); mask[list(idx)] = True
        if abs(stats.ttest_ind(pool[~mask], pool[mask], equal_var=False).statistic) >= obs - 1e-12:
            cnt += 1
        tot += 1
    return cnt / tot


def _perm_anova(groups, cap=200000):
    """Exact permutation p-value for the one-way ANOVA F statistic; returns NaN once
    the number of partitions exceeds ``cap``."""
    from scipy import stats
    from itertools import combinations as C
    obs = stats.f_oneway(*groups).statistic
    sizes = [len(g) for g in groups]; pool = np.concatenate(groups); n = len(pool)

    def parts(remaining, sizes):
        if not sizes:
            yield []; return
        for first in C(sorted(remaining), sizes[0]):
            for rest in parts(remaining - set(first), sizes[1:]):
                yield [list(first)] + rest
    cnt = tot = 0
    for p in parts(set(range(n)), sizes):
        if stats.f_oneway(*[pool[np.array(ix)] for ix in p]).statistic >= obs - 1e-9:
            cnt += 1
        tot += 1
        if tot > cap:
            return np.nan
    return cnt / tot


def compute_statistics(per_matrix, cfg):
    """Condition-level statistics from gel-median moduli: per-condition mean, SD,
    geometric mean and CI; omnibus ANOVA/Kruskal with exact permutation; and Welch
    pairwise tests with Holm correction, exact permutation and Hedges' g."""
    from scipy import stats
    concs = sorted(per_matrix["conc_mgml"].unique())
    groups = {c: per_matrix.loc[per_matrix["conc_mgml"] == c, "median_E_kPa"].dropna().to_numpy() for c in concs}
    lvl = cfg["stats"]["ci_level"]
    cond = []
    for c in concs:
        v = groups[c]; sem = v.std(ddof=1) / math.sqrt(len(v))
        tc = stats.t.ppf(0.5 + lvl / 2, len(v) - 1)
        cond.append({"conc_mgml": c, "n_gels": len(v), "mean_E_kPa": round(v.mean(), 3),
                     "sd_E_kPa": round(v.std(ddof=1), 3),
                     "geomean_E_kPa": round(float(np.exp(np.log(v).mean())), 3),
                     "ci95_lo": round(v.mean() - tc * sem, 3), "ci95_hi": round(v.mean() + tc * sem, 3)})
    cond = pd.DataFrame(cond)
    ordered = [groups[c] for c in concs]
    if len(concs) >= 2:
        F, p = stats.f_oneway(*ordered)
        pk = stats.kruskal(*ordered).pvalue
        omni = pd.DataFrame([{"test": "one_way_ANOVA", "stat": round(float(F), 3),
                              "df1": len(concs) - 1, "df2": len(per_matrix) - len(concs),
                              "p": p, "p_exact_perm": _perm_anova(ordered),
                              "kruskal_p": pk}])
    else:
        omni = pd.DataFrame()
    rows, wps = [], []
    for lo, hi in combinations(concs, 2):
        a, b = groups[lo], groups[hi]
        t, wp = stats.ttest_ind(b, a, equal_var=False)
        rows.append({"comparison": f"{lo} vs {hi} mg/mL", "mean_diff_kPa": round(b.mean() - a.mean(), 3),
                     "fold": round(b.mean() / a.mean(), 3), "welch_t": round(float(t), 3),
                     "welch_p": wp, "p_exact_perm": round(_perm_two(a, b), 4),
                     "hedges_g": round(_hedges_g(a, b), 3)})
        wps.append(wp)
    order = np.argsort(wps); m = len(wps); holm = [np.nan] * m; run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * wps[i]); holm[i] = min(run, 1.0)
    for r, h in zip(rows, holm):
        r["welch_p_holm"] = round(h, 4); r["welch_p"] = round(r["welch_p"], 4)
    return cond, omni, pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# SENSITIVITY / STABILITY
# ----------------------------------------------------------------------------
def sensitivity_analysis(master, matrices_by_name, cfg, out_dir):
    """Show the per-condition result is stable to the contact rule, the fit
    window, and curve-processing choices. Curve set held fixed to the primary
    included curves; each setting only changes how E is estimated."""
    inc = master[master["included"]].copy()
    concs = sorted(inc["conc"].unique())

    def per_condition_geomean(evalues):
        """evalues: apparent modulus aligned to the rows of ``inc``. Returns
        {conc: geometric mean of the per-gel median E}."""
        s = inc.copy(); s["Ev"] = np.asarray(evalues, dtype=float)
        out = {}
        for c in concs:
            gel_meds = s[s["conc"] == c].groupby("gel")["Ev"].median().dropna()
            out[c] = float(np.exp(np.log(gel_meds).mean())) if len(gel_meds) else np.nan
        return out

    settings = []  # (axis, label, {conc: geomean})
    base = per_condition_geomean(inc["E_app_kPa"])
    settings.append(("primary", "primary (R2-max, 4um)", base))

    # fit-window cap
    for cap in cfg["sensitivity"]["caps_nm"]:
        col = f"E_cap{cap}"
        if col in inc:
            settings.append(("cap", f"cap {cap/1000:.1f}um",
                             per_condition_geomean(inc[col])))
    # contact-point shift
    for sh in cfg["sensitivity"]["z0_shifts_nm"]:
        col = f"E_shift{sh}"
        if col in inc:
            settings.append(("z0_shift", f"z0 {sh:+d}nm",
                             per_condition_geomean(inc[col])))
    # contact-detection heuristic: vary the bracket sigma-threshold that seeds
    # the objective R2-max search, and re-run the full search (full re-analysis).
    for st in cfg["sensitivity"].get("sigma_thresholds", []):
        if st == cfg["contact"]["sigma_threshold"]:
            continue
        cc = {**cfg, "contact": {**cfg["contact"], "sigma_threshold": st}}
        settings.append(("contact_bracket", f"bracket {st:g}sigma",
                         _reanalyse_geomean(inc, matrices_by_name, cc, concs)))

    # curve-processing: smoothing window + baseline fraction (full re-analysis
    # of the included curves only, matched by file identity)
    for sw in cfg["sensitivity"]["smoothing_windows"]:
        if sw == cfg["contact"]["smoothing_window"]:
            continue
        cc = {**cfg, "contact": {**cfg["contact"], "smoothing_window": sw}}
        settings.append(("smoothing", f"smoothing {sw}", _reanalyse_geomean(inc, matrices_by_name, cc, concs)))
    for bf in cfg["sensitivity"]["baseline_fractions"]:
        if bf == cfg["contact"]["baseline_fraction"]:
            continue
        cc = {**cfg, "contact": {**cfg["contact"], "baseline_fraction": bf}}
        settings.append(("baseline", f"baseline frac {bf}", _reanalyse_geomean(inc, matrices_by_name, cc, concs)))

    # table
    rows = []
    for axis, label, gm in settings:
        row = {"axis": axis, "setting": label}
        for c in concs:
            row[f"E_{c}"] = round(gm.get(c, np.nan), 3)
            row[f"dev_{c}_pct"] = (round(100 * (gm.get(c, np.nan) - base[c]) / base[c], 2)
                                   if np.isfinite(gm.get(c, np.nan)) else np.nan)
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / "sensitivity_analysis.csv", index=False)

    axes_order = ["cap", "z0_shift", "contact_bracket", "smoothing", "baseline"]
    xlabels = {"cap": r"Fit-window cap [$\mu$m]", "z0_shift": "Contact-point shift [nm]",
               "contact_bracket": r"Contact bracket [$\sigma$]", "smoothing": "Smoothing window [points]",
               "baseline": "Baseline fraction"}

    def numlabel(s):
        found = re.findall(r"[-+]?\d*\.?\d+", s)
        return found[-1] if found else s

    present = [a for a in axes_order if a in tab["axis"].values]
    fig, axs = plt.subplots(1, len(present) + 1, figsize=(4.2 * (len(present) + 1), 4.6))
    from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
    yt = sorted(base.values())
    for k, axis in enumerate(present):
        ax = axs[k]; sub = tab[tab["axis"] == axis]
        for c in concs:
            ax.plot(range(len(sub)), sub[f"E_{c}"], "-o", color=CFG["palette"].get(c, "#555"),
                    label=f"{c}", ms=4)
            ax.axhline(base[c], color=CFG["palette"].get(c, "#555"), lw=0.7, ls=":")
        ax.set_xticks(range(len(sub))); ax.set_xticklabels([numlabel(s) for s in sub["setting"]], fontsize=8)
        ax.set_xlabel(xlabels[axis]); ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator(yt))
        ax.yaxis.set_major_formatter(FixedFormatter([f"{v:.1f}" for v in yt]))
        ax.set_ylim(min(yt) * 0.75, max(yt) * 1.3)
        ax.spines[["top", "right"]].set_visible(False)
        if k == 0:
            ax.set_ylabel(r"Geometric-mean apparent modulus [kPa]")
            ax.legend(frameon=False, fontsize=8, title=r"[mg mL$^{-1}$]")
    axd = axs[-1]
    dev_cols = [f"dev_{c}_pct" for c in concs]
    maxdev = tab[tab["axis"] != "primary"][dev_cols].abs().max().max()
    for c in concs:
        d = tab[tab["axis"] != "primary"][f"dev_{c}_pct"].abs()
        axd.scatter([f"{c}"] * len(d), d, color=CFG["palette"].get(c, "#555"), alpha=0.6, s=25)
    axd.axhline(5, color="0.5", ls="--", lw=0.8)
    axd.set_ylabel(r"|Deviation from primary| [%]"); axd.set_xlabel(r"Concentration [mg mL$^{-1}$]")
    axd.text(0.5, 0.96, f"All within {maxdev:.1f} %", transform=axd.transAxes, ha="center", fontsize=9)
    axd.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.16, top=0.95, wspace=0.32)
    save_fig(fig, out_dir / "sensitivity_analysis")
    return tab, float(maxdev)


def _reanalyse_geomean(inc, matrices_by_name, cc, concs):
    """Re-fit only the included curves under a modified config and return the
    per-condition geometric mean of the per-gel median E. Curves are matched to
    ``inc`` by row index, so identical grid file names in different gels never clash."""
    newE = pd.Series(np.nan, index=inc.index, dtype=float)
    for name, sub in inc.groupby("source_name"):
        mpath = matrices_by_name[name]
        for idx, r in sub.iterrows():
            f = mpath / r["file"]
            try:
                _, data = parse_curve(f)
                res = analyse_curve(data, cc, sensitivity=False)
                newE.loc[idx] = res.get("E_app_kPa", np.nan)
            except Exception:
                newE.loc[idx] = np.nan
    s = inc.copy(); s["Ev"] = newE
    out = {}
    for c in concs:
        gm = s[s["conc"] == c].groupby("gel")["Ev"].median().dropna()
        out[c] = float(np.exp(np.log(gm).mean())) if len(gm) else np.nan
    return out


# ----------------------------------------------------------------------------
# DRIVER
# ----------------------------------------------------------------------------
def choose_folder_gui():
    """Open a folder picker for the session directory; return None if no GUI or no choice."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.update()
        d = filedialog.askdirectory(title="Select the session folder (all gels of one characterisation session)")
        root.destroy()
        return Path(d) if d else None
    except Exception as exc:
        print(f"[GUI unavailable: {exc}] Pass --raw <folder> instead.", file=sys.stderr)
        return None


def run(raw_root: Path, out_root: Path):
    """Analyse a whole session: every gel, then session-level CSVs, statistics,
    figures, the sensitivity analysis and the session report."""
    matrices = discover_matrices(raw_root)
    if not matrices:
        sys.exit(f"No gel matrix folders found under {raw_root}. Expected sub-folders "
                 f"named GelMA_<conc>mgmL_gel<n> (or matrix_scan.._<pct>_<rep>).")
    out_root.mkdir(parents=True, exist_ok=True)
    matrices_by_name = {m["name"]: m["path"] for m in matrices}

    all_df, summaries = [], []
    for mat in matrices:
        print(f"[matrix] {mat['name']}", flush=True)
        df, summ = analyse_matrix(mat, CFG, out_root)
        df["conc"] = mat["conc"]; df["gel"] = mat["gel"]; df["source_name"] = mat["name"]
        all_df.append(df); summaries.append(summ)

    master = pd.concat(all_df, ignore_index=True)   # full frame (keeps sensitivity cols in memory)
    clean_cols = ["sample_id", "conc", "gel", "source_name", "file", "X", "Y",
                  "instr_status", "instr_k_Nm", "instr_R_um", "instr_Eeff_kPa",
                  "instr_Zsurface_um", "instr_Zposition_um", "instr_Pmax_uN",
                  "z0_zeta_nm", "z0_thr_nm", "E_app_kPa", "R2", "nrmse", "snr",
                  "depth_nm", "a_contact_um", "late_ratio", "z0_sensitivity_frac",
                  "qc_status", "qc_snr", "qc_depth", "qc_R2", "qc_nrmse",
                  "qc_late_ratio", "qc_E_range", "QC_pass", "is_outlier",
                  "included", "QC_fail_reason"]
    master.reindex(columns=[c for c in clean_cols if c in master.columns]).to_csv(
        out_root / "master_per_curve.csv", index=False)
    per_matrix = pd.DataFrame(summaries).sort_values(["conc_mgml", "gel"])
    per_matrix.to_csv(out_root / "per_matrix_summary.csv", index=False)

    cond, omni, pair = compute_statistics(per_matrix, CFG)
    cond.to_csv(out_root / "per_condition_summary.csv", index=False)
    pd.concat([omni.assign(kind="omnibus"), pair.assign(kind="pairwise")],
              ignore_index=True).to_csv(out_root / "statistics.csv", index=False)

    superplot(master, per_matrix, out_root / "superplot")
    _map_atlas(master, per_matrix, "E_app_kPa", "Apparent modulus [kPa]",
               out_root / "modulus_map_atlas", "viridis")
    _map_atlas(master[master["z0_zeta_nm"].notna()].assign(z0_um=lambda d: d["z0_zeta_nm"] / 1000),
               per_matrix, "z0_um", r"Contact point $z_0$ [$\mu$m]",
               out_root / "contactpoint_map_atlas", "cividis")

    print("[sensitivity] ...", flush=True)
    sens_tab, maxdev = sensitivity_analysis(master, matrices_by_name, CFG, out_root)

    with open(out_root / "session_report.txt", "w") as fh:
        fh.write("GelMA spherical-nanoindentation - session recap\n" + "=" * 52 + "\n")
        fh.write(f"gels: {len(matrices)}   curves: {len(master)}   "
                 f"included: {int(master['included'].sum())}   "
                 f"QC-fail: {int((~master['QC_pass']).sum())}   "
                 f"outliers: {int(master['is_outlier'].sum())}\n\n")
        fh.write("Per-condition (mean +/- SD of gel medians):\n")
        for _, r in cond.iterrows():
            fh.write(f"  {int(r['conc_mgml']):>4} mg/mL : {r['mean_E_kPa']:.2f} +/- {r['sd_E_kPa']:.2f} kPa "
                     f"(geomean {r['geomean_E_kPa']:.2f}; n={int(r['n_gels'])})\n")
        if not omni.empty:
            o = omni.iloc[0]
            fh.write(f"\nOmnibus: {o['test']} stat={o['stat']} p={o['p']:.2e} "
                     f"(exact-perm {o['p_exact_perm']})\n")
        for _, r in pair.iterrows():
            fh.write(f"  {r['comparison']}: Holm p={r['welch_p_holm']}, "
                     f"exact-perm {r['p_exact_perm']}, Hedges g={r['hedges_g']}\n")
        fh.write(f"\nStability: every alternative contact rule / fit window / curve-processing "
                 f"choice keeps each per-condition modulus within {maxdev:.1f}% of the primary "
                 f"value (see sensitivity_analysis.*).\n")
        fh.write(f"\nMethod: apparent reduced modulus from zero-intercept Hertz sphere over 0-4 um, "
                 f"objective R^2-max contact; probe R={CFG['probe']['tip_radius_m']*1e6:.0f} um; "
                 f"gel=experimental unit. numpy {np.__version__}, pandas {pd.__version__}.\n")

    print("\n=== per-condition ==="); print(cond.to_string(index=False))
    if not omni.empty:
        print("=== omnibus ==="); print(omni.to_string(index=False))
    print("=== pairwise ==="); print(pair.to_string(index=False))
    print(f"=== stability: max deviation across all alternatives = {maxdev:.2f}% ===")
    print(f"Outputs under {out_root}/")
    return per_matrix, maxdev


def main(argv=None):
    """Parse CLI arguments (or open the GUI chooser) and run the session analysis."""
    ap = argparse.ArgumentParser(description="GelMA nanoindentation publication suite.")
    ap.add_argument("--raw", type=Path, help="session folder (omit to open a GUI chooser)")
    ap.add_argument("--out", type=Path, help="output folder (default: <session>_analysis)")
    args = ap.parse_args(argv)
    raw = args.raw or choose_folder_gui()
    if raw is None:
        sys.exit("No folder selected.")
    out = args.out or raw.parent / (raw.name + "_analysis")
    run(Path(raw), Path(out))


if __name__ == "__main__":
    main()
