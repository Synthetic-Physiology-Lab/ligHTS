# -*- coding: utf-8 -*-
"""Figure 6f - physical groove depth by pitch design.

Both modalities are plotted in the same physical unit, converted with the locked
constants in outputs/calibration_constants.json:

    HPI       d = C * phi          (per field, swarm + mean +/- SD)
    confocal  d = zeta * a         (group mean, bordeaux line + SD band)

The grey box is the expanded uncertainty U = k*u_c, k = 2, of the HPI-minus-
confocal difference, propagated per JCGM 100:2008.  It is a tolerance, not a
descriptive spread: a group agrees when its two means differ by less than U.
"""
import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "outputs")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--stem", default="Figure6f_physical_groove_depth")
ap.add_argument("--width-mm", type=float, default=80.0)
ap.add_argument("--height-mm", type=float, default=100.0)
args = ap.parse_args()

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 12, "font.weight": "bold",
    "axes.labelsize": 12, "axes.labelweight": "bold",
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.5, "ytick.major.width": 1.5,
    "xtick.direction": "out", "ytick.direction": "out",
})

const = json.load(open(os.path.join(OUT, "calibration_constants.json")))
f6 = pd.read_csv(os.path.join(OUT, "figure6f_source_data.csv"))
G = pd.read_csv(os.path.join(OUT, "group_summary.csv"))

PITCHES = [40, 60, 80]
SWARM = {40: "#4c92c3", 60: "#5aa469", 80: "#e8964a"}
BORDEAUX = "#7b1a2b"

fig, ax = plt.subplots(figsize=(args.width_mm / 25.4, args.height_mm / 25.4))
rng = np.random.default_rng(20260820)

for i, p in enumerate(PITCHES):
    g = G[G.pitch_design_um == p].iloc[0]
    d = f6[f6.pitch_design_um == p].physical_depth_um.values

    # grey tolerance box, centred on the confocal group mean
    half = g.U_k2_um
    ax.add_patch(mpl.patches.Rectangle(
        (i - 0.34, g.confocal_mean_um - half), 0.68, 2 * half,
        facecolor="0.90", edgecolor="0.25", linewidth=1.5, zorder=1))

    # bordeaux confocal group mean
    ax.plot([i - 0.34, i + 0.34], [g.confocal_mean_um] * 2,
            color=BORDEAUX, lw=2.8, zorder=3, solid_capstyle="butt")

    # HPI swarm
    x = i + np.clip(rng.normal(0, 0.075, d.size), -0.22, 0.22)
    ax.scatter(x, d, s=7, color=SWARM[p], alpha=0.75, linewidths=0, zorder=2)

    # HPI mean +/- SD
    ax.plot([i - 0.24, i + 0.24], [d.mean()] * 2, color="k", lw=3.0, zorder=5,
            solid_capstyle="butt")
    ax.errorbar([i], [d.mean()], yerr=[d.std(ddof=1)], fmt="none", ecolor="k",
                elinewidth=2.4, capsize=6, capthick=2.4, zorder=5)

ax.set_xticks(range(len(PITCHES)))
ax.set_xticklabels([str(p) for p in PITCHES])
ax.set_xlim(-0.6, len(PITCHES) - 0.4)
ax.set_ylim(0, max(35, np.ceil(f6.physical_depth_um.max() / 5) * 5 + 5))
ax.set_xlabel("Pitch Design [µm]")
ax.set_ylabel("Physical Groove Depth [µm]")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()

for ext in ("png", "pdf", "eps"):
    fig.savefig(os.path.join(FIG, args.stem + "." + ext), dpi=600)

# a separate legend key, so the panel itself stays clean in the plate
lf, la = plt.subplots(figsize=(args.width_mm / 25.4, 0.9))
la.axis("off")
handles = [
    mpl.lines.Line2D([], [], color="k", lw=3.0, label="HPI mean ± SD"),
    mpl.lines.Line2D([], [], marker="o", ls="none", color="0.45", ms=4,
                     label="HPI, one field of view"),
    mpl.lines.Line2D([], [], color=BORDEAUX, lw=2.8, label="Confocal group mean × ζ"),
    mpl.patches.Patch(facecolor="0.90", edgecolor="0.25", label="U (k = 2) of the difference"),
]
la.legend(handles=handles, loc="center", ncol=2, frameon=False, fontsize=9)
for ext in ("png", "pdf", "eps"):
    lf.savefig(os.path.join(FIG, args.stem + "_legend." + ext), dpi=600)

print("zeta = %.3f, C = %.3f um per native phase unit"
      % (const["constants"]["zeta"], const["constants"]["C_um_per_phase_unit"]))
print(G[["pitch_design_um", "hpi_n", "hpi_mean_um", "hpi_sd_um", "confocal_mean_um",
         "difference_um", "difference_pct", "U_k2_um", "agrees"]].round(3).to_string(index=False))
print("\nwrote %s.{png,pdf,eps} (+ _legend)" % os.path.join(FIG, args.stem))
