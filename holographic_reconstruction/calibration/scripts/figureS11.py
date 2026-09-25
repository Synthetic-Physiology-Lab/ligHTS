# -*- coding: utf-8 -*-
"""Figure S11 - nanoindentation-anchored cross-calibration of the three modalities.

Six panels, all drawn from data/calibration_source.npz (written by step 1) and
outputs/calibration_constants.json (written by step 3):

  a  physical relief of the calibration specimen, nanoindentation Z-surface map
  b  apparent optical relief, confocal brightest-slice surface (field 1 of 3)
  c  optical path difference, HPI phase map (field 1 of 5), OPD = phi * lambda
  d  the three groove profiles on one axis: physical relief and apparent optical
     relief on the left, OPD on the right.  The two axes are tied by the locked
     constant, OPD[nm] = relief[um] * lambda[nm] / C, so the overlay of the black
     and orange curves *is* the calibration of C; the blue curve stays shallow by
     exactly the axial factor zeta, which panel f then removes.
  e  zeta against the geometrical-optics band of the dry NA 0.75 objective
  f  the same profiles after conversion, confocal x zeta and HPI x C

Every map is displayed with the detrending of the estimator itself
(groovekit.measure_field): per-row offset removal followed by a quadratic in the
groove-normal coordinate.  The nanoindentation map is shown with the per-line
offsets that measure_scatter carries as nuisance regressors already removed.
The profiles are cropped to the common window of panels a-c and shifted by less
than one pitch so that a crest sits at x = 0; no other alignment is applied, and
the amplitudes are untouched.

Run after step1 and step3:
    python scripts/figureS11.py
"""
import os
import json
import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "outputs")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--stem", default="FigureS11_calibration")
ap.add_argument("--width-mm", type=float, default=280.0)
ap.add_argument("--height-mm", type=float, default=150.0)
ap.add_argument("--window-um", type=float, default=None,
                help="side of the common field shown in a-c and of the profile "
                     "window in d and f; default: the nanoindentation extent")
ap.add_argument("--clim-a", type=float, default=None, help="+/- um, panel a")
ap.add_argument("--clim-b", type=float, default=None, help="+/- um, panel b")
ap.add_argument("--clim-c", type=float, default=None, help="+/- nm, panel c")
ap.add_argument("--formats", default="png,pdf,eps")
args = ap.parse_args()

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 12, "font.weight": "bold",
    "axes.labelsize": 12, "axes.labelweight": "bold",
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.5, "ytick.major.width": 1.5,
    "xtick.direction": "out", "ytick.direction": "out",
})

NANO_C = "k"
CONF_C = "#4c92c3"
HPI_C = "#e8964a"
CMAP = "RdBu_r"

src_path = os.path.join(DATA, "calibration_source.npz")
const_path = os.path.join(OUT, "calibration_constants.json")
if not os.path.exists(src_path):
    raise SystemExit("missing %s - run scripts/step1_measure_calibration.py first" % src_path)
if not os.path.exists(const_path):
    raise SystemExit("missing %s - run scripts/step3_lock_constants.py first" % const_path)

src = np.load(src_path)
const = json.load(open(const_path))

ZETA = float(const["constants"]["zeta"])
U_ZETA = float(const["constants"]["u_zeta"])
C_UM = float(const["constants"]["C_um_per_phase_unit"])
BOUNDS = const["optics"]["axial_bounds"]
LAM_NM = 1000.0 * float(src["lam_um"])
CONF_PX = float(src["conf_px"])
HPI_PX = float(src["hpi_px"])


# ----------------------------------------------------------------- helpers
def order_axis(values):
    """Sort a stage coordinate ascending and zero it; returns (order, coord)."""
    v = np.asarray(values, float)
    o = np.argsort(v)
    return o, v[o] - v[o].min()


def detrend_map(M, px):
    """The display detrending of groovekit.measure_field.

    Per-row offset removal, then the quadratic in the groove-normal coordinate
    that the estimator fits to the collapsed profile, then mean centring.  The
    grooves are vertical in these maps, so the groove-normal coordinate is the
    column index.
    """
    A = np.asarray(M, float) - np.nanmedian(M, axis=1, keepdims=True)
    x = np.arange(A.shape[1]) * px
    prof = np.nanmedian(A, axis=0)
    ok = np.isfinite(prof)
    A = A - np.polyval(np.polyfit(x[ok], prof[ok], 2), x)[None, :]
    return A - np.nanmean(A)


def center_crop(M, px, side_um):
    """Square crop of `side_um` about the centre of the field."""
    n = int(round(side_um / px))
    out = M
    for axis in (0, 1):
        k = min(n, out.shape[axis])
        i0 = (out.shape[axis] - k) // 2
        out = out[i0:i0 + k, :] if axis == 0 else out[:, i0:i0 + k]
    return out


def sym_clim(curve, margin=1.15):
    """Symmetric colour limit from the fitted groove amplitude of the modality.

    A percentile of the map itself is set by the salt-and-pepper outliers of the
    confocal surface rather than by the relief, so the colour scale is tied to
    the quantity the panel is about: the peak-to-trough of the fitted waveform.
    """
    return float(margin * np.ptp(curve) / 2.0)


def show_map(ax, A, px, clim, title, cbar_label=None):
    extent = (0.0, A.shape[1] * px, 0.0, A.shape[0] * px)
    im = ax.imshow(A, origin="lower", extent=extent, cmap=CMAP,
                   vmin=-clim, vmax=clim, interpolation="nearest", aspect="equal")
    ax.set_title(title, pad=8)
    ax.set_xlabel("Across Grooves [µm]")
    ax.set_ylabel("Along Grooves [µm]")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_linewidth(1.5)
    cb.ax.tick_params(width=1.5, labelsize=10)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=11, fontweight="bold")
    return im


def tile_wave(xq, t, w, pitch, t0):
    """The fitted waveform repeated over `xq`, with its crest at xq = 0."""
    tt = np.concatenate([np.asarray(t, float), [pitch]])
    ww = np.concatenate([np.asarray(w, float), [w[0]]])
    return np.interp(np.mod(xq + t0, pitch), tt, ww)


def prepare_profile(x, prof, t, w, pitch, window):
    """Crop one profile to `window`, then shift it so that a crest sits at x = 0.

    Returns (x_plot, profile_centred, crest_offset_in_the_waveform).  A little
    more than `window` is kept so that the shift cannot shorten the curve.
    """
    x = np.asarray(x, float)
    prof = np.asarray(prof, float)
    w = np.asarray(w, float)
    t = np.asarray(t, float)
    take = min(x.size, int(round((window + pitch) / (x[1] - x[0]))))
    i0 = max(0, (x.size - take) // 2)
    xs = x[i0:i0 + take]
    ps = prof[i0:i0 + take]
    t0 = float(t[int(np.argmax(w))])                 # crest within one pitch
    delta = float(np.mod(t0 - xs[0], pitch))
    return xs - xs[0] - delta, ps - np.nanmean(ps), t0


def panel_letter(fig, ax, letter, dx=0.045, dy=0.055):
    """Letter in figure coordinates, so it clears the y-label and the title."""
    p = ax.get_position()
    fig.text(p.x0 - dx, min(p.y1 + dy, 0.995), letter, fontsize=19,
             fontweight="bold", va="top", ha="left")


# ------------------------------------------------------------------- data
nano_Z = np.asarray(src["nano_Z"], float)            # [iy, ix] = [across, along]
oy, y_across = order_axis(src["nano_Y"])             # grooves run along stage X
ox, x_along = order_axis(src["nano_X"])
nano_map = nano_Z[np.ix_(oy, ox)]
# the per-line offsets that measure_scatter carries as nuisance regressors, i.e.
# one offset per along-groove coordinate (one column of the stage raster)
nano_map = nano_map - np.nanmean(nano_map, axis=0, keepdims=True)
nano_map = nano_map - np.nanmean(nano_map)
nano_map = nano_map.T                                # rows = along, cols = across
nano_px = float(np.median(np.diff(y_across)))

WINDOW = args.window_um if args.window_um else float(y_across.max() + nano_px)

conf_map = center_crop(detrend_map(src["conf_map"], CONF_PX), CONF_PX, WINDOW)
hpi_map = center_crop(detrend_map(src["hpi_map"], HPI_PX), HPI_PX, WINDOW) * LAM_NM

nano_pitch = float(src["nano_pitch"])
conf_pitch = float(src["conf_pitch"][0])
hpi_pitch = float(src["hpi_pitch"][0])

nano_wave = np.asarray(src["nano_wave"], float)
nano_wave = nano_wave - nano_wave.mean()
nano_t = np.asarray(src["nano_wave_t"], float)
nano_t0 = float(nano_t[int(np.argmax(nano_wave))])

conf_x, conf_prof, conf_t0 = prepare_profile(
    src["conf_x"], src["conf_profile"], src["conf_wave_t"], src["conf_wave"],
    conf_pitch, WINDOW)
conf_wave = np.asarray(src["conf_wave"], float)
conf_wave = conf_wave - conf_wave.mean()

hpi_x, hpi_prof, hpi_t0 = prepare_profile(
    src["hpi_x"], src["hpi_profile"], src["hpi_wave_t"], src["hpi_wave"],
    hpi_pitch, WINDOW)
hpi_wave = np.asarray(src["hpi_wave"], float)
hpi_wave = hpi_wave - hpi_wave.mean()

xq = np.linspace(0.0, WINDOW, 2000)
nano_curve = tile_wave(xq, nano_t, nano_wave, nano_pitch, nano_t0)
conf_curve = tile_wave(xq, src["conf_wave_t"], conf_wave, conf_pitch, conf_t0)
hpi_curve = tile_wave(xq, src["hpi_wave_t"], hpi_wave, hpi_pitch, hpi_t0)


# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(args.width_mm / 25.4, args.height_mm / 25.4))
gs = fig.add_gridspec(2, 3, hspace=0.60, wspace=0.62,
                      left=0.075, right=0.965, top=0.88, bottom=0.10)
axa = fig.add_subplot(gs[0, 0])
axb = fig.add_subplot(gs[0, 1])
axc = fig.add_subplot(gs[0, 2])
axd = fig.add_subplot(gs[1, 0])
axe = fig.add_subplot(gs[1, 1])
axf = fig.add_subplot(gs[1, 2])

# -- a, b, c : the three maps
show_map(axa, nano_map, nano_px,
         args.clim_a or sym_clim(nano_curve), "Physical Relief [µm]")
show_map(axb, conf_map, CONF_PX,
         args.clim_b or sym_clim(conf_curve), "Apparent Optical Relief [µm]")
show_map(axc, hpi_map, HPI_PX,
         args.clim_c or sym_clim(hpi_curve * LAM_NM), "Optical Path Difference [nm]")

# -- d : the three profiles, the two axes tied by C
axd.plot(conf_x, conf_prof, color=CONF_C, lw=1.0, alpha=0.35, zorder=1)
axd.plot(xq, conf_curve, color=CONF_C, lw=2.0, zorder=3, label="Confocal")
axd.plot(xq, nano_curve, color=NANO_C, lw=2.0, zorder=4, label="Nanoindentation")

axd2 = axd.twinx()
axd2.plot(hpi_x, hpi_prof * LAM_NM, color=HPI_C, lw=1.0, alpha=0.35, zorder=1)
axd2.plot(xq, hpi_curve * LAM_NM, color=HPI_C, lw=2.0, zorder=3, label="HPI")

lo = min(np.nanmin(nano_curve), np.nanmin(conf_prof), np.nanmin(conf_curve))
hi = max(np.nanmax(nano_curve), np.nanmax(conf_prof), np.nanmax(conf_curve))
pad = 0.10 * (hi - lo)
axd.set_ylim(lo - pad, hi + pad)
# OPD[nm] = relief[um] * lambda[nm] / C : the locked constant sets the twin axis,
# so nanoindentation and HPI coincide only if C is right
axd2.set_ylim((lo - pad) * LAM_NM / C_UM, (hi + pad) * LAM_NM / C_UM)
axd2.spines["top"].set_visible(False)
axd2.tick_params(width=1.5, labelsize=11)
axd.set_xlim(0, WINDOW)
axd.set_xlabel("Across Grooves [µm]")
axd.set_ylabel("Relief [µm]")
axd2.set_ylabel("OPD [nm]")
axd.spines["top"].set_visible(False)
handles = [mpl.lines.Line2D([], [], color=NANO_C, lw=2.0, label="Nanoindentation"),
           mpl.lines.Line2D([], [], color=CONF_C, lw=2.0, label="Confocal"),
           mpl.lines.Line2D([], [], color=HPI_C, lw=2.0, label="HPI")]
axd.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
           ncol=3, frameon=False, fontsize=9, handlelength=1.2,
           columnspacing=0.8, handletextpad=0.4)

# -- e : zeta against the geometrical-optics band
par = float(BOUNDS["paraxial"])
lya = float(BOUNDS["lyakin_stallinga"])
mar = float(BOUNDS["marginal_ray"])
axe.axhspan(par, mar, facecolor="0.90", edgecolor="none", zorder=0)
axe.axhline(mar, color="#e07bb0", ls="--", lw=2.0, zorder=2)
axe.axhline(lya, color="#2e8b57", ls="-.", lw=2.0, zorder=2)
axe.axhline(par, color=CONF_C, ls="--", lw=2.0, zorder=2)
axe.errorbar([0.66], [ZETA], yerr=[U_ZETA], fmt="o", color=HPI_C, ms=11,
             elinewidth=2.4, capsize=7, capthick=2.4, zorder=4)
span = max(mar - par, 4 * U_ZETA)
axe.set_ylim(min(par, ZETA - U_ZETA) - 0.10 * span,
             max(mar, ZETA + U_ZETA) + 0.10 * span)
axe.set_xlim(0, 1)
axe.set_xticks([])
axe.set_ylabel("Axial Factor ζ [a.u.]")
axe.text(0.04, mar, "Marginal ray", color="#e07bb0", fontsize=11,
         fontweight="bold", va="bottom", ha="left")
axe.text(0.04, lya, "Lyakin / Stallinga", color="#2e8b57", fontsize=11,
         fontweight="bold", va="bottom", ha="left")
axe.text(0.04, par, "Paraxial", color=CONF_C, fontsize=11,
         fontweight="bold", va="bottom", ha="left")
axe.text(0.05, 0.5 * (lya + mar), "ζ = %.2f ± %.2f" % (ZETA, U_ZETA),
         color=HPI_C, fontsize=12, fontweight="bold", va="center", ha="left")
axe.spines["top"].set_visible(False)
axe.spines["right"].set_visible(False)

# -- f : the same profiles converted to physical relief
axf.plot(conf_x, conf_prof * ZETA, color=CONF_C, lw=1.0, alpha=0.35, zorder=1)
axf.plot(hpi_x, hpi_prof * C_UM, color=HPI_C, lw=1.0, alpha=0.35, zorder=1)
axf.plot(xq, conf_curve * ZETA, color=CONF_C, lw=2.0, zorder=3)
axf.plot(xq, hpi_curve * C_UM, color=HPI_C, lw=2.0, zorder=3)
axf.plot(xq, nano_curve, color=NANO_C, lw=2.0, zorder=4)
axf.set_xlim(0, WINDOW)
axf.set_xlabel("Across Grooves [µm]")
axf.set_ylabel("Physical Relief [µm]")
axf.spines["top"].set_visible(False)
axf.spines["right"].set_visible(False)
handles = [mpl.lines.Line2D([], [], color=NANO_C, lw=2.0, label="Nanoindentation"),
           mpl.lines.Line2D([], [], color=CONF_C, lw=2.0, label="Confocal × ζ"),
           mpl.lines.Line2D([], [], color=HPI_C, lw=2.0, label="HPI × C")]
axf.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
           ncol=3, frameon=False, fontsize=9, handlelength=1.2,
           columnspacing=0.8, handletextpad=0.4)

# the letters of the lower row clear a legend, those of the upper row a title
for ax, letter, dy in ((axa, "a", 0.055), (axb, "b", 0.055), (axc, "c", 0.055),
                       (axd, "d", 0.095), (axe, "e", 0.095), (axf, "f", 0.095)):
    panel_letter(fig, ax, letter, dy=dy)

for ext in [e.strip() for e in args.formats.split(",") if e.strip()]:
    fig.savefig(os.path.join(FIG, args.stem + "." + ext), dpi=600)

print("window %.1f um | zeta = %.3f +/- %.3f | C = %.3f um per phase unit"
      % (WINDOW, ZETA, U_ZETA, C_UM))
print("peak-to-trough of the plotted waveforms:")
print("  nanoindentation  %7.3f um" % np.ptp(nano_curve))
print("  confocal         %7.3f um apparent -> %7.3f um x zeta"
      % (np.ptp(conf_curve), ZETA * np.ptp(conf_curve)))
print("  HPI              %7.4f phase     -> %7.3f um x C"
      % (np.ptp(hpi_curve), C_UM * np.ptp(hpi_curve)))
print("wrote figures/%s.{%s}" % (args.stem, args.formats))
