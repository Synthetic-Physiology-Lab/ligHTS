# -*- coding: utf-8 -*-
"""Step 1 - measure the calibration specimen in all three modalities.

Specimen: one 40 um pitch, 75 mg/mL GelMA hydrogel.
  nanoindentation  1 map,  40 x 40 points, 7 um step   (Optics11 Chiaro)
  confocal         3 z-stacks at three locations       (dz 0.9 um, dry NA 0.75)
  HPI              5 fields of view, 7 frames each     (PHI HoloMonitor M4)

The confocal z-stacks and the HPI fields are discovered, not hard-coded: every
'*.nd2' in e01_confocal is one location, and the frames in e02_holographic are
grouped into fields of view by their name.

Every field is rotated so the grooves are vertical before anything is measured,
and every depth uses the single estimator in groovekit.measure_*.

The nanoindentation curves in --nano-dir are analysed: the 40 x 40 map X-11..X-50
(1600 curves) obtained from the 50 x 40 Chiaro raster by discarding the first ten
columns (X-01..X-10, stage-settling offset). Grid nodes absent from --nano-dir, if
any, are dropped before fitting.

Inputs (defaults relative to this folder, override on the command line):
  --raw       default raw/c06_multimodal-calibration
              <RAW>/e02_holographic/w001_field<nn>_f<nn>.tiff   HPI frames
              <RAW>/e01_confocal/zstack<nn>.nd2                 confocal stacks
  --nano-dir  default raw/Nanoindentation_Calibration/matrix_scan18
              folder with the Chiaro '*I-01.txt' curves to analyse

HPI frames of one field of view share everything up to the '_f<nn>' suffix, which
is the frame ordinal within that field; fields are measured in field-number order
and frames in ordinal order.

Writes data/calibration_fields.csv and data/calibration_source.npz
"""
import os, re, glob, sys, argparse
import numpy as np
import pandas as pd
from PIL import Image
import nd2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groovekit as gk

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ap = argparse.ArgumentParser()
_ap.add_argument("--raw", default=os.path.join(HERE, "raw", "c06_multimodal-calibration"))
_ap.add_argument("--nano-dir", default=os.path.join(HERE, "raw", "Nanoindentation_Calibration", "matrix_scan18"))
_args = _ap.parse_args()
RAW = _args.raw
NANO_DIR = _args.nano_dir
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

LAM_UM = 0.635                      # HoloMonitor M4 source laser
CONF_PX = 1.343985993926428         # um/px, from the ND2 metadata
CONF_DZ = 0.900                     # um/slice, nominal z
HPI_PX = 25400.0 / 45872.308        # um/px, from the TIFF XResolution tag
PITCH_NOMINAL = 40.0

CONFOCAL_DIR = "e01_confocal"       # subfolders of --raw
HOLO_DIR = "e02_holographic"
# '<anything>field<nn>_f<nn>.tif[f]': the part before '_f<nn>' identifies the field
# of view, so frames of different wells or fields never share a group.
FRAME_RE = re.compile(r"(?P<fov>.*field(?P<field>\d+))_f(?P<frame>\d+)\.tiff?$", re.I)

rows = []


# ----------------------------------------------------------- nanoindentation
def load_nano():
    rec = []
    for f in glob.glob(os.path.join(NANO_DIR, "*I-01.txt")):
        m = re.search(r"X-(\d+) Y-(\d+)", os.path.basename(f))
        if not m:
            continue
        with open(f, errors="replace") as fh:
            head = "".join([l for i, l in enumerate(fh) if i <= 25])
        g = lambda p: (float(re.search(p, head).group(1)) if re.search(p, head) else np.nan)
        rec.append(dict(ix=int(m.group(1)), iy=int(m.group(2)),
                        x=g(r"X-position \(um\)\s+([-\d.]+)"),
                        y=g(r"Y-position \(um\)\s+([-\d.]+)"),
                        z=g(r"Z surface \(um\)\s+([-\d.]+)")))
    d = pd.DataFrame(rec).dropna(subset=["z"])
    # groove normal is stage Y (the grooves run along stage X)
    return d


print("[1/3] nanoindentation")
nano = load_nano()
Zg = nano.pivot(index="iy", columns="ix", values="z").values
Yg = nano.groupby("iy").y.mean().values
Xg = nano.groupby("ix").x.mean().values
ny, nx = Zg.shape
YY = np.repeat(Yg[:, None], nx, 1).ravel()
XX = np.repeat(Xg[None, :], ny, 0).ravel()
ZZ = Zg.ravel()
sel = np.isfinite(ZZ)                      # grid nodes outside the selection are NaN
YY, XX, ZZ = YY[sel], XX[sel], ZZ[sel]
N = gk.measure_scatter(YY, XX, ZZ, PITCH_NOMINAL)
sd_nano, ci_nano = gk.bootstrap_depth(N["u"], ZZ, N["pitch_um"],
                                      groups=XX, n_boot=400, seed=20260820)
print("    n=%d points, pitch %.3f um, residual tilt %+.3f deg, contact noise sd %.2f um"
      % (ZZ.size, N["pitch_um"], N["tilt_deg"], N["noise_sd"]))
print("    physical depth  %.3f +/- %.3f um  (95%% CI %.2f-%.2f)"
      % (N["depth"], sd_nano, *ci_nano))
rows.append(dict(modality="nanoindentation", field="matrix_scan", n=ZZ.size,
                 pitch_um=N["pitch_um"], angle_deg=N["tilt_deg"],
                 value=N["depth"], value_fundamental=N["depth_fundamental"],
                 unit="um_physical", noise=N["noise_sd"]))

# ------------------------------------------------------------------ confocal
print("[2/3] confocal")
conf_dir = os.path.join(RAW, CONFOCAL_DIR)
conf_files = sorted(glob.glob(os.path.join(conf_dir, "*.nd2")))
if not conf_files:
    sys.exit("no '*.nd2' z-stacks in %s" % conf_dir)
print("    %d z-stack(s): %s"
      % (len(conf_files), ", ".join(os.path.basename(p) for p in conf_files)))
conf_maps, conf_res = [], []
for p in conf_files:
    tag = os.path.splitext(os.path.basename(p))[0]
    with nd2.ND2File(p) as g:
        a = g.asarray().astype(np.float32)
    a = np.clip(a - np.median(a[:5]), 0, None)
    h = a.argmax(0) * CONF_DZ                    # brightest-slice surface, nominal z
    med = np.median(h)
    mad = 1.4826 * np.median(np.abs(h - med)) + 1e-9
    h = np.where(np.abs(h - med) < 3 * mad, h, np.nan)
    R = gk.measure_field(h, CONF_PX, PITCH_NOMINAL, do_orient=True)
    conf_maps.append(R["image"])
    conf_res.append(R)
    print("    %s  angle %+.2f deg  pitch %.3f um  apparent depth %.3f um"
          % (tag, R["angle_deg"], R["pitch_um"], R["depth"]))
    rows.append(dict(modality="confocal", field=tag, n=1,
                     pitch_um=R["pitch_um"], angle_deg=R["angle_deg"],
                     value=R["depth"], value_fundamental=R["depth_fundamental"],
                     unit="um_apparent_optical", noise=R["residual_rms"]))
a_conf = np.array([r["depth"] for r in conf_res])
_sd = a_conf.std(ddof=1) if a_conf.size > 1 else np.nan
print("    mean apparent optical depth  %.3f +/- %.3f um (SD, n=%d), SEM %.3f"
      % (a_conf.mean(), _sd, a_conf.size, _sd / np.sqrt(a_conf.size)))


# ----------------------------------------------------------------------- HPI
def read_phase(path):
    """HoloMonitor 16-bit phase TIFF -> native stored phase units."""
    with Image.open(path) as im:
        s = None
        for tid in (270, 40092, 40094, 40095):
            if tid in im.tag_v2:
                v = im.tag_v2.get(tid)
                if isinstance(v, (bytes, bytearray)):
                    v = v.decode("utf-16le", errors="ignore")
                v = str(v).replace(chr(0), "").strip()
                if "Min" in v and "Max" in v:
                    s = v
                    break
        m = re.search(r"Min\s*\(\s*0\s*\)\s*=\s*([-\d.,]+)\s*Max\s*\(\s*65535\s*\)\s*=\s*([-\d.,]+)", s)
        lo = float(m.group(1).replace(",", "."))
        hi = float(m.group(2).replace(",", "."))
        u = np.asarray(im, dtype=np.float32)
    return lo + (u / 65535.0) * (hi - lo)


def group_hpi_frames(folder):
    """Group 'field<nn>_f<nn>.tiff' frames into fields of view.

    Frames of one field share the whole name up to the '_f<nn>' ordinal, so
    fields of different wells stay separate even when the field numbers repeat.
    Returns [(fov, [paths in frame order])], ordered by field number.
    """
    groups, ignored = {}, []
    for p in glob.glob(os.path.join(folder, "*.tif*")):
        m = FRAME_RE.match(os.path.basename(p))
        if not m:
            ignored.append(os.path.basename(p))
            continue
        key = (int(m.group("field")), m.group("fov"))
        groups.setdefault(key, []).append((int(m.group("frame")), p))
    if ignored:
        print("    ignored %d file(s) not named field<nn>_f<nn>: %s"
              % (len(ignored), ", ".join(sorted(ignored)[:4])))
    if not groups:
        sys.exit("no 'field<nn>_f<nn>.tiff' frames in %s" % folder)
    out = []
    for key in sorted(groups):
        frames = sorted(groups[key])
        ordinals = [n for n, _ in frames]
        if len(set(ordinals)) != len(ordinals):
            sys.exit("field %s: repeated frame number in %s" % (key[1], ordinals))
        out.append((key[1], [p for _, p in frames]))
    return out


print("[3/3] HPI")
hpi_maps, hpi_res, align_scans = [], [], []
holo_dir = os.path.join(RAW, HOLO_DIR)
hpi_fields = group_hpi_frames(holo_dir)
print("    %d field(s): %s"
      % (len(hpi_fields), ", ".join("%s[%d]" % (f, len(fs)) for f, fs in hpi_fields)))
if len(set(len(fs) for _, fs in hpi_fields)) != 1:
    print("    note: fields do not all have the same number of frames")
for fov, fs in hpi_fields:
    stack = np.stack([read_phase(f) for f in fs])
    med = np.median(stack, axis=0)               # per-pixel median over the frames
    frame_sd = float(np.median(np.std(stack, axis=0)))
    R = gk.measure_field(med, HPI_PX, PITCH_NOMINAL, do_orient=True)
    # sensitivity of the recovered contrast to a residual rotation, for the
    # alignment-validation panel of the supplementary figure
    from scipy.ndimage import rotate as _rot
    small, fac = gk._downsample(np.nan_to_num(med, nan=float(np.nanmedian(med))),
                                max(1, int(min(med.shape) // 280)))
    scan_a = np.arange(-4.0, 4.001, 0.25)
    base = R["angle_deg"]
    scan_c = []
    for da in scan_a:
        rr = _rot(small, -(base + da), reshape=False, order=1, mode="nearest")
        cc = int(0.15 * rr.shape[0])
        scan_c.append(gk._profile_contrast(rr[cc:-cc, cc:-cc], HPI_PX * fac, PITCH_NOMINAL))
    align_scans.append(np.array(scan_c) / max(scan_c))
    hpi_maps.append(R["image"])
    hpi_res.append(R)
    print("    %-18s %d frames  angle %+.2f deg  pitch %.3f um  phase contrast %.4f"
          % (fov, len(fs), R["angle_deg"], R["pitch_um"], R["depth"]))
    rows.append(dict(modality="HPI", field=fov, n=len(fs),
                     pitch_um=R["pitch_um"], angle_deg=R["angle_deg"],
                     value=R["depth"], value_fundamental=R["depth_fundamental"],
                     unit="phase_native", noise=frame_sd))
phi = np.array([r["depth"] for r in hpi_res])
_phi_sd = phi.std(ddof=1) if phi.size > 1 else np.nan
print("    mean phase contrast  %.4f +/- %.4f (SD, n=%d), SEM %.4f"
      % (phi.mean(), _phi_sd, phi.size, _phi_sd / np.sqrt(phi.size)))

pd.DataFrame(rows).to_csv(os.path.join(DATA, "calibration_fields.csv"), index=False)
np.savez_compressed(
    os.path.join(DATA, "calibration_source.npz"),
    nano_Z=Zg, nano_Y=Yg, nano_X=Xg, nano_u=N["u"],
    nano_wave_t=N["wave_t"], nano_wave=N["wave"],
    nano_pitch=N["pitch_um"], nano_depth=N["depth"], nano_depth_sd=sd_nano,
    nano_tilt=N["tilt_deg"], nano_noise=N["noise_sd"],
    conf_map=conf_maps[0].astype(np.float32),
    conf_x=conf_res[0]["x"], conf_profile=conf_res[0]["profile"],
    conf_wave_t=conf_res[0]["wave_t"], conf_wave=conf_res[0]["wave"],
    conf_depth=a_conf, conf_pitch=np.array([r["pitch_um"] for r in conf_res]),
    conf_angle=np.array([r["angle_deg"] for r in conf_res]),
    hpi_map=hpi_maps[0].astype(np.float32),
    hpi_x=hpi_res[0]["x"], hpi_profile=hpi_res[0]["profile"],
    hpi_wave_t=hpi_res[0]["wave_t"], hpi_wave=hpi_res[0]["wave"],
    hpi_phase=phi, hpi_pitch=np.array([r["pitch_um"] for r in hpi_res]),
    hpi_angle=np.array([r["angle_deg"] for r in hpi_res]),
    hpi_fundamental=np.array([r["depth_fundamental"] for r in hpi_res]),
    conf_fundamental=np.array([r["depth_fundamental"] for r in conf_res]),
    nano_fundamental=N["depth_fundamental"],
    align_scan_angle=np.arange(-4.0, 4.001, 0.25),
    align_scan=np.vstack(align_scans),
    lam_um=LAM_UM, conf_px=CONF_PX, conf_dz=CONF_DZ, hpi_px=HPI_PX)
print("\nwrote data/calibration_fields.csv and data/calibration_source.npz")
