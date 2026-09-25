# -*- coding: utf-8 -*-
"""Step 2 - re-measure the historical HPI plate fields with the step-1 estimator.

The archived fields are the 8-bit median projections written by the original
`holo_to_tiff.py`: height in micrometres, linearly mapped from -25..+50 um onto
0..255, computed with h = phi * lambda / (2*pi*0.0064).  Undoing that mapping
recovers the native stored phase of each field, so the plates and the
calibration specimen can be put through exactly the same measurement chain.

Two things are deliberately NOT inherited from the archive:
  - the rotation, because the original angle came from a radial-power FFT that
    is several degrees biased; every field is re-oriented here
  - the depth estimator, because the original per-row ridge-to-trough extremum
    is noise-dependent

The encoding (wavelength, encoding constant, 8-bit window, pixel size) is read
from each file's ImageDescription; the archive values below are only fallbacks.
The encoding constant is not a calibration: it is undone exactly here, and the
physical depth is obtained later as C * phase (step3).

Input: --plates folder with the archived '*_MEDIAN.tif' fields.
Writes data/plate_fields.csv
"""
import os, re, glob, sys, time, math, argparse
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groovekit as gk

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ap = argparse.ArgumentParser()
_ap.add_argument("--plates", default=os.path.join(HERE, "raw", "1536_HOLO_processed"))
PLATES = _ap.parse_args().plates
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

LAM_UM = 0.635                                # fallback, if absent from metadata
ARCHIVE_DELTA_N = 0.0064                      # fallback encoding constant of the 8-bit files
WINDOW_LO, WINDOW_HI = -25.0, 50.0            # fallback 8-bit height window
DEFAULT_PX = 0.553
NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"      # numeric literal in the ImageDescription

# pre-declared QC, fixed before any depth was looked at
QC_PITCH_TOL = 0.10                           # |measured - nominal| / nominal
QC_MIN_PERIODS = 4.0
QC_MAX_RESIDUAL_FRAC = 0.60                   # residual RMS / depth


def read_field(path):
    with Image.open(path) as im:
        desc = str(im.tag_v2.get(270, ""))
        u8 = np.asarray(im, dtype=np.float32)
    m = re.search(r"Pixel size:\s*([\d.]+)", desc)
    px = float(m.group(1)) if m else DEFAULT_PX
    # Greek letters may be stored as '?' in the archived ImageDescription
    m_l = re.search(r"(?:λ|\?)=([\d.]+)\s*(?:µ|\?)m", desc)
    m_n = re.search(r"(?:Δ|\?)n=([\d.]+)", desc)
    m_w = re.search(r"Output mapping:\s*([-\d.]+)\.\.([-\d.]+)", desc)
    # '≈' may likewise be stored as '?'
    m_s = re.search(rf"storage scale\s*[≈~=:?]?\s*({NUM})", desc)
    lam = float(m_l.group(1)) if m_l else LAM_UM
    dn_enc = float(m_n.group(1)) if m_n else ARCHIVE_DELTA_N
    lo, hi = (float(m_w.group(1)), float(m_w.group(2))) if m_w else (WINDOW_LO, WINDOW_HI)
    scale_um_per_unit = float(m_s.group(1)) if m_s else (lam/(2*math.pi*dn_enc))   # encoding scale used by the converter
    h_um = lo + (u8 / 255.0) * (hi - lo)
    return h_um / scale_um_per_unit, px, desc     # native stored phase


def main():
    files = sorted(glob.glob(os.path.join(PLATES, "*_MEDIAN.tif")))
    print("%d archived plate fields" % len(files))
    rows = []
    t0 = time.time()
    for i, f in enumerate(files, 1):
        base = os.path.basename(f)
        m = re.match(r"(\d+)_(\d+)_MEDIAN\.tif$", base)
        if not m:
            continue
        well, nominal = int(m.group(1)), float(m.group(2))
        phase, px, desc = read_field(f)
        old_ang = re.search(r"Rotation applied \(deg\):\s*([-\d.]+)", desc)
        try:
            R = gk.measure_field(phase, px, nominal, do_orient=True)
        except Exception as exc:                  # estimator cannot run: record as a QC failure
            print("  %s: estimator failed (%s) -> qc_pass = False" % (base, exc))
            rows.append(dict(file=base, well=well, pitch_nominal_um=nominal,
                             pitch_measured_um=np.nan, phase_fundamental=np.nan,
                             phase_ptp=np.nan, angle_applied_deg=np.nan,
                             angle_archive_deg=(float(old_ang.group(1)) if old_ang else np.nan),
                             residual_rms=np.nan, n_periods=np.nan, px_um=px, qc_pass=False))
            continue
        depth_frac = abs(R["pitch_um"] - nominal) / nominal
        # maps written by the updated holo_to_tiff flag fields with < 3 registered frames
        reg_ok = "Registration QC: FAIL" not in desc
        qc = (depth_frac <= QC_PITCH_TOL
              and R["n_periods"] >= QC_MIN_PERIODS
              and R["residual_rms"] <= QC_MAX_RESIDUAL_FRAC * max(R["depth_fundamental"], 1e-9)
              and reg_ok)
        rows.append(dict(file=base, well=well, pitch_nominal_um=nominal,
                         pitch_measured_um=R["pitch_um"],
                         phase_fundamental=R["depth_fundamental"],
                         phase_ptp=R["depth"],
                         angle_applied_deg=R["angle_deg"],
                         angle_archive_deg=(float(old_ang.group(1)) if old_ang else np.nan),
                         residual_rms=R["residual_rms"], n_periods=R["n_periods"],
                         px_um=px, qc_pass=bool(qc)))
        if i % 25 == 0 or i == len(files):
            print("  %3d/%d  %.0f s elapsed" % (i, len(files), time.time() - t0), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(DATA, "plate_fields.csv"), index=False)
    print("\nQC: %d of %d fields pass" % (d.qc_pass.sum(), len(d)))
    g = d[d.qc_pass].groupby("pitch_nominal_um")
    print(g.agg(n=("phase_fundamental", "size"),
                phase=("phase_fundamental", "mean"),
                phase_sd=("phase_fundamental", "std"),
                pitch=("pitch_measured_um", "mean"),
                reorient=("angle_applied_deg", lambda s: float(np.mean(np.abs(s)))))
          .round(4))
    print("\nwrote data/plate_fields.csv")


if __name__ == "__main__":
    main()
