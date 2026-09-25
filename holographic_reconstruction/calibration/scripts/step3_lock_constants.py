# -*- coding: utf-8 -*-
"""Step 3 - lock the calibration constants, propagate the uncertainty, and test
whether the calibration transfers to the historical plate dataset.

Definitions used throughout (all convention-free unless stated):

    phi      native stored phase value of a HoloMonitor field (dimensionless)
    a        confocal apparent optical groove depth, in nominal z micrometres
    d        physical groove depth, micrometres

    d = zeta * a            zeta  = optical-to-physical axial factor
    d = C    * phi          C     = micrometres of relief per native phase unit

Both constants are fixed by the calibration specimen against nanoindentation:

    zeta = d_nano / a_cal            C = d_nano / phi_cal

The quantity that can be tested against the plates without the nanoindentation
is their ratio, which is a pure observable of the two optical instruments:

    Lambda = C / zeta = a / phi

Lambda carries no wavelength and no phase-unit convention, so it is the right
transfer statistic.  Delta n_eff is reported as a derived quantity.  HoloMonitor
AppSuite stores phase in units of one wavelength of optical path (OPL = phi *
lambda), so the physical value is delta_n_eff_wavelength_convention = lambda / C;
the radian-convention number is kept only as the numerically equivalent constant
for converters that use h = phi * lambda / (2 pi dn).

Writes outputs/calibration_constants.json, outputs/uncertainty_budget.csv,
       outputs/group_summary.csv, outputs/figure6f_source_data.csv
"""
import os, json, math, sys, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

# plate confocal depths: groove_analyzer.py recap of the 1536-well plates
# (apparent optical depths, pitch-bias corrected by groove_analyzer.py)
_ap = argparse.ArgumentParser()
_ap.add_argument("--confocal-recap",
                 default=os.path.join(HERE, "raw", "groove_recap_260805_confo.csv"))
CONFOCAL_RECAP = _ap.parse_args().confocal_recap

LAM_UM = 0.635
NA = 0.75
N_IMMERSION = 1.0            # dry objective
N_MEDIUM = 1.3316            # PBS at 635 nm, 37 C
DZ_UM = 0.900                # confocal z step
QC_DEPTH_OVER_PITCH = 0.5    # pre-declared confocal QC


def axial_bounds(n2, n1=N_IMMERSION, na=NA):
    paraxial = n2 / n1
    marginal = (math.tan(math.asin(min(na / n1, 1.0)))
                / math.tan(math.asin(min(na / n2, 1.0))))
    lyakin = math.sqrt((n2 ** 2 - na ** 2 / 2.0) / (n1 ** 2 - na ** 2 / 2.0))
    return dict(paraxial=paraxial, lyakin_stallinga=lyakin, marginal_ray=marginal)


# ------------------------------------------------------------ calibration set
cal = pd.read_csv(os.path.join(DATA, "calibration_fields.csv"))
src = np.load(os.path.join(DATA, "calibration_source.npz"))

nano = cal[cal.modality == "nanoindentation"].iloc[0]
d_nano = float(nano.value_fundamental)
u_d_nano = float(src["nano_depth_sd"])                    # bootstrap over columns

conf = cal[cal.modality == "confocal"]
a_cal = float(conf.value_fundamental.mean())
u_a_typeA = float(conf.value_fundamental.std(ddof=1) / np.sqrt(len(conf)))

hpi = cal[cal.modality == "HPI"]
phi_cal = float(hpi.value_fundamental.mean())
u_phi_typeA = float(hpi.value_fundamental.std(ddof=1) / np.sqrt(len(hpi)))

# Type B: estimator form, from the spread between the fundamental and the full
# harmonic peak-to-trough of the same profile, treated as a rectangular interval
def type_b_form(sub):
    rel = np.abs(sub.value.values - sub.value_fundamental.values) / sub.value_fundamental.values
    return float(np.mean(rel) / math.sqrt(3.0))


u_d_form = d_nano * type_b_form(cal[cal.modality == "nanoindentation"])
u_a_form = a_cal * type_b_form(conf)
u_phi_form = phi_cal * type_b_form(hpi)
# Type B: the confocal surface is quantised at one z step, and a depth is a
# difference of two surfaces
u_a_quant = math.sqrt(2.0) * DZ_UM / math.sqrt(12.0)

u_d = math.hypot(u_d_nano, u_d_form)
u_a = math.sqrt(u_a_typeA ** 2 + u_a_form ** 2 + u_a_quant ** 2)
u_phi = math.hypot(u_phi_typeA, u_phi_form)

zeta = d_nano / a_cal
C = d_nano / phi_cal
Lambda_cal = a_cal / phi_cal
u_zeta = zeta * math.hypot(u_d / d_nano, u_a / a_cal)
u_C = C * math.hypot(u_d / d_nano, u_phi / phi_cal)
u_Lambda_cal = Lambda_cal * math.hypot(u_a / a_cal, u_phi / phi_cal)

dn_wave = LAM_UM / C                       # stored phase read as wavelengths
dn_rad = LAM_UM / (2 * math.pi * C)        # stored phase read as radians
bounds = axial_bounds(N_MEDIUM + dn_wave)

print("=" * 74)
print("CALIBRATION SPECIMEN  (40 um pitch, 75 mg/mL GelMA)")
print("=" * 74)
print("  nanoindentation physical depth   d_nano  = %7.3f +/- %.3f um" % (d_nano, u_d))
print("  confocal apparent optical depth  a_cal   = %7.3f +/- %.3f um  (n=3)" % (a_cal, u_a))
print("  HPI phase contrast               phi_cal = %7.4f +/- %.4f    (n=5)" % (phi_cal, u_phi))
print()
print("  zeta   = d_nano / a_cal   = %6.3f +/- %.3f" % (zeta, u_zeta))
print("  C      = d_nano / phi_cal = %6.3f +/- %.3f um per native phase unit" % (C, u_C))
print("  Lambda = a_cal  / phi_cal = %6.3f +/- %.3f um per native phase unit" % (Lambda_cal, u_Lambda_cal))
print()
print("  delta_n_eff = %.5f  (stored phase read as wavelengths) -> n_gel = %.4f"
      % (dn_wave, N_MEDIUM + dn_wave))
print("  delta_n_eff = %.5f  (stored phase read as radians)     -> n_gel = %.4f"
      % (dn_rad, N_MEDIUM + dn_rad))
print("  axial bounds at n_gel = %.4f : paraxial %.4f | Lyakin/Stallinga %.4f | marginal %.4f"
      % (N_MEDIUM + dn_wave, bounds["paraxial"], bounds["lyakin_stallinga"], bounds["marginal_ray"]))
adm = bounds["paraxial"] <= zeta <= bounds["marginal_ray"]
print("  zeta %.3f is %s the geometrical-optics band"
      % (zeta, "INSIDE" if adm else "OUTSIDE"))

# ------------------------------------------------------------ historical set
plate = pd.read_csv(os.path.join(DATA, "plate_fields.csv"))
plate = plate[plate.qc_pass]
c = pd.read_csv(CONFOCAL_RECAP)
c["pitch_nominal_um"] = c.file.str.extract(r"_(\d+)\.tiff")[0].astype(float)
c = c[c.depth_um / c.pitch_um <= QC_DEPTH_OVER_PITCH]

pitches = [40.0, 60.0, 80.0]
rowsL = []
for p in pitches:
    ph = plate[plate.pitch_nominal_um == p].phase_fundamental
    ap = c[c.pitch_nominal_um == p].depth_um
    if len(ph) < 2 or len(ap) < 2:
        raise SystemExit("pitch %g: only %d HPI and %d confocal fields survive QC; "
                         "check the QC thresholds in step2." % (p, len(ph), len(ap)))
    L = ap.mean() / ph.mean()
    uL = L * math.hypot(ap.std(ddof=1) / math.sqrt(len(ap)) / ap.mean(),
                        ph.std(ddof=1) / math.sqrt(len(ph)) / ph.mean())
    rowsL.append(dict(pitch_um=p, n_hpi=len(ph), phase_mean=ph.mean(), phase_sd=ph.std(ddof=1),
                      n_confocal=len(ap), apparent_mean=ap.mean(), apparent_sd=ap.std(ddof=1),
                      Lambda=L, u_Lambda=uL))
L = pd.DataFrame(rowsL)

print()
print("=" * 74)
print("TRANSFER TEST   Lambda = apparent optical depth per native phase unit")
print("=" * 74)
for _, r in L.iterrows():
    print("  plates %2.0f um  n=%3d/%3d   Lambda = %6.3f +/- %.3f   (%+5.1f %% vs specimen)"
          % (r.pitch_um, r.n_hpi, r.n_confocal, r.Lambda, r.u_Lambda,
             100 * (r.Lambda - Lambda_cal) / Lambda_cal))
print("  specimen                     Lambda = %6.3f +/- %.3f" % (Lambda_cal, u_Lambda_cal))

# the sharpest internal statement: the plates alone, with no specimen and no
# calibration constant, already show that 40 um is not on the same optical
# transfer as 60 and 80 um
anch = L[L.pitch_um.isin([60.0, 80.0])]
w = 1.0 / anch.u_Lambda.values ** 2
L_anchor = float(np.sum(w * anch.Lambda.values) / np.sum(w))
u_anchor = float(np.sqrt(1.0 / np.sum(w)))
spread_6080 = float(100 * (anch.Lambda.values[1] - anch.Lambda.values[0]) / L_anchor)
L40 = float(L[L.pitch_um == 40.0].Lambda.iloc[0])
u40 = float(L[L.pitch_um == 40.0].u_Lambda.iloc[0])
print()
print("  60/80 anchor          Lambda = %6.3f +/- %.3f   (60 vs 80 spread %+.1f %%)"
      % (L_anchor, u_anchor, spread_6080))
print("  40 um vs that anchor         %+.1f %% +/- %.1f %%   -> %.1f sigma"
      % (100 * (L40 - L_anchor) / L_anchor,
         100 * math.hypot(u40, u_anchor) / L_anchor,
         abs(L40 - L_anchor) / math.hypot(u40, u_anchor)))
print("  specimen vs that anchor      %+.1f %% +/- %.1f %%"
      % (100 * (Lambda_cal - L_anchor) / L_anchor,
         100 * math.hypot(u_Lambda_cal, u_anchor) / L_anchor))

# --------------------------------------------------------------- Figure 6f
rows = []
for p in pitches:
    sub = plate[plate.pitch_nominal_um == p]
    for _, r in sub.iterrows():
        rows.append(dict(pitch_design_um=p, well=r.well, file=r.file,
                         phase_native=r.phase_fundamental,
                         physical_depth_um=C * r.phase_fundamental,
                         pitch_measured_um=r.pitch_measured_um, modality="HPI"))
f6 = pd.DataFrame(rows)
f6.to_csv(os.path.join(OUT, "figure6f_source_data.csv"), index=False)

grp = []
for p in pitches:
    dh = f6[f6.pitch_design_um == p].physical_depth_um
    ap = c[c.pitch_nominal_um == p].depth_um
    dc = zeta * ap
    diff = dh.mean() - dc.mean()
    # Expanded uncertainty of the difference, JCGM 100:2008, k = 2.
    #
    # d_HPI / d_confocal = (C * phi) / (zeta * a) = Lambda_specimen / Lambda_plate,
    # so the nanoindentation depth cancels exactly: it enters C and zeta with the
    # same sign and to the same power.  Propagating u(C) and u(zeta) as if they
    # were independent would count d_nano twice and inflate the tolerance.  The
    # systematic term is therefore u(Lambda_specimen) alone.
    Lp = L[L.pitch_um == p].iloc[0]
    rel = math.sqrt((u_Lambda_cal / Lambda_cal) ** 2          # calibration transfer
                    + (Lp.u_Lambda / Lp.Lambda) ** 2          # plate Type A
                    + (u_a_quant / Lp.apparent_mean) ** 2)    # confocal axial quantisation
    uc = rel * dc.mean()
    grp.append(dict(pitch_design_um=p, hpi_n=len(dh), hpi_mean_um=dh.mean(), hpi_sd_um=dh.std(ddof=1),
                    confocal_n=len(dc), confocal_mean_um=dc.mean(), confocal_sd_um=dc.std(ddof=1),
                    confocal_apparent_um=ap.mean(),
                    difference_um=diff, difference_pct=100 * diff / dc.mean(),
                    u_combined_um=uc, U_k2_um=2 * uc, U_k2_pct=200 * uc / dc.mean(),
                    agrees=bool(abs(diff) <= 2 * uc)))
G = pd.DataFrame(grp)
G.to_csv(os.path.join(OUT, "group_summary.csv"), index=False)

print()
print("=" * 74)
print("FIGURE 6f  physical groove depth")
print("=" * 74)
print(G[["pitch_design_um", "hpi_n", "hpi_mean_um", "hpi_sd_um", "confocal_n",
         "confocal_mean_um", "confocal_sd_um", "difference_um", "difference_pct",
         "U_k2_um", "U_k2_pct", "agrees"]].round(3).to_string(index=False))

budget = pd.DataFrame([
    dict(quantity="d_nano", term="Type A, bootstrap over columns", value=u_d_nano, unit="um"),
    dict(quantity="d_nano", term="Type B, estimator form", value=u_d_form, unit="um"),
    dict(quantity="a_confocal", term="Type A, 3 fields (SEM)", value=u_a_typeA, unit="um"),
    dict(quantity="a_confocal", term="Type B, estimator form", value=u_a_form, unit="um"),
    dict(quantity="a_confocal", term="Type B, axial quantisation sqrt(2)*dz/sqrt(12)", value=u_a_quant, unit="um"),
    dict(quantity="phi_HPI", term="Type A, 5 fields (SEM)", value=u_phi_typeA, unit="phase"),
    dict(quantity="phi_HPI", term="Type B, estimator form", value=u_phi_form, unit="phase"),
    dict(quantity="zeta", term="combined", value=u_zeta, unit="-"),
    dict(quantity="C", term="combined", value=u_C, unit="um/phase"),
    dict(quantity="Lambda", term="combined", value=u_Lambda_cal, unit="um/phase"),
    dict(quantity="Lambda", term="relative, dominates the Fig. 6f tolerance",
         value=u_Lambda_cal / Lambda_cal, unit="fraction"),
])
budget.to_csv(os.path.join(OUT, "uncertainty_budget.csv"), index=False)

const = dict(
    specimen=dict(pitch_design_um=40.0, formulation="GelMA 75 mg/mL",
                  d_nano_um=d_nano, u_d_nano_um=u_d,
                  a_confocal_um=a_cal, u_a_confocal_um=u_a, n_confocal_fields=int(len(conf)),
                  phi_hpi=phi_cal, u_phi_hpi=u_phi, n_hpi_fields=int(len(hpi))),
    constants=dict(zeta=zeta, u_zeta=u_zeta,
                   C_um_per_phase_unit=C, u_C=u_C,
                   Lambda_um_per_phase_unit=Lambda_cal, u_Lambda=u_Lambda_cal,
                   delta_n_eff_wavelength_convention=dn_wave,
                   delta_n_eff_radian_convention=dn_rad,
                   n_gel_implied=N_MEDIUM + dn_wave),
    optics=dict(wavelength_um=LAM_UM, NA=NA, n_immersion=N_IMMERSION,
                n_medium_PBS=N_MEDIUM, dz_um=DZ_UM, axial_bounds=bounds,
                zeta_inside_band=bool(adm)),
    transfer=dict(Lambda_specimen=Lambda_cal, u_Lambda_specimen=u_Lambda_cal,
                  Lambda_anchor_60_80=L_anchor, u_Lambda_anchor=u_anchor,
                  pct_40um_vs_anchor=100 * (L40 - L_anchor) / L_anchor,
                  pct_specimen_vs_anchor=100 * (Lambda_cal - L_anchor) / L_anchor,
                  Lambda_plates={str(int(r.pitch_um)): r.Lambda for _, r in L.iterrows()},
                  Lambda_plates_pct_vs_specimen={
                      str(int(r.pitch_um)): 100 * (r.Lambda - Lambda_cal) / Lambda_cal
                      for _, r in L.iterrows()}),
    qc=dict(confocal="depth/pitch <= 0.5",
            hpi="pitch within 10 % of nominal, >= 4 periods, residual RMS <= 0.6 x depth"),
    references=[
        "Carlsson K, J Microsc 163, 167 (1991) - paraxial axial scaling",
        "Visser T D et al., Optik 90, 17 (1992) - marginal-ray axial scaling",
        "Lyakin D V et al., Opt Spectrosc 99, 515 (2005); Stallinga S, Appl Opt 44, 849 (2005)",
        "JCGM 100:2008 - propagation of uncertainty",
    ])
L.to_csv(os.path.join(OUT, "transfer_test.csv"), index=False)
with open(os.path.join(OUT, "calibration_constants.json"), "w") as fh:
    json.dump(const, fh, indent=2)
print("\nwrote outputs/calibration_constants.json, uncertainty_budget.csv,")
print("      transfer_test.csv, group_summary.csv, figure6f_source_data.csv")
