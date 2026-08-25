"""Forward simulation of Chiaro indentation files

Non-circularity
---------------
Four separate barriers:

1. **Opposite direction of computation.** The generator goes
   ``indentation -> force -> deflection -> piezo -> file``, simulating the
   instrument: the controller advances the sample displacement at a commanded
   velocity, the probe deflects by d = F/k, and the piezo position is whatever
   it must be, z = s + d. The estimators go the other way, ``file -> s ->
   contact -> modulus``. 

2. **Different mathematical primitives for the same physics.** Where an
   estimator models an effect, the generator produces it by a *different*
   construction, so agreement cannot come from using the same formula twice:

   =========================  ==============================================
   effect                     generator uses ... while the fitter uses ...
   =========================  ==============================================
   finite thickness           the Dimitriadis bonded series / the film
                              method corrects with the Garcia series
   compliant surface layer    an elastic foundation in series with the bulk,
                              solved parametrically in contact force / no
                              fitter contains any layer model at all
   depth grading              modulus sampled at 0.6 x contact radius / no
                              fitter contains any grading model
   adhesion                   the JKR parametric solution in contact radius /
                              no fitter contains adhesion
   time dependence            the Ting / Lee-Radok hereditary integral,
                              convolved numerically / every fitter is elastic
   =========================  ==============================================

3. The truth is written before anything is fitted into the file header
   and into a sidecar table.

4. The noise enters as an error on the measured deflection so it appears in force
   and in sample displacement with the opposite sign.


Marking
-------
Every synthetic file is marked in three independent places: the file name
begins with ``SYNTHETIC_``, the header carries ``SYNTHETIC  TRUE`` together
with the scenario and the ground-truth values, and the ``Comment:`` field -
which is present and empty in every real file - is filled with a warning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "SCENARIOS",
    "SynthConfig",
    "Truth",
    "generate_curve",
    "write_curve",
]

SYNTH_VERSION = "1.0.0"

#: Conversions. Forces are computed in newtons and written in uN; lengths are
#: computed in metres and written in nm.
NM = 1e-9
UM = 1e-6


@dataclass
class Truth:
    """Everything the generator knows and the estimators must not."""

    scenario: str
    e_star_true_Pa: float
    s0_true_nm: float
    radius_um: float
    spring_constant_n_m: float
    thickness_um: float = math.inf
    layer_thickness_um: float = 0.0
    layer_modulus_Pa: float = math.nan
    grading_length_um: float = math.inf
    grading_ratio: float = 1.0
    work_of_adhesion_J_m2: float = 0.0
    relax_times_s: tuple = ()
    relax_weights: tuple = ()
    e_instant_over_e_equilibrium: float = 1.0
    #: What an ideal elastic half-space Hertz fit *should* return over the
    #: analysis window, given the scenario. For scenarios that are a genuine
    #: half-space this equals ``e_star_true_Pa``; where the scenario adds a
    #: substrate or a layer it does not, and the validator is told which.
    expect_recovers_e_star: bool = True
    notes: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class SynthConfig:
    """Acquisition matched to the real files."""

    dt_s: float = 0.001
    dwell_s: float = 0.10
    #: Nominal approach length, only used when the surface search never
    #: trips. The real length is decided by ``trigger_force_nN``.
    approach_s: float = 1.30
    #: Where the surface is, in seconds of descent from the start. The real
    #: files put contact 1.2-1.4 s into the record.
    contact_time_s: float = 1.15
    #: The instrument stops its surface search and starts the profile when
    #: the force crosses this. Read off the real files: the vendor's own
    #: ``Indentation`` channel starts counting at 30-40 nN above baseline.
    trigger_force_nN: float = 35.0
    #: How far the search is allowed to run before giving up.
    max_approach_s: float = 3.00
    ramp_s: float = 1.00
    #: The hold and the retract are generated but never modelled or read.
    #: They exist so that the end of the ramp is something the pipeline has
    #: to detect rather than something the end of the file gives away.
    hold_s: float = 5.00
    retract_s: float = 1.00
    velocity_nm_s: float = 5000.0
    ramp_travel_nm: float = 5000.0
    #: Deflection measurement noise, in nm. 1.7 nm at k = 0.48 N/m is 0.8 nN,
    #: which is the median baseline sigma measured on the real curves.
    deflection_noise_nm: float = 1.7
    piezo_noise_nm: float = 0.8
    #: Slow ferrule-top oscillation: 88 Hz on the real files, a few nN.
    oscillation_nN: float = 0.8
    oscillation_hz: float = 88.0
    drift_nN_per_s: float = 1.0
    #: Where the probe starts, and where the sample surface is.
    s_start_nm: float = 60.0
    seed: int = 0


# ----------------------------------------------------------------------
# forward mechanics 
# ----------------------------------------------------------------------
def _hertz_force(delta_m, e_star_Pa, radius_m):
    """F = (4/3) E* sqrt(R) delta^{3/2} (rigid-sphere half-space law) """
    d = np.clip(delta_m, 0.0, None)
    return (4.0 / 3.0) * e_star_Pa * np.sqrt(radius_m) * np.power(d, 1.5)


def _dimitriadis_factor(delta_m, radius_m, thickness_m):
    """Bonded-film stiffening using the Dimitriadis series.

    """
    chi = np.sqrt(radius_m * np.clip(delta_m, 0.0, None)) / thickness_m
    return (
        1.0 + 1.133 * chi + 1.283 * chi**2 + 0.769 * chi**3 + 0.0975 * chi**4
    )


def _graded_modulus(
    delta_m, e_star_Pa, radius_m, length_m, ratio, mode="saturating"
):
    """Modulus rising with depth sampled where the strain field peaks.
    """
    a = np.sqrt(radius_m * np.clip(delta_m, 0.0, None))
    z = 0.6 * a
    if mode == "linear":
        # E(z) = E_surface (1 + z / length)
        return e_star_Pa * (1.0 + z / length_m)
    return e_star_Pa * (1.0 + (ratio - 1.0) * z / (z + length_m))


def _jkr_curve(e_star_Pa, radius_m, work_J_m2, n=4000):
    """JKR contact, parametric in contact radius; returns (delta, force).

    P = K a^3 / R - sqrt(6 pi w K a^3),  delta = a^2/R - (2/3) sqrt(6 pi w a/K)
    with K = (4/3) E*."""

    k_mod = (4.0 / 3.0) * e_star_Pa
    a = np.linspace(1e-12, 0.6 * radius_m, n)
    force = k_mod * a**3 / radius_m - np.sqrt(
        6.0 * math.pi * work_J_m2 * k_mod * a**3
    )
    delta = a**2 / radius_m - (2.0 / 3.0) * np.sqrt(
        6.0 * math.pi * work_J_m2 * a / k_mod
    )
    return delta, force


def _ting_generic(force_elastic_N, dt_s, weights, times):
    """Ting / Lee-Radok correspondence applied to elastic force laws.
    """
    increments = np.diff(force_elastic_N, prepend=0.0)
    n = increments.size
    lag = np.arange(n) * dt_s
    kernel = np.ones(n)
    for w, tau in zip(weights, times, strict=False):
        kernel = kernel + w * np.exp(-lag / tau)
    return np.convolve(increments, kernel)[:n]


def _layer_compression(delta_bulk_m, force_N, radius_m, tau_m, e_layer_Pa):
    """Compression of a soft surface film carrying the contact load.
    """
    a2 = radius_m * np.clip(delta_bulk_m, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        pressure = np.where(a2 > 0, force_N / (math.pi * a2), 0.0)
    return np.clip(pressure * tau_m / max(e_layer_Pa, 1e-9), 0.0, tau_m)


# ----------------------------------------------------------------------
# scenario table
# ----------------------------------------------------------------------
#: name -> (description, kwargs for ``generate_curve``)
SCENARIOS: dict[str, dict] = {
    "ideal": {
        "note": "pure Hertz half-space; the estimator must return the truth",
        "kw": {},
    },
    "ideal_100hz": {
        "note": "same, sampled at 100 Hz: the July campaign's time base",
        "kw": {"dt_s": 0.010},
    },
    "soft_2kPa": {
        "note": "2 kPa: the low end of the real range, worst force-to-noise",
        "kw": {"e_star_Pa": 2000.0},
    },
    "stiff_20kPa": {
        "note": "20 kPa: the high end; shallowest indentation reached",
        "kw": {"e_star_Pa": 20000.0},
    },
    "drift": {
        "note": "linear baseline drift at 8 nN/s, ~8x the real median",
        "kw": {"drift_nN_per_s": 8.0},
    },
    "drift_extreme": {
        "note": "40 nN/s: a thermally unsettled measurement, QC must reject",
        "kw": {"drift_nN_per_s": 40.0},
    },
    "oscillation_real": {
        "note": (
            "the ferrule-top wobble at the amplitude and frequency actually "
            "measured on the real files, 2 nN at 88 Hz"
        ),
        "kw": {"oscillation_nN": 2.0, "oscillation_hz": 88.0},
    },
    "oscillation_mild": {
        "note": "5 nN at 10 Hz: slower and larger than anything measured",
        "kw": {"oscillation_nN": 5.0, "oscillation_hz": 10.0},
    },
    "oscillation": {
        "note": (
            "20 nN at 1 Hz: a stress test 25x larger and 88x slower than the "
            "real wobble, where a fit window spans only one period"
        ),
        "kw": {"oscillation_nN": 20.0, "oscillation_hz": 1.0},
    },
    "film_50um": {
        "note": "bonded 50 um film: uncorrected Hertz must over-read",
        "kw": {"thickness_um": 50.0},
    },
    "film_20um": {
        "note": "bonded 20 um film: a large, unmistakable bottom effect",
        "kw": {"thickness_um": 20.0},
    },
    "layer_1um": {
        "note": "1 um surface film at a fifth of the bulk modulus",
        "kw": {"layer_thickness_um": 1.0, "layer_ratio": 0.2},
    },
    "layer_2um": {
        "note": "2 um surface film at a tenth of the bulk modulus",
        "kw": {"layer_thickness_um": 2.0, "layer_ratio": 0.1},
    },
    "graded": {
        "note": "modulus rising 2x with depth over a 4 um length scale",
        "kw": {"grading_length_um": 4.0, "grading_ratio": 2.0},
    },

    "graded_series_3kPa": {
        "note": "graded, surface E* = 3 kPa: the soft arm of a series",
        "kw": {
            "e_star_Pa": 3000.0,
            "grading_length_um": 4.0,
            "grading_ratio": 2.0,
        },
    },
    "graded_series_8kPa": {
        "note": "graded, surface E* = 8 kPa: the middle arm",
        "kw": {
            "e_star_Pa": 8000.0,
            "grading_length_um": 4.0,
            "grading_ratio": 2.0,
        },
    },
    "graded_series_16kPa": {
        "note": "graded, surface E* = 16 kPa: the stiff arm",
        "kw": {
            "e_star_Pa": 16000.0,
            "grading_length_um": 4.0,
            "grading_ratio": 2.0,
        },
    },

    "realistic_3kPa": {
        "note": "curve shape calibrated to the real data, bulk E* = 3 kPa",
        "kw": {
            "e_star_Pa": 3000.0,
            "layer_thickness_um": 0.5,
            "layer_ratio": 0.75,
            "thickness_um": 50.0,
            "grading_length_um": 2.0,
            "grading_mode": "linear",
        },
    },
    "realistic_8kPa": {
        "note": "curve shape calibrated to the real data, bulk E* = 8 kPa",
        "kw": {
            "e_star_Pa": 8000.0,
            "layer_thickness_um": 0.5,
            "layer_ratio": 0.75,
            "thickness_um": 50.0,
            "grading_length_um": 2.0,
            "grading_mode": "linear",
        },
    },
    "realistic_16kPa": {
        "note": "curve shape calibrated to the real data, bulk E* = 16 kPa",
        "kw": {
            "e_star_Pa": 16000.0,
            "layer_thickness_um": 0.5,
            "layer_ratio": 0.75,
            "thickness_um": 50.0,
            "grading_length_um": 2.0,
            "grading_mode": "linear",
        },
    },
    "adhesion_weak": {
        "note": "JKR, w = 0.05 mJ/m2: pull-off a few per cent of peak",
        "kw": {"work_of_adhesion_J_m2": 5e-5},
    },
    "adhesion_strong": {
        "note": "JKR, w = 0.5 mJ/m2: a large adhesive offset",
        "kw": {"work_of_adhesion_J_m2": 5e-4},
    },
    "viscoelastic": {
        "note": "SLS, tau = 0.5 s, instantaneous/equilibrium = 1.8",
        "kw": {"relax_times_s": (0.5,), "relax_weights": (0.8,)},
    },
    "poroviscoelastic": {
        "note": "two timescales, 0.1 s and 3 s, as a hydrated gel network",
        "kw": {"relax_times_s": (0.1, 3.0), "relax_weights": (0.5, 0.6)},
    },
    "no_contact": {
        "note": "QC extreme: the probe never reaches the surface",
        "kw": {"contact_offset_nm": 20000.0},
    },
    "starts_in_contact": {
        "note": (
            "QC extreme: the surface is above the starting position, so the "
            "probe is already loaded at the first sample and there is no "
            "baseline anywhere in the file"
        ),
        "kw": {"contact_offset_nm": -7000.0},
    },
    "shallow": {
        "note": (
            "QC extreme: the surface search is cut short, so the ramp starts "
            "before contact and the analysis window is never filled"
        ),
        "kw": {"approach_override_s": 0.45},
    },
    "hard_contact": {
        "note": (
            "QC extreme: a rigid inclusion or the glass showing through, "
            "E* = 300 kPa, peak force ~23 uN as seen on the real outliers"
        ),
        "kw": {"e_star_Pa": 3.0e5},
    },
    "snap_in": {
        "note": "QC extreme: a 60 nN discontinuity at contact",
        "kw": {"snap_in_nN": 60.0},
    },
    "slip": {
        "note": "QC extreme: the probe slips, force plateaus for 0.8 um",
        "kw": {"slip_span_nm": 800.0},
    },
    "spike": {
        "note": "QC extreme: a single-sample debris spike mid-loading",
        "kw": {"spike_nN": 200.0},
    },
    "noisy": {
        "note": "QC extreme: 12 nN of deflection noise",
        "kw": {"deflection_noise_nm": 25.0},
    },
}


# ----------------------------------------------------------------------
# generator
# ----------------------------------------------------------------------
def _profile(cfg: SynthConfig, approach_s: float | None = None) -> np.ndarray:
    """Commanded sample displacement s(t) for the four-step profile.

    ``approach_s`` is how long the surface search runs before the profile
    ramp begins."""
    n_dwell = round(cfg.dwell_s / cfg.dt_s)
    n_app = round(
        (cfg.approach_s if approach_s is None else approach_s) / cfg.dt_s
    )
    n_app = max(n_app, 2)
    n_ramp = round(cfg.ramp_s / cfg.dt_s)
    n_hold = round(cfg.hold_s / cfg.dt_s)
    n_ret = round(cfg.retract_s / cfg.dt_s)
    s = [np.full(n_dwell, cfg.s_start_nm)]
    approach = cfg.s_start_nm + cfg.velocity_nm_s * cfg.dt_s * np.arange(
        1, n_app + 1
    )
    s.append(approach)
    top_of_approach = approach[-1]
    ramp = (
        top_of_approach
        + cfg.ramp_travel_nm * np.arange(1, n_ramp + 1) / n_ramp
    )
    s.append(ramp)
    s.append(np.full(n_hold, ramp[-1]))
    s.append(ramp[-1] - cfg.ramp_travel_nm * np.arange(1, n_ret + 1) / n_ret)
    return np.concatenate(s)


def generate_curve(
    scenario: str,
    *,
    cfg: SynthConfig | None = None,
    e_star_Pa: float = 8000.0,
    radius_um: float = 27.0,
    spring_constant_n_m: float = 0.480,
    thickness_um: float = math.inf,
    layer_thickness_um: float = 0.0,
    layer_ratio: float = 1.0,
    grading_length_um: float = math.inf,
    grading_ratio: float = 1.0,
    grading_mode: str = "saturating",
    work_of_adhesion_J_m2: float = 0.0,
    relax_times_s: tuple = (),
    relax_weights: tuple = (),
    contact_offset_nm: float = 0.0,
    approach_override_s: float | None = None,
    snap_in_nN: float = 0.0,
    slip_span_nm: float = 0.0,
    spike_nN: float = 0.0,
    dt_s: float | None = None,
    deflection_noise_nm: float | None = None,
    oscillation_nN: float | None = None,
    oscillation_hz: float | None = None,
    drift_nN_per_s: float | None = None,
    seed: int = 0,
):
    """One synthetic curve: returns (columns dict, Truth, header dict)."""
    cfg = cfg or SynthConfig()
    if dt_s is not None:
        cfg = SynthConfig(**{**cfg.__dict__, "dt_s": dt_s})
    for name, value in (
        ("deflection_noise_nm", deflection_noise_nm),
        ("oscillation_nN", oscillation_nN),
        ("oscillation_hz", oscillation_hz),
        ("drift_nN_per_s", drift_nN_per_s),
    ):
        if value is not None:
            setattr(cfg, name, value)

    rng = np.random.default_rng(seed)
    radius_m = radius_um * UM
    # The surface sits at a fixed place; ``contact_offset_nm`` moves it, which
    # is how the QC extremes are made.
    s0_true = (
        cfg.s_start_nm
        + cfg.velocity_nm_s * cfg.contact_time_s
        + contact_offset_nm
    )

    def mechanics(s_nm: np.ndarray, dt_s: float) -> np.ndarray:
        """Force in newtons for a commanded displacement history.
        """
        delta = np.clip(s_nm - s0_true, 0.0, None) * NM
        if work_of_adhesion_J_m2 > 0:
            d_grid, f_grid = _jkr_curve(
                e_star_Pa, radius_m, work_of_adhesion_J_m2
            )
            order = np.argsort(d_grid)
            out = np.interp(
                s_nm * NM - s0_true * NM,
                d_grid[order],
                f_grid[order],
                left=0.0,
                right=f_grid[order][-1],
            )
            return np.where(s_nm * NM - s0_true * NM < d_grid.min(), 0.0, out)

        span = max(float(np.nanmax(delta)) * 1.3, 1e-9)
        d_bulk = np.linspace(0.0, span, 12000)
        modulus = (
            _graded_modulus(
                d_bulk,
                e_star_Pa,
                radius_m,
                grading_length_um * UM,
                grading_ratio,
                grading_mode,
            )
            if math.isfinite(grading_length_um)
            else e_star_Pa
        )
        f_bulk = _hertz_force(d_bulk, modulus, radius_m)
        if math.isfinite(thickness_um):
            f_bulk = f_bulk * _dimitriadis_factor(
                d_bulk, radius_m, thickness_um * UM
            )
        if layer_thickness_um > 0:
            d_total = d_bulk + _layer_compression(
                d_bulk,
                f_bulk,
                radius_m,
                layer_thickness_um * UM,
                max(layer_ratio, 1e-6) * e_star_Pa,
            )
        else:
            d_total = d_bulk
        elastic = np.interp(delta, d_total, f_bulk, left=0.0, right=f_bulk[-1])
        elastic = np.where(delta > 0, elastic, 0.0)
        if relax_times_s:
            return _ting_generic(elastic, dt_s, relax_weights, relax_times_s)
        return elastic


    # threshold trips
    n_probe = round((cfg.dwell_s + cfg.max_approach_s) / cfg.dt_s)
    t_probe = np.arange(n_probe) * cfg.dt_s
    s_probe = cfg.s_start_nm + cfg.velocity_nm_s * np.clip(
        t_probe - cfg.dwell_s, 0.0, None
    )
    f_probe = mechanics(s_probe, cfg.dt_s)
    tripped = np.flatnonzero(f_probe * 1e9 >= cfg.trigger_force_nN)
    approach_s = (
        float(t_probe[tripped[0]] - cfg.dwell_s)
        if tripped.size
        else cfg.max_approach_s
    )
    if approach_override_s is not None:

        approach_s = float(approach_override_s)
    approach_s = float(np.clip(approach_s, 2 * cfg.dt_s, cfg.max_approach_s))


    s_cmd = _profile(cfg, approach_s)
    n = s_cmd.size
    t = np.arange(n) * cfg.dt_s
    delta_m = np.clip(s_cmd - s0_true, 0.0, None) * NM
    force_N = mechanics(s_cmd, cfg.dt_s)


    in_contact = delta_m > 0
    if snap_in_nN > 0 and in_contact.any():
        first = int(np.flatnonzero(in_contact)[0])
        force_N = force_N.copy()
        force_N[first:] += snap_in_nN * 1e-9
    if slip_span_nm > 0 and in_contact.any():
        start = (
            int(np.flatnonzero(delta_m > 1500.0 * NM)[0])
            if (delta_m > 1500.0 * NM).any()
            else None
        )
        if start is not None:
            span = round(slip_span_nm / cfg.velocity_nm_s / cfg.dt_s)
            stop = min(start + span, n)
            force_N = force_N.copy()
            lost = force_N[stop - 1] - force_N[start]
            force_N[start:stop] = force_N[start]
            force_N[stop:] = force_N[stop:] - lost
            force_N = np.clip(force_N, 0.0, None)
    if spike_nN > 0 and in_contact.any():
        idx = (
            int(np.flatnonzero(delta_m > 2000.0 * NM)[0])
            if (delta_m > 2000.0 * NM).any()
            else None
        )
        if idx is not None:
            force_N = force_N.copy()
            force_N[idx] += spike_nN * 1e-9

    d_true_nm = force_N / spring_constant_n_m * 1e9
    piezo_true_nm = s_cmd + d_true_nm
    e_nm = rng.normal(0.0, cfg.deflection_noise_nm, n)
    e_nm = e_nm + (cfg.oscillation_nN / spring_constant_n_m) * np.sin(
        2 * math.pi * cfg.oscillation_hz * t + rng.uniform(0, 2 * math.pi)
    )
    e_nm = e_nm + (cfg.drift_nN_per_s / spring_constant_n_m) * t
    d_meas_nm = d_true_nm + e_nm
    piezo_meas_nm = piezo_true_nm + rng.normal(0.0, cfg.piezo_noise_nm, n)
    load_uN = spring_constant_n_m * d_meas_nm * 1e-3
    s_meas_nm = piezo_meas_nm - d_meas_nm

    vendor_s0_nm, e_eff_Pa = _emulate_vendor(
        t, load_uN, s_meas_nm, radius_um, cfg
    )
    indentation_nm = np.clip(s_meas_nm - vendor_s0_nm, 0.0, None)

    truth = Truth(
        scenario=scenario,
        e_star_true_Pa=e_star_Pa,
        s0_true_nm=s0_true,
        radius_um=radius_um,
        spring_constant_n_m=spring_constant_n_m,
        thickness_um=thickness_um,
        layer_thickness_um=layer_thickness_um,
        layer_modulus_Pa=(
            layer_ratio * e_star_Pa if layer_thickness_um > 0 else math.nan
        ),
        grading_length_um=grading_length_um,
        grading_ratio=grading_ratio,
        work_of_adhesion_J_m2=work_of_adhesion_J_m2,
        relax_times_s=tuple(relax_times_s),
        relax_weights=tuple(relax_weights),
        e_instant_over_e_equilibrium=1.0 + sum(relax_weights),
        expect_recovers_e_star=(
            not math.isfinite(thickness_um)
            and layer_thickness_um == 0.0
            and not math.isfinite(grading_length_um)
            and work_of_adhesion_J_m2 == 0.0
            and not relax_times_s
            and snap_in_nN == 0.0
            and slip_span_nm == 0.0
        ),
        notes=SCENARIOS.get(scenario, {}).get("note", ""),
        extra={
            "dt_s": cfg.dt_s,
            "peak_force_nN": float(np.nanmax(force_N) * 1e9),
            "max_true_indentation_nm": float(np.nanmax(delta_m) / NM),
            "vendor_s0_nm": vendor_s0_nm,
            "e_eff_Pa": e_eff_Pa,
            "approach_s": approach_s,
            "pre_ramp_indentation_nm": float(
                max(
                    0.0,
                    cfg.s_start_nm + cfg.velocity_nm_s * approach_s - s0_true,
                )
            ),
        },
    )
    columns = {
        "Time (s)": t,
        "Load (uN)": load_uN,
        "Indentation (nm)": indentation_nm,
        "Cantilever (nm)": d_meas_nm,
        "Piezo (nm)": piezo_meas_nm,
        "Auxiliary": 0.0153 + 1e-5 * np.arange(n),
    }
    header = _header(cfg, truth, approach_s)
    return columns, truth, header


def _emulate_vendor(t, load_uN, s_nm, radius_um, cfg: SynthConfig):
    """A stand-in for the instrument's own contact test and Hertz fit.
    """
    f_nN = load_uN * 1000.0
    early = t <= t[0] + 0.10
    base = float(np.median(f_nN[early]))
    sigma = float(1.4826 * np.median(np.abs(f_nN[early] - base))) or 1.0
    peak_i = int(np.nanargmax(f_nN))
    above = np.flatnonzero(
        (f_nN[: peak_i + 1] - base) > max(30.0, 8.0 * sigma)
    )
    if above.size == 0:
        return float("nan"), float("nan")
    s0 = float(s_nm[above[0]])
    depth = s_nm - s0
    win = (depth > 0) & (np.arange(f_nN.size) <= peak_i)
    if int(win.sum()) < 20:
        return s0, float("nan")
    x = np.power(depth[win] * NM, 1.5)
    y = (f_nN[win] - base) * 1e-9
    slope = float((x @ y) / (x @ x))
    e_eff = 0.75 * slope / math.sqrt(radius_um * UM)
    return s0, e_eff


def _header(cfg: SynthConfig, truth: Truth, approach_s: float) -> dict:
    """Header fields including truth lines."""
    ramp_start = cfg.dwell_s + approach_s
    return {
        "Date": "01/01/2000",
        "Time": "00:00:00",
        "Status": "OK",
        "k (N/m)": f"{truth.spring_constant_n_m:.3f}",
        "Tip radius (um)": f"{truth.radius_um:.3f}",
        "Calibration factor": "1.000",
        "SMDuration (s)": f"{ramp_start:.3f}",
        "D[Z1] (nm)": f"{cfg.ramp_travel_nm:.3f}",
        "t[1] (s)": f"{cfg.ramp_s:.3f}",
        "D[Z2] (nm)": f"{cfg.ramp_travel_nm:.3f}",
        "t[2] (s)": f"{cfg.hold_s:.3f}",
        "D[Z3] (nm)": "0.000",
        "t[3] (s)": f"{cfg.retract_s:.3f}",
        "step_start": (
            f"{ramp_start:.3f},"
            f"{ramp_start + cfg.ramp_s + cfg.dt_s:.3f},"
            f"{ramp_start + cfg.ramp_s + cfg.hold_s + cfg.dt_s:.3f}"
        ),
        "step_end": (
            f"{ramp_start + cfg.ramp_s:.3f},"
            f"{ramp_start + cfg.ramp_s + cfg.hold_s:.3f},"
            f"{ramp_start + cfg.ramp_s + cfg.hold_s + cfg.retract_s:.3f}"
        ),
        "P[max] (uN)": f"{truth.extra['peak_force_nN'] / 1000.0:.3f}",
        "E[eff] (Pa)": (
            f"{truth.extra['e_eff_Pa']:.3f}"
            if np.isfinite(truth.extra["e_eff_Pa"])
            else ""
        ),
    }


def write_curve(path: Path, columns: dict, truth: Truth, header: dict) -> None:
    """Write one file in the Chiaro layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Date\t{header['Date']}\tTime\t{header['Time']}"
        f"\tStatus\t{header['Status']}",
        path.parent.name,
        "Scan (#)\t1\tX (#)\t1\tY (#)\t1\tIndentation (#)\t1",
        "X-position (um)\t0.000",
        "Y-position (um)\t0.000",
        "Z-position (um)\t0.000",
        "Z surface (um)\t5.000",
        "Piezo position (nm) (Measured)\t0.0",
        "",
        f"k (N/m)\t{header['k (N/m)']}",
        f"Tip radius (um)\t{header['Tip radius (um)']}",
        f"Calibration factor\t{header['Calibration factor']}",
        f"SMDuration (s) {header['SMDuration (s)']}",
        "",
        "Device:\tChiaro",
        "Software version: V3.5.0",
        "",
        "Control mode: Indentation",
        "Measurement: Indentation",
        "",
        "Profile:",
        f"D[Z1] (nm)\t{header['D[Z1] (nm)']}\tt[1] (s)\t{header['t[1] (s)']}",
        f"D[Z2] (nm)\t{header['D[Z2] (nm)']}\tt[2] (s)\t{header['t[2] (s)']}",
        f"D[Z3] (nm)\t{header['D[Z3] (nm)']}\tt[3] (s)\t{header['t[3] (s)']}",
        f"Step absolute start times (s)\t{header['step_start']}",
        f"Step absolute end times (s) {header['step_end']}",
        "",
        "Model: Hertz",
        f"P[max] (uN)\t{header['P[max] (uN)']}",
        "D[max] (nm)\t0.000",
        "D[final] (nm)\t0.000",
        "D[max-final] (nm)\t0.000",
        "Slope (N/m)\t0.000",
        f"E[eff] (Pa)\t{header['E[eff] (Pa)']}",
        "E[v=0.495] (Pa)\t0.000",
        "",
        "SYNTHETIC\tTRUE",
        f"SYNTHETIC generator\tlights_synth {SYNTH_VERSION}",
        f"SYNTHETIC scenario\t{truth.scenario}",
        f"SYNTHETIC note\t{truth.notes}",
        f"SYNTHETIC E_star_true (Pa)\t{truth.e_star_true_Pa:.6f}",
        f"SYNTHETIC s0_true (nm)\t{truth.s0_true_nm:.6f}",
        f"SYNTHETIC thickness (um)\t{truth.thickness_um}",
        f"SYNTHETIC layer_thickness (um)\t{truth.layer_thickness_um}",
        f"SYNTHETIC layer_modulus (Pa)\t{truth.layer_modulus_Pa}",
        f"SYNTHETIC grading_length (um)\t{truth.grading_length_um}",
        f"SYNTHETIC grading_ratio\t{truth.grading_ratio}",
        f"SYNTHETIC work_of_adhesion (J/m2)\t{truth.work_of_adhesion_J_m2}",
        f"SYNTHETIC relax_times (s)\t{list(truth.relax_times_s)}",
        f"SYNTHETIC relax_weights\t{list(truth.relax_weights)}",
        f"SYNTHETIC expect_recovers_E_star\t{truth.expect_recovers_e_star}",
        "SYNTHETIC modelled_segment\tsurface search and loading ramp",
        "Comment: SYNTHETIC DATA - GENERATED BY lights_synth, NOT A "
        "MEASUREMENT. Do not mix with experimental files.",
        "",
        "Time (s)\tLoad (uN)\tIndentation (nm)\tCantilever (nm)\t"
        "Piezo (nm)\tAuxiliary",
    ]
    body = np.column_stack([columns[k] for k in columns])
    rows = "\n".join("\t".join(f"{v:.6f}" for v in row) for row in body)
    path.write_text("\n".join(lines) + "\n" + rows + "\n", encoding="utf-8")
