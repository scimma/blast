#!/usr/bin/env python
# D. Jones - 5/26/23
# modified by B. Wang - 6/7/23
"""Implementation of SBI++ training for Blast"""
import argparse
import glob
import json
import math
import os
import pickle
import sys

# ---------------------------------------------------------------------------
# Django bootstrap. Django is only set up when something actually needs the
# database (the filter list used to simulate photometry, i.e. --do-phot).
# --do-train runs without loading any Django apps. Under
# `python manage.py runcrons ...` Django is already set up and this is a no-op.
# ---------------------------------------------------------------------------
_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.settings")


def _django_ready():
    try:
        from django.apps import apps
    except ImportError:
        return False
    return apps.ready


def _setup_django():
    """Load the Django apps (and database connection) on first use."""
    if not _django_ready():
        import django

        django.setup()


# django_cron's models need a configured Django, so only import it when Django
# is already running (runcrons); standalone runs use a stand-in base class.
if _django_ready():
    from django_cron import CronJobBase
    from django_cron import Schedule
else:

    class CronJobBase:
        pass

    def Schedule(**kwargs):
        return None


import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.cosmology import WMAP9 as cosmo
from prospect.fitting import fit_model as fit_model_prospect
from prospect.io import write_results as writer
from prospect.models import priors
from prospect.models import SpecModel
from prospect.models.sedmodel import PolySpecModel
from prospect.models.templates import TemplateLibrary
from prospect.models import priors_beta
from prospect.sources import CSPSpecBasis
from prospect.sources import FastStepBasis
from prospect.utils.obsutils import fix_obs
from sbi import inference as Inference
from sbi.neural_nets import posterior_nn

# ---------------------------------------------------------------------------
# Defaults (all overridable on the command line; paths are relative to app/)
# ---------------------------------------------------------------------------
DEFAULT_ZMIN = 0.0
DEFAULT_ZMAX = 0.2
DEFAULT_NSAMPLES = 150000
DEFAULT_PHOT_FILE = "host/SBI/sbi_phot_1.h5"
DEFAULT_TRAIN_FILES = [f"host/SBI/sbi_phot_{i}.h5" for i in range(1, 11)]
DEFAULT_MERGED_FILE = "host/SBI/sbi_phot.h5"
DEFAULT_MODEL_FILE = "host/SBI/SBI_model.pt"  # trained ANPE state dict
DEFAULT_SUMMARY_FILE = "host/SBI/SBI_model_summary.p"  # training summary (convergence etc.)
# --do-zphot-model defaults, kept separate so the two models never overwrite each other
DEFAULT_ZPHOT_MERGED_FILE = "host/SBI/sbi_phot_zphot.h5"
DEFAULT_ZPHOT_MODEL_FILE = "host/SBI/SBI_model_zphot.pt"
DEFAULT_ZPHOT_SUMMARY_FILE = "host/SBI/SBI_model_zphot_summary.p"

ZRED_PRIOR_PAD = 1e-3  # upper edge of the zred prior is zmax + this
MASSFUNC_ZMIN = 0.2  # below this z, logmass is drawn flat instead of from the mass function

EXCLUDED_FILTERS = ("DES_i", "DES_Y")

# Filters for the simulated photometry, in the order the database returns them
# for Filter.objects.filter(~Q(name="DES_i") & ~Q(name="DES_Y")). The trained
# model's inputs are in this order, so it must match the fitting code. Update
# it by hand if filters are added to the database (or run with --filters db).
DEFAULT_FILTERS = [
    "2MASS_H", "2MASS_J", "2MASS_K",
    "WISE_W1", "WISE_W2", "WISE_W3", "WISE_W4",
    "GALEX_NUV", "GALEX_FUV",
    "PanSTARRS_g", "PanSTARRS_r", "PanSTARRS_i", "PanSTARRS_z", "PanSTARRS_y",
    "SDSS_i", "SDSS_r", "SDSS_u", "SDSS_z", "SDSS_g",
    "DES_g", "DES_r", "DES_z",
]

_all_filters = None
_filter_file = os.environ.get("SBI_FILTER_FILE") or None  # set by --filter-file
_filter_names = list(DEFAULT_FILTERS)  # set by --filters; ["db"] = use the database
# same location as settings.TRANSMISSION_CURVES_ROOT in a standard checkout
DEFAULT_TRANSMISSION_DIR = os.environ.get(
    "TRANSMISSION_CURVES_ROOT",
    os.path.abspath(os.path.join(_APP_DIR, "..", "data", "transmission")),
)
_transmission_dir = DEFAULT_TRANSMISSION_DIR


class CachedFilter:
    """Stand-in for host.models.Filter, built from a --filter-file export.

    Provides the two things this script uses: .name and .transmission_curve().
    """

    def __init__(self, name, wavelength, transmission):
        self.name = name
        self.wavelength = np.asarray(wavelength, dtype=float)
        self.transmission = np.asarray(transmission, dtype=float)

    def __str__(self):
        return self.name

    def transmission_curve(self):
        from sedpy import observate

        return observate.Filter(
            kname=self.name, nick=self.name, data=(self.wavelength, self.transmission)
        )


def _load_db_filters():
    _setup_django()
    from django.db.models import Q
    from host.models import Filter

    query = Q()
    for name in EXCLUDED_FILTERS:
        query &= ~Q(name=name)
    return list(Filter.objects.filter(query))


def _load_filter_file(path):
    with open(path) as fin:
        data = json.load(fin)
    return [
        CachedFilter(f["name"], f["wavelength"], f["transmission"])
        for f in data["filters"]
    ]


def _read_curve(path):
    """Read a transmission curve the same way Filter.transmission_curve() does."""
    import pandas as pd

    curve = pd.read_csv(path, sep=r"\s+", header=None)
    return curve[0].to_numpy(), curve[1].to_numpy()


def _load_transmission_dir(directory, names):
    """Build filters straight from the transmission-curve files.

    names == ["auto"] uses every curve in the directory (minus *_corrmodel
    files and EXCLUDED_FILTERS) in alphabetical order.
    """
    if list(names) == ["auto"]:
        names = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(directory)
            if f.endswith(".txt") and not f.endswith("_corrmodel.txt")
        )
        names = [n for n in names if n not in EXCLUDED_FILTERS]
        print(
            "WARNING: --filters auto uses alphabetical order. The model's inputs "
            "must be in the same filter order when it is applied.",
            flush=True,
        )
    filters = []
    for name in names:
        path = os.path.join(directory, f"{name}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No transmission curve for {name}: {path}")
        filters.append(CachedFilter(name, *_read_curve(path)))
    return filters


def get_all_filters():
    """Filters used for the simulated photometry, in a fixed order.

    In priority order: the --filter-file export, or the --filters names
    (DEFAULT_FILTERS unless overridden) read from --transmission-dir.
    --filters db reads the database instead, which is the only option that
    needs Django.
    """
    global _all_filters
    if _all_filters is None:
        if _filter_file:
            _all_filters = _load_filter_file(_filter_file)
            source = _filter_file
        elif _filter_names == ["db"]:
            _all_filters = _load_db_filters()
            source = "the database"
        else:
            _all_filters = _load_transmission_dir(_transmission_dir, _filter_names)
            source = _transmission_dir
        print(
            f"Using {len(_all_filters)} filters from {source}: "
            f"{[f.name for f in _all_filters]}",
            flush=True,
        )
    return _all_filters


def export_filters(path):
    """Write the filter list and raw transmission curves to a JSON file.

    Run once wherever the database works; afterwards --do-phot can use
    --filter-file and never needs Django. The file fixes the filter order,
    which must match the order used when the trained model is applied.
    """
    filters = _load_db_filters()
    from django.conf import settings

    out = []
    for f in filters:
        wavelength, transmission = _read_curve(
            f"{settings.TRANSMISSION_CURVES_ROOT}/{f.name}.txt"
        )
        out.append(
            {
                "name": f.name,
                "wavelength": wavelength.tolist(),
                "transmission": transmission.tolist(),
            }
        )
    _ensure_parent_dir(path)
    with open(path, "w") as fout:
        json.dump(
            {
                "excluded": list(EXCLUDED_FILTERS),
                "transmission_curves_root": str(settings.TRANSMISSION_CURVES_ROOT),
                "filters": out,
            },
            fout,
        )
    print(f"Wrote {len(out)} filters to {path}: {[f['name'] for f in out]}", flush=True)

massmet = np.loadtxt("host/SBI/priors/gallazzi_05_massmet.txt")

nhidden = 500  # architecture
nblocks = 15  # architecture

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def maggies_to_asinh(x):
    """asinh magnitudes"""
    a = 2.50 * np.log10(np.e)
    mu = 35.0
    return -a * math.asinh((x / 2.0) * np.exp(mu / a)) + mu


def build_obs(**extras):  ##transient, aperture_type):
    """
    This functions is required by prospector and should return
    a dictionary defined by
    https://prospect.readthedocs.io/en/latest/dataformat.html.

    """
    all_filters = get_all_filters()
    filters = []
    for filter in all_filters:
        filters.append(filter.transmission_curve())

    obs_data = dict(
        wavelength=None,
        spectrum=None,
        unc=None,
        mask=None,
        ##redshift=z,
        maggies=np.ones(len(all_filters)),  # np.array(flux_maggies),
        maggies_unc=np.ones(len(all_filters)),
        filters=filters,
    )
    obs_data["phot_wave"] = np.array([f.wave_effective for f in obs_data["filters"]])
    obs_data["phot_mask"] = [True] * len(obs_data["filters"])

    return fix_obs(obs_data)


def build_model(obs=None, zmin=DEFAULT_ZMIN, zmax=DEFAULT_ZMAX, **extras):
    """prospector-alpha"""
    fit_order = [
        "zred",
        "logmass",
        "logzsol",
        "logsfr_ratios",
        "dust2",
        "dust_index",
        "dust1_fraction",
        "log_fagn",
        "log_agn_tau",
        "gas_logz",
        "duste_qpah",
        "duste_umin",
        "log_duste_gamma",
    ]

    # -------------
    # MODEL_PARAMS
    model_params = {}

    # --- BASIC PARAMETERS ---
    model_params["zred"] = {
        "N": 1,
        "isfree": True,
        "init": 0.5,
        "prior": priors.FastUniform(a=zmin, b=zmax + ZRED_PRIOR_PAD),
    }

    model_params["logmass"] = {
        "N": 1,
        "isfree": True,
        "init": 8.0,
        "units": "Msun",
        "prior": priors.FastUniform(a=6.0, b=12.0),
    }

    model_params["logzsol"] = {
        "N": 1,
        "isfree": True,
        "init": -0.5,
        "units": r"$\log (Z/Z_\odot)$",
        "prior": priors.FastUniform(a=-1.98, b=0.19),
    }

    model_params["imf_type"] = {
        "N": 1,
        "isfree": False,
        "init": 1,  # 1 = chabrier
        "units": None,
        "prior": None,
    }
    model_params["add_igm_absorption"] = {"N": 1, "isfree": False, "init": True}
    model_params["add_agb_dust_model"] = {"N": 1, "isfree": False, "init": True}
    model_params["pmetals"] = {"N": 1, "isfree": False, "init": -99}

    # --- SFH ---
    nbins_sfh = 7
    model_params["sfh"] = {"N": 1, "isfree": False, "init": 3}
    model_params["logsfr_ratios"] = {
        "N": 6,
        "isfree": True,
        "init": 0.0,
        "prior": priors.FastTruncatedEvenStudentTFreeDeg2(
            hw=np.ones(6) * 5.0, sig=np.ones(6) * 0.3
        ),
    }

    # add redshift scaling to agebins, such that
    # t_max = t_univ
    def zred_to_agebins(zred=None, **extras):
        amin = 7.1295
        nbins_sfh = 7
        tuniv = cosmo.age(zred)[0].value * 1e9
        tbinmax = tuniv * 0.9
        if zred <= 3.0:
            agelims = (
                [0.0, 7.47712]
                + np.linspace(8.0, np.log10(tbinmax), nbins_sfh - 2).tolist()
                + [np.log10(tuniv)]
            )
        else:
            agelims = np.linspace(amin, np.log10(tbinmax), nbins_sfh).tolist() + [
                np.log10(tuniv)
            ]
            agelims[0] = 0

        agebins = np.array([agelims[:-1], agelims[1:]])
        return agebins.T

    def logsfr_ratios_to_masses(
        logmass=None, logsfr_ratios=None, agebins=None, **extras
    ):
        r"""This converts from an array of log_10(SFR_j / SFR_{j+1}) and a value of
        log10(\Sum_i M_i) to values of M_i.  j=0 is the most recent bin in lookback
        time.
        """
        nbins = agebins.shape[0]
        sratios = 10 ** np.clip(logsfr_ratios, -100, 100)
        dt = 10 ** agebins[:, 1] - 10 ** agebins[:, 0]
        coeffs = np.array(
            [
                (1.0 / np.prod(sratios[:i]))
                * (np.prod(dt[1 : i + 1]) / np.prod(dt[:i]))
                for i in range(nbins)
            ]
        )
        m1 = (10**logmass) / coeffs.sum()

        return m1 * coeffs

    model_params["mass"] = {
        "N": 7,
        "isfree": False,
        "init": 1e6,
        "units": r"M$_\odot$",
        "depends_on": logsfr_ratios_to_masses,
    }

    model_params["agebins"] = {
        "N": 7,
        "isfree": False,
        "init": zred_to_agebins(np.atleast_1d(0.5)),
        "prior": None,
        "depends_on": zred_to_agebins,
    }

    # --- Dust Absorption ---
    model_params["dust_type"] = {
        "N": 1,
        "isfree": False,
        "init": 4,
        "units": "FSPS index",
    }
    model_params["dust1_fraction"] = {
        "N": 1,
        "isfree": True,
        "init": 1.0,
        "prior": priors.FastTruncatedNormal(a=0.0, b=2.0, mu=1.0, sig=0.3),
    }

    model_params["dust2"] = {
        "N": 1,
        "isfree": True,
        "init": 0.0,
        "units": "",
        "prior": priors.FastTruncatedNormal(a=0.0, b=4.0, mu=0.3, sig=1.0),
    }

    def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
        return dust1_fraction * dust2

    model_params["dust1"] = {
        "N": 1,
        "isfree": False,
        "depends_on": to_dust1,
        "init": 0.0,
        "units": "optical depth towards young stars",
        "prior": None,
    }
    model_params["dust_index"] = {
        "N": 1,
        "isfree": True,
        "init": 0.7,
        "units": "",
        "prior": priors.FastUniform(a=-1.0, b=0.2),
    }

    # --- Nebular Emission ---
    model_params["add_neb_emission"] = {"N": 1, "isfree": False, "init": True}
    model_params["add_neb_continuum"] = {"N": 1, "isfree": False, "init": True}
    model_params["gas_logz"] = {
        "N": 1,
        "isfree": True,
        "init": -0.5,
        "units": r"log Z/Z_\odot",
        "prior": priors.FastUniform(a=-2.0, b=0.5),
    }
    model_params["gas_logu"] = {
        "N": 1,
        "isfree": False,
        "init": -1.0,
        "units": r"Q_H/N_H",
        "prior": priors.FastUniform(a=-4, b=-1),
    }

    # --- AGN dust ---
    model_params["add_agn_dust"] = {"N": 1, "isfree": False, "init": True}

    model_params["log_fagn"] = {
        "N": 1,
        "isfree": True,
        "init": -7.0e-5,
        "prior": priors.FastUniform(a=-5.0, b=-4.9),
    }

    def to_fagn(log_fagn=None, **extras):
        return 10**log_fagn

    model_params["fagn"] = {"N": 1, "isfree": False, "init": 0, "depends_on": to_fagn}

    model_params["log_agn_tau"] = {
        "N": 1,
        "isfree": True,
        "init": np.log10(20.0),
        "prior": priors.FastUniform(a=np.log10(15.0), b=np.log10(15.1)),
    }

    def to_agn_tau(log_agn_tau=None, **extras):
        return 10**log_agn_tau

    model_params["agn_tau"] = {
        "N": 1,
        "isfree": False,
        "init": 0,
        "depends_on": to_agn_tau,
    }

    # --- Dust Emission ---
    model_params["duste_qpah"] = {
        "N": 1,
        "isfree": True,
        "init": 2.0,
        "prior": priors.FastTruncatedNormal(a=0.9, b=1.1, mu=2.0, sig=2.0),
    }

    model_params["duste_umin"] = {
        "N": 1,
        "isfree": True,
        "init": 1.0,
        "prior": priors.FastTruncatedNormal(a=0.9, b=1.1, mu=1.0, sig=10.0),
    }

    model_params["log_duste_gamma"] = {
        "N": 1,
        "isfree": True,
        "init": -2.0,
        "prior": priors.FastTruncatedNormal(a=-2.1, b=-1.9, mu=-2.0, sig=1.0),
    }

    def to_duste_gamma(log_duste_gamma=None, **extras):
        return 10**log_duste_gamma

    model_params["duste_gamma"] = {
        "N": 1,
        "isfree": False,
        "init": 0,
        "depends_on": to_duste_gamma,
    }

    # ---- Units ----
    model_params["peraa"] = {"N": 1, "isfree": False, "init": False}

    model_params["mass_units"] = {"N": 1, "isfree": False, "init": "mformed"}

    tparams = {}
    for i in fit_order:
        tparams[i] = model_params[i]
    for i in list(model_params.keys()):
        if i not in fit_order:
            tparams[i] = model_params[i]
    model_params = tparams

    return PolySpecModel(model_params)


def build_sps(zcontinuous=2, compute_vega_mags=False, **extras):
    sps = FastStepBasis(zcontinuous=zcontinuous, compute_vega_mags=compute_vega_mags)
    return sps


def build_noise(**extras):
    return None, None


def scale(mass):
    upper_84 = np.interp(mass, massmet[:, 0], massmet[:, 3])
    lower_16 = np.interp(mass, massmet[:, 0], massmet[:, 2])
    return upper_84 - lower_16


def loc(mass):
    return np.interp(mass, massmet[:, 0], massmet[:, 1])


def draw_thetas(flat=False, zmin=DEFAULT_ZMIN, zmax=DEFAULT_ZMAX):
    if flat:
        zred = priors.FastUniform(a=zmin, b=zmax + ZRED_PRIOR_PAD).sample()
        logmass = priors.FastUniform(a=6.0, b=12.0).sample()
        logzsol = priors.FastUniform(a=-1.98, b=0.19).sample()

        logsfrratio_0 = priors.FastUniform(a=-5.0, b=5.0).sample()
        logsfrratio_1 = priors.FastUniform(a=-5.0, b=5.0).sample()
        logsfrratio_2 = priors.FastUniform(a=-5.0, b=5.0).sample()
        logsfrratio_3 = priors.FastUniform(a=-5.0, b=5.0).sample()
        logsfrratio_4 = priors.FastUniform(a=-5.0, b=5.0).sample()
        logsfrratio_5 = priors.FastUniform(a=-5.0, b=5.0).sample()

        dust2 = priors.FastUniform(a=0.0, b=4.0).sample()
        dust_index = priors.FastUniform(a=-1.0, b=0.4).sample()
        dust1_fraction = priors.FastUniform(a=0.0, b=2.0).sample()
        log_fagn = priors.FastUniform(a=-5.0, b=np.log10(3.0)).sample()
        log_agn_tau = priors.FastUniform(a=np.log10(5.0), b=np.log10(150.0)).sample()
        gas_logz = priors.FastUniform(a=-2.0, b=0.5).sample()
        duste_qpah = priors.FastUniform(a=0.0, b=7.0).sample()
        duste_umin = priors.FastUniform(a=0.1, b=25.0).sample()
        log_duste_gamma = priors.FastUniform(a=-4.0, b=0.0).sample()

    else:
        zred = priors.FastUniform(a=zmin, b=zmax + ZRED_PRIOR_PAD).sample()

        if zred < MASSFUNC_ZMIN:
            logmass = priors.FastUniform(a=6.0, b=12.0).sample()
        else:
            cdf_mass = priors_beta.cdf_mass_func_at_z(zred, np.arange(6,12,0.0001), const_phi=False, bounds=[6.0, 12.0])
            logmass = priors_beta.draw_sample(np.arange(6,12,0.0001), cdf=cdf_mass)
        logzsol = priors.FastTruncatedNormal(
            a=-1.98, b=0.19, mu=loc(logmass), sig=scale(logmass)
        ).sample()

        logsfrratio_0 = priors.FastTruncatedEvenStudentTFreeDeg2(
            hw=5.0, sig=0.3
        ).sample()
        logsfrratio_1 = priors.FastTruncatedEvenStudentTFreeDeg2(
            hw=5.0, sig=0.3
        ).sample()
        logsfrratio_2 = priors.FastTruncatedEvenStudentTFreeDeg2(
            hw=5.0, sig=0.3
        ).sample()
        logsfrratio_3 = priors.FastTruncatedEvenStudentTFreeDeg2(
            hw=5.0, sig=0.3
        ).sample()
        logsfrratio_4 = priors.FastTruncatedEvenStudentTFreeDeg2(
            hw=5.0, sig=0.3
        ).sample()
        logsfrratio_5 = priors.FastTruncatedEvenStudentTFreeDeg2(
            hw=5.0, sig=0.3
        ).sample()

        dust2 = priors.FastTruncatedNormal(a=0.0, b=4.0, mu=0.3, sig=1.0).sample()
        dust_index = priors.FastUniform(a=-1.0, b=0.4).sample()
        dust1_fraction = priors.FastTruncatedNormal(
            a=0.0, b=2.0, mu=1.0, sig=0.3
        ).sample()
        log_fagn = priors.FastUniform(a=-5.0, b=np.log10(3.0)).sample()
        log_agn_tau = priors.FastUniform(a=np.log10(5.0), b=np.log10(150.0)).sample()
        gas_logz = priors.FastUniform(a=-2.0, b=0.5).sample()
        duste_qpah = priors.FastTruncatedNormal(a=0.0, b=7.0, mu=2.0, sig=2.0).sample()
        duste_umin = priors.FastTruncatedNormal(
            a=0.1, b=25.0, mu=1.0, sig=10.0
        ).sample()
        log_duste_gamma = priors.FastTruncatedNormal(
            a=-4.0, b=0.0, mu=-2.0, sig=1.0
        ).sample()

    return np.array(
        [
            zred,
            logmass,
            logzsol,
            logsfrratio_0,
            logsfrratio_1,
            logsfrratio_2,
            logsfrratio_3,
            logsfrratio_4,
            logsfrratio_5,
            dust2,
            dust_index,
            dust1_fraction,
            log_fagn,
            log_agn_tau,
            gas_logz,
            duste_qpah,
            duste_umin,
            log_duste_gamma,
        ]
    )


def _env_bool(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_float(name, default):
    val = os.environ.get(name)
    return default if val is None else float(val)


class TrainSBI(CronJobBase):
    RUN_EVERY_MINS = 3

    schedule = Schedule(run_every_mins=RUN_EVERY_MINS)
    code = "host.SBI.train_SBI.TrainSBI"

    def build_all(self, **kwargs):
        return (
            build_obs(**kwargs),
            build_model(**kwargs),
            build_sps(**kwargs),
            build_noise(**kwargs),
        )

    def do(
        self,
        do_phot=None,
        do_train=None,
        outfile=None,
        zmin=None,
        zmax=None,
        nsamples=None,
        train_files=None,
        merged_file=None,
        model_file=None,
        summary_file=None,
        do_zphot_model=None,
        filter_file=None,
        export_filters_file=None,
        filters=None,
        transmission_dir=None,
    ):
        """Entry point for both `manage.py runcrons` and the command line.

        Anything not passed explicitly falls back to an environment variable
        (DO_PHOT, DO_TRAIN, DO_ZPHOT_MODEL, OUTFILE, ZMIN, ZMAX, SBI_FILTER_FILE) and then to the module defaults,
        so `runcrons` with no environment set behaves as before (train only).
        """
        do_phot = _env_bool("DO_PHOT", False) if do_phot is None else do_phot
        do_train = _env_bool("DO_TRAIN", True) if do_train is None else do_train
        outfile = outfile or os.environ.get("OUTFILE") or DEFAULT_PHOT_FILE
        zmin = _env_float("ZMIN", DEFAULT_ZMIN) if zmin is None else zmin
        zmax = _env_float("ZMAX", DEFAULT_ZMAX) if zmax is None else zmax
        nsamples = DEFAULT_NSAMPLES if nsamples is None else nsamples
        train_files = train_files or DEFAULT_TRAIN_FILES
        do_zphot_model = (
            _env_bool("DO_ZPHOT_MODEL", False) if do_zphot_model is None else do_zphot_model
        )
        # the photo-z model gets its own default output names so it never
        # overwrites the spec-z model (explicit paths always win)
        if do_zphot_model:
            merged_default = DEFAULT_ZPHOT_MERGED_FILE
            model_default = DEFAULT_ZPHOT_MODEL_FILE
            summary_default = DEFAULT_ZPHOT_SUMMARY_FILE
        else:
            merged_default = DEFAULT_MERGED_FILE
            model_default = DEFAULT_MODEL_FILE
            summary_default = DEFAULT_SUMMARY_FILE
        merged_file = merged_default if merged_file is None else merged_file
        model_file = model_file or model_default
        summary_file = summary_file or summary_default

        if zmin < 0 or zmax <= zmin:
            raise ValueError(f"Need 0 <= zmin < zmax (got zmin={zmin}, zmax={zmax})")

        print(
            f"TrainSBI: do_phot={do_phot} do_train={do_train} "
            f"do_zphot_model={do_zphot_model} zmin={zmin} zmax={zmax}",
            flush=True,
        )

        if export_filters_file:
            export_filters(export_filters_file)

        global _filter_file, _filter_names, _transmission_dir, _all_filters
        if filter_file:
            _filter_file = filter_file
        if filters:
            _filter_names = list(filters)
        if transmission_dir:
            _transmission_dir = transmission_dir
        _all_filters = None

        if do_phot:
            self.make_phot(outfile, zmin=zmin, zmax=zmax, nsamples=nsamples)

        if do_train:
            self.train(
                train_files,
                merged_file=merged_file,
                model_file=model_file,
                summary_file=summary_file,
                do_zphot_model=do_zphot_model,
            )

        print("Finished.")

    def make_phot(self, outfile, zmin=DEFAULT_ZMIN, zmax=DEFAULT_ZMAX, nsamples=DEFAULT_NSAMPLES):
        """Simulate noisy photometry for `nsamples` draws and write it to `outfile`."""
        print(f"Simulating {nsamples} SEDs for {zmin} <= z <= {zmax} -> {outfile}", flush=True)
        _ensure_parent_dir(outfile)

        run_params = {"ichunk": 0, "needed_size": nsamples, "zmin": zmin, "zmax": zmax}
        run_params["add_duste"] = True
        run_params["add_igm"] = True
        run_params["add_neb"] = True
        run_params["dynesty"] = False
        run_params["optmization"] = False
        run_params["emcee"] = False

        all_filters = get_all_filters()
        obs, model, sps, noise = self.build_all(**run_params)

        run_params["sps_libraries"] = sps.ssp.libraries
        run_params["param_file"] = __file__

        # get the minimum, maximum magnitudes
        cat_min, cat_max, cat_full = {}, {}, {}
        for f in all_filters:
            mag, snr, snrstd = np.loadtxt(
                f"host/SBI/snrfiles/{f.name}_magvsnr.txt", unpack=True
            )

            cat_min[f.name] = np.min(mag)
            cat_max[f.name] = np.max(mag)
            cat_full[f.name] = (mag, snr, snrstd)

        ### start putting together the synthetic data
        list_thetas = []
        list_mfrac = []
        list_phot = []
        while len(list_phot) < nsamples:
            list_phot_len = len(list_phot)
            if not list_phot_len % 10: print(list_phot_len)
            if not list_phot_len % 3:
                theta = draw_thetas(flat=True, zmin=zmin, zmax=zmax)
            else:
                theta = draw_thetas(flat=False, zmin=zmin, zmax=zmax)

            # call prospector
            # generate the model SED at given theta
            spec, phot, mfrac = model.predict(theta, obs=obs, sps=sps)

            predicted_mags = -2.5 * np.log10(phot)

            flag = True
            for i, f in enumerate(all_filters):
                # probably gonna have some unit issues here
                # expand the magnitude range a bit
                flag &= (predicted_mags[i] >= cat_min[f.name] - 1) & (
                    predicted_mags[i] <= cat_max[f.name] + 2
                )

            # if all phot is within valid range, we can proceed
            if not flag:
                continue

            list_thetas.append(theta)
            list_mfrac.append(mfrac)

            # simulate the noised-up photometry
            list_phot_single = np.array([])
            list_phot_errs_single = np.array([])
            for i, f in enumerate(all_filters):
                snr = np.interp(
                    predicted_mags[i], cat_full[f.name][0], cat_full[f.name][1]
                )
                snrstd = np.interp(
                    predicted_mags[i], cat_full[f.name][0], cat_full[f.name][2]
                )
                snr_sim = np.random.normal(snr,snrstd)
                if snr_sim < 0.010: snr_sim = 0.010
                phot_err = phot[i] / snr_sim
                phot_random = np.random.normal(phot[i], phot_err)
                phot_random_mags = maggies_to_asinh(phot_random)
                phot_err_mags = 2.5 / np.log(10) * phot_err / phot[i]

                list_phot_single = np.append(list_phot_single, [phot_random_mags])
                list_phot_errs_single = np.append(
                    list_phot_errs_single, [phot_err_mags]
                )
#                    import pdb; pdb.set_trace()
            # adding in the redshift here
            list_phot.append(
                np.concatenate(
                    (list_phot_single, list_phot_errs_single, [theta[0]]),
                )
            )
            print(len(list_phot))

        with h5py.File(outfile, "w") as hf_phot:
            hf_phot.create_dataset("wphot", data=obs["phot_wave"])
            hf_phot.create_dataset("phot", data=list_phot)
            hf_phot.create_dataset("mfrac", data=list_mfrac)
            hf_phot.create_dataset("theta", data=list_thetas)
            hf_phot.attrs["zmin"] = zmin
            hf_phot.attrs["zmax"] = zmax
        print(f"Wrote {outfile}", flush=True)

    def train(
        self,
        train_files,
        merged_file=DEFAULT_MERGED_FILE,
        model_file=DEFAULT_MODEL_FILE,
        summary_file=DEFAULT_SUMMARY_FILE,
        do_zphot_model=False,
    ):
        """Combine the simulated-photometry files and train the ANPE.

        Each row of the stored "phot" array is [mags, mag errors, redshift].
        With do_zphot_model the redshift column is dropped from the observed
        data, so the network has to infer zred (theta[0]) from photometry alone.
        The simulated files themselves are the same for both models.
        """
        # expand any wildcards (quoted on the command line so they resolve
        # relative to app/, not the directory the job was submitted from)
        files = []
        for pattern in train_files:
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"No training file matches {pattern!r}")
            files.extend(m for m in matches if m not in files)
        print(f"Training on {len(files)} file(s): {files}", flush=True)

        thetas, phots, mfracs = [], [], []
        list_phot_wave = None
        for fname in files:
            with h5py.File(fname, "r") as data:
                thetas.append(np.array(data["theta"]))
                phots.append(np.array(data["phot"]))
                mfracs.append(np.array(data["mfrac"]))
                wphot = np.array(data["wphot"])
            if list_phot_wave is None:
                list_phot_wave = wphot
            elif not np.allclose(wphot, list_phot_wave):
                raise ValueError(f"{fname} uses a different filter set from {files[0]}")

        list_thetas = np.concatenate(thetas, axis=0)
        list_phot = np.concatenate(phots, axis=0)
        list_mfrac = np.concatenate(mfracs, axis=0)

        if merged_file:
            _ensure_parent_dir(merged_file)
            with h5py.File(merged_file, "w") as hf_phot:
                hf_phot.create_dataset("wphot", data=list_phot_wave)
                hf_phot.create_dataset("phot", data=list_phot)
                hf_phot.create_dataset("mfrac", data=list_mfrac)
                hf_phot.create_dataset("theta", data=list_thetas)
            print(f"Wrote merged training set {merged_file}", flush=True)

        x_train = np.array(list_thetas)
        y_train = np.array(list_phot)
        if do_zphot_model:
            # sanity check: the last observed column should be the true redshift
            if not np.allclose(y_train[:, -1], x_train[:, 0]):
                raise ValueError(
                    "Last column of 'phot' does not match theta[:, 0] (zred); "
                    "refusing to drop it for the photo-z model"
                )
            y_train = y_train[:, :-1]
            print(
                f"Photo-z model: dropped redshift from observed data "
                f"({y_train.shape[1]} observed columns)",
                flush=True,
            )
        # now do the training
        anpe = Inference.SNPE(
            density_estimator=posterior_nn(
                "maf", hidden_features=nhidden, num_transforms=nblocks
            ),
            device=device,
        )
        # because we append_simulations, training set == prior
        anpe.append_simulations(
            torch.as_tensor(x_train.astype(np.float32), device="cpu"),
            torch.as_tensor(y_train.astype(np.float32), device="cpu"),
        )
        p_x_y_estimator = anpe.train()

        # save trained ANPE
        _ensure_parent_dir(model_file)
        torch.save(p_x_y_estimator.state_dict(), model_file)

        # save training summary
        _ensure_parent_dir(summary_file)
        with open(summary_file, "wb") as fout:
            pickle.dump(anpe._summary, fout)
        print(anpe._summary)
        print(f"Wrote {model_file} and {summary_file}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Simulate photometry and/or train the SBI++ model for Blast. "
        "Run from the app/ directory; relative paths are resolved from there."
    )
    parser.add_argument("--do-phot", action="store_true",
                        help="simulate a training set of noisy photometry")
    parser.add_argument("--do-train", action="store_true",
                        help="train the neural posterior estimator")
    parser.add_argument("--do-zphot-model", action="store_true",
                        help="train a photometric-redshift model: redshift is removed "
                        "from the observed data and inferred along with the other "
                        "parameters; changes the default output names to *_zphot*")
    parser.add_argument("--filter-file", default=os.environ.get("SBI_FILTER_FILE"),
                        help="JSON filter list from --export-filters; with it, --do-phot "
                        "does not need Django or the database (default: $SBI_FILTER_FILE)")
    parser.add_argument("--filters", nargs="+", metavar="NAME", default=None,
                        help="filter names, in order, to read from --transmission-dir "
                        "(default: the built-in DEFAULT_FILTERS list, matching the "
                        "database order); 'db' = read the filter list from the "
                        "database; 'auto' = every curve in --transmission-dir except "
                        f"{', '.join(EXCLUDED_FILTERS)}, alphabetically")
    parser.add_argument("--transmission-dir", default=DEFAULT_TRANSMISSION_DIR,
                        help="directory of <filter>.txt transmission curves "
                        "(default: $TRANSMISSION_CURVES_ROOT or %(default)s)")
    parser.add_argument("--export-filters", dest="export_filters_file", metavar="FILE",
                        help="write the database filter list and transmission curves "
                        "to FILE (needs the database), e.g. host/SBI/sbi_filters.json")
    parser.add_argument("--zmin", type=float, default=DEFAULT_ZMIN,
                        help="lower redshift limit (default: %(default)s)")
    parser.add_argument("--zmax", type=float, default=DEFAULT_ZMAX,
                        help="upper redshift limit (default: %(default)s)")
    parser.add_argument("--nsamples", type=int, default=DEFAULT_NSAMPLES,
                        help="simulated SEDs to keep with --do-phot (default: %(default)s)")
    parser.add_argument("--outfile", default=os.environ.get("OUTFILE", DEFAULT_PHOT_FILE),
                        help="HDF5 file written by --do-phot (default: $OUTFILE or %(default)s)")
    parser.add_argument("--train-files", nargs="+", default=DEFAULT_TRAIN_FILES,
                        help="HDF5 files (or quoted wildcards) to train on "
                        "(default: host/SBI/sbi_phot_1.h5 ... sbi_phot_10.h5)")
    parser.add_argument("--merged-file", default=None,
                        help="where to write the combined training set; '' to skip "
                        f"(default: {DEFAULT_MERGED_FILE}, or "
                        f"{DEFAULT_ZPHOT_MERGED_FILE} with --do-zphot-model)")
    parser.add_argument("--model-file", default=None,
                        help=f"trained model output (.pt) (default: {DEFAULT_MODEL_FILE}, "
                        f"or {DEFAULT_ZPHOT_MODEL_FILE} with --do-zphot-model)")
    parser.add_argument("--summary-file", default=None,
                        help=f"training summary output (.p) (default: {DEFAULT_SUMMARY_FILE}, "
                        f"or {DEFAULT_ZPHOT_SUMMARY_FILE} with --do-zphot-model)")
    args = parser.parse_args(argv)

    if not (args.do_phot or args.do_train or args.export_filters_file):
        parser.error("give at least one of --do-phot / --do-train / --export-filters")
    if args.do_zphot_model and not args.do_train:
        parser.error("--do-zphot-model only affects training; add --do-train")
    if args.zmin < 0 or args.zmax <= args.zmin:
        parser.error("need 0 <= --zmin < --zmax")
    if args.filter_file and args.filters:
        parser.error("give --filter-file or --filters, not both")
    if args.filters and len(args.filters) > 1 and {"db", "auto"} & set(args.filters):
        parser.error("'db' and 'auto' can't be combined with filter names")
    return args


if __name__ == "__main__":
    args = parse_args()
    TrainSBI().do(**vars(args))
