# Utils and wrappers for the prospector SED fitting code
import copy
import json
import os
import time

import extinction
import h5py
import numpy as np
import prospect.io.read_results as reader
from astropy.cosmology import WMAP9 as cosmo
from django.conf import settings
from django.db.models import Q
from host import postprocess_prosp as pp
from host.SBI.run_sbi_blast import fit_sbi_pp
from prospect.fitting import fit_model as fit_model_prospect
from prospect.fitting import lnprobfn
from prospect.io import write_results as writer
from prospect.io.write_results import write_h5_header
from prospect.io.write_results import write_obs_to_h5
from prospect.models import priors
from prospect.models import SpecModel
from prospect.models.sedmodel import PolySpecModel
from prospect.models.templates import TemplateLibrary
from prospect.models.transforms import logsfr_ratios_to_sfrs
from prospect.models.transforms import zred_to_agebins
from prospect.sources import CSPSpecBasis
from prospect.sources import FastStepBasis
from prospect.utils.obsutils import fix_obs
from scipy.special import gamma
from scipy.special import gammainc

from .host_utils import get_dust_maps
from .models import AperturePhotometry
from .models import Filter
from .models import hdf5_file_path
from .photometric_calibration import mJy_to_maggies  ##jansky_to_maggies


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


def get_CI(chain):
    chainlen = len(chain)
    chainsort = np.sort(chain)
    return (
        chainsort[int(chainlen * 0.16)],
        chainsort[int(chainlen * 0.50)],
        chainsort[int(chainlen * 0.84)],
    )


# I don't remember where this came from
# somewhere in the prospector docs
def psi_from_sfh(mass, tage, tau):
    return (
        mass
        * (tage / tau**2)
        * np.exp(-tage / tau)
        / (gamma(2) * gammainc(2, tage / tau))
        * 1e-9
    )


def build_obs(transient, aperture_type, use_mag_offset=True):
    """
    This functions is required by prospector and should return
    a dictionary defined by
    https://prospect.readthedocs.io/en/latest/dataformat.html.

    """

    photometry = AperturePhotometry.objects.filter(
        transient=transient, aperture__type__exact=aperture_type
    ).filter(Q(is_validated="true") | Q(is_validated="contamination warning"))

    if not photometry.exists():
        raise ValueError(f"No host photometry of type {aperture_type}")

    if transient.host is None:
        raise ValueError("No host galaxy match")

    z = transient.best_redshift
    if z is not None and z < 0.015 and use_mag_offset:
        mag_offset = cosmo.distmod(z + 0.015).value - cosmo.distmod(z).value
        z += 0.015
    else:
        mag_offset = 0

    filters, flux_maggies, flux_maggies_error = [], [], []

    for filter in Filter.objects.all():
        # if 'PanSTARRS' in filter.name or 'DES' in filter.name: continue
        try:
            datapoint = photometry.get(filter=filter)
        except AperturePhotometry.DoesNotExist:
            # sometimes data just don't exist, we can ignore
            continue
        except AperturePhotometry.MultipleObjectsReturned:
            raise

        if datapoint.flux is None:
            continue

        # correct for MW reddening
        if aperture_type == "global":
            mwebv = transient.host.milkyway_dust_reddening
            if mwebv is None:
                # try once more
                mwebv = get_dust_maps(transient.host.sky_coord)
        elif aperture_type == "local":
            mwebv = transient.milkyway_dust_reddening
            if mwebv is None:
                # try once more
                mwebv = get_dust_maps(transient.sky_coord)
        else:
            raise ValueError(
                f"aperture_type must be 'global' or 'local', currently set to {aperture_type}"
            )
        wave_eff = filter.transmission_curve().wave_effective
        ext_corr = extinction.fitzpatrick99(np.array([wave_eff]), mwebv * 3.1, r_v=3.1)[
            0
        ]
        flux_mwcorr = (
            datapoint.flux * 10 ** (-0.4 * filter.ab_offset) * 10 ** (0.4 * ext_corr)
        )
        # 1% error floor
        fluxerr_mwcorr = np.sqrt(
            (
                datapoint.flux_error
                * 10 ** (-0.4 * filter.ab_offset)
                * 10 ** (0.4 * ext_corr)
            )
            ** 2.0
            + (0.01 * flux_mwcorr) ** 2.0
        )

        # TEST - are low-S/N observations messing up prospector?
        if flux_mwcorr / fluxerr_mwcorr < 3:
            continue

        filters.append(filter.transmission_curve())
        flux_maggies.append(mJy_to_maggies(flux_mwcorr * 10 ** (-0.4 * mag_offset)))
        flux_maggies_error.append(
            mJy_to_maggies(fluxerr_mwcorr * 10 ** (-0.4 * mag_offset))
        )

    obs_data = dict(
        wavelength=None,
        spectrum=None,
        unc=None,
        redshift=z,
        maggies=np.array(flux_maggies),
        maggies_unc=np.array(flux_maggies_error),
        filters=filters,
    )

    return fix_obs(obs_data)


def build_model_nonparam(obs=None, **extras):
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
        "prior": priors.FastUniform(a=0, b=0.2),
    }

    model_params["logmass"] = {
        "N": 1,
        "isfree": True,
        "init": 8.0,
        "units": "Msun",
        "prior": priors.FastUniform(a=7.0, b=12.5),
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
        """This converts from an array of log_10(SFR_j / SFR_{j+1}) and a value of
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


def build_model_nonparam_bkp(obs=None, **extras):
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
        "isfree": False,
        "init": obs["redshift"],
        # "prior": priors.FastUniform(a=0, b=0.2),
    }

    model_params["logmass"] = {
        "N": 1,
        "isfree": True,
        "init": 8.0,
        "units": "Msun",
        "prior": priors.FastUniform(a=7.0, b=12.5),
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
        """This converts from an array of log_10(SFR_j / SFR_{j+1}) and a value of
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
        "prior": priors.FastUniform(a=-1.0, b=0.4),
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


def build_model(observations):
    """
    Construct all model components
    """

    # model_params = TemplateLibrary["parametric_sfh"]
    # model_params.update(TemplateLibrary["nebular"])
    # model_params["zred"]["init"] = observations["redshift"]
    # model = SpecModel(model_params)
    model = build_model_nonparam(observations)

    # new SPS model
    # sps = CSPSpecBasis(zcontinuous=1)
    sps = FastStepBasis(zcontinuous=2, compute_vega_mags=False)
    noise_model = (None, None)

    return {"model": model, "sps": sps, "noise_model": noise_model}


def fit_model(
    observations, model_components, fitting_kwargs, sbipp=False, fit_type="global"
):
    """Fit the model"""

    if sbipp:
        output, errflag = fit_sbi_pp(observations, fit_type=fit_type)
    else:
        output = fit_model_prospect(
            observations,
            model_components["model"],
            model_components["sps"],
            optimize=False,
            dynesty=True,
            lnprobfn=lnprobfn,
            noise=model_components["noise_model"],
            **fitting_kwargs,
        )
        errflag = 0

    return output, errflag


def prospector_result_to_blast(
    transient,
    aperture,
    prospector_output,
    model_components,
    observations,
    sed_output_root=settings.SED_OUTPUT_ROOT,
    parametric_sfh=False,
    sbipp=False,
):

    # write the results
    hdf5_file_name = (
        f"{sed_output_root}/{transient.name}/{transient.name}_{aperture.type}.h5"
    )
    hdf5_file = hdf5_file_name

    if not os.path.exists(f"{sed_output_root}/{transient.name}"):
        os.makedirs(f"{sed_output_root}/{transient.name}/")
    if os.path.exists(hdf5_file):
        # prospector won't overwrite, which causes problems
        os.remove(hdf5_file)

    if sbipp:

        hf = h5py.File(hdf5_file_name, "w")

        sdat = hf.create_group("sampling")
        sdat.create_dataset("chain", data=prospector_output["sampling"][0]["samples"])
        sdat.attrs["theta_labels"] = json.dumps(
            list(model_components["model"].theta_labels())
        )

        # High level parameter and version info
        write_h5_header(hf, {}, model_components["model"])

        # ----------------------
        # Observational data
        write_obs_to_h5(hf, observations)
        hf.flush()
    else:
        writer.write_hdf5(
            hdf5_file,
            {},
            model_components["model"],
            observations,
            prospector_output["sampling"][0],
            None,
            sps=model_components["sps"],
            tsample=prospector_output["sampling"][1],
            toptimize=0.0,
        )

    # load up the hdf5 file to get the results
    resultpars, obs, _ = reader.results_from(hdf5_file, dangerous=False)

    # if sbipp:
    #     theta_max = resultpars["chain"][100] #np.mean(resultpars["chain"], axis=0) # + np.std(resultpars["chain"], axis=0)
    # else:
    #    imax = np.argmax(resultpars["lnprobability"])
    #    csz = resultpars["chain"].shape
    #    if resultpars["chain"].ndim > 2:
    # emcee
    #        i, j = np.unravel_index(imax, resultpars["lnprobability"].shape)
    #        theta_max = resultpars["chain"][i, j, :].copy()
    #    else:
    # dynesty
    #        theta_max = resultpars["chain"][imax, :].copy()

    model_init = copy.deepcopy(model_components["model"])
    tstart = time.time()
    ### take the mean of 50 random samples to get the "best fit" model
    # best_phot_store = np.array([])
    for i in range(50):
        theta = resultpars["chain"][
            np.random.choice(np.arange(np.shape(resultpars["chain"])[0])), :
        ]
        theta[0] = observations["redshift"]
        if i == 0:
            best_spec, best_phot, mfrac = model_components["model"].predict(
                theta, obs=observations, sps=model_components["sps"]
            )

            theta[1] -= np.log10(mfrac)
            ### we need our mass to be conv to total \
            ###    formed mass for the model to be accurate
            best_spec, best_phot, mfrac = model_init.predict(
                theta, obs=observations, sps=model_components["sps"]
            )
            best_phot_store = best_phot[:]
            best_spec_store = best_spec[:]
        else:
            best_spec_single, best_phot_single, mfrac_single = model_components[
                "model"
            ].predict(theta, obs=observations, sps=model_components["sps"])
            theta[1] -= np.log10(mfrac)
            ### we need our mass to be conv to total \
            ###    formed mass for the model to be accurate
            best_spec_single, best_phot_single, mfrac_single = model_init.predict(
                theta, obs=observations, sps=model_components["sps"]
            )

            # iteratively update the mean
            best_spec = (best_spec * i + best_spec_single) / (i + 1)
            best_phot = (best_phot * i + best_phot_single) / (i + 1)
            mfrac = (mfrac * i + mfrac_single) / (i + 1)
            best_phot_store = np.vstack([best_phot_store, best_phot_single])
            best_spec_store = np.vstack([best_spec_store, best_spec_single])
    best_phot = np.median(best_phot_store, axis=0)
    best_spec = np.median(best_spec_store, axis=0)
    phot_16, phot_84 = np.percentile(best_phot_store, [16, 84], axis=0)
    spec_16, spec_84 = np.percentile(best_spec_store, [16, 84], axis=0)
    tfin = time.time()
    print(f"sampling chains to get best-fit model took {tfin-tstart:.0f} seconds")

    if not parametric_sfh:
        use_weights = not sbipp

        pp.run_all(
            hdf5_file_name,
            hdf5_file_name.replace(".h5", "_chain.npz"),
            hdf5_file_name.replace(".h5", "_perc.npz"),
            model_components["model"].init_config["zred"]["init"],
            prior="p-alpha",
            mod_fsps=model_components["model"],
            sps=model_components["sps"],
            percents=[15.9, 50, 84.1],
            use_weights=use_weights,
            sbipp=sbipp,
            obs=observations,
        )

        percentiles = np.load(
            hdf5_file_name.replace(".h5", "_perc.npz"), allow_pickle=True
        )
        perc = np.atleast_1d(percentiles["percentiles"])[0]

    logmass16, logmass50, logmass84 = perc["stellar_mass"]
    age16, age50, age84 = perc["mwa"]
    logsfr16, logsfr50, logsfr84 = np.log10(perc["sfr"][0])
    logssfr16, logssfr50, logssfr84 = np.log10(perc["ssfr"][0])

    if parametric_sfh:
        tau = resultpars["chain"][
            ..., np.where(np.array(resultpars["theta_labels"]) == "tau")[0][0]
        ]
        tau16, tau50, tau84 = get_CI(tau)

    prosp_results = {
        "transient": transient,
        "aperture": aperture,
        "posterior": hdf5_file,
        "chains_file": hdf5_file_name.replace(".h5", "_chain.npz"),
        "percentiles_file": hdf5_file_name.replace(".h5", "_perc.npz"),
        "model_file": hdf5_file_name.replace(".h5", "_modeldata.npz"),
        "log_mass_16": logmass16,
        "log_mass_50": logmass50,
        "log_mass_84": logmass84,
        "log_sfr_16": logsfr16,
        "log_sfr_50": logsfr50,
        "log_sfr_84": logsfr84,
        "log_ssfr_16": logssfr16,
        "log_ssfr_50": logssfr50,
        "log_ssfr_84": logssfr84,
        "log_age_16": age16,
        "log_age_50": age50,
        "log_age_84": age84,
        "mass_surviving_ratio": mfrac,
    }
    if parametric_sfh:
        prosp_results["log_tau_16"] = (tau16,)
        prosp_results["log_tau_50"] = (tau50,)
        prosp_results["log_tau_84"] = (tau84,)

    np.savez(
        hdf5_file_name.replace(".h5", "_modeldata.npz"),
        rest_wavelength=model_components["sps"].wavelengths,
        spec=best_spec,
        phot=best_phot,
        spec_16=spec_16,
        spec_84=spec_84,
        phot_16=phot_16,
        phot_84=phot_84,
    )

    return prosp_results
