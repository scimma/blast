# $ conda activate sbi_env
import copy
import os
import signal
import sys
import time
import warnings

import pylab as plt
from astropy.stats import sigma_clipped_stats
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

# import importlib
# importlib.import_module("host.prospector.build_model")
# importlib.import_module("host.prospector.build_obs")

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from operator import itemgetter
import numpy as np
from numpy.random import normal, uniform
from scipy import stats
from scipy.interpolate import interp1d

# torch
import torch

torch.set_num_threads(1)
# torch.multiprocessing.set_start_method('forkserver',force=True)
# spawn
import torch.nn as nn
import torch.nn.functional as F
from sbi import utils as Ut
from sbi import inference as Inference

from host.models import AperturePhotometry, Transient, Filter
from django.db.models import Q
from host.host_utils import get_dust_maps
import extinction
from prospect.utils.obsutils import fix_obs
from host.photometric_calibration import mJy_to_maggies
from astropy.cosmology import WMAP9 as cosmo

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
all_filters = Filter.objects.filter(~Q(name="DES_i") & ~Q(name="DES_Y"))


def build_all(**kwargs):
    return (
        build_obs(**kwargs),
        build_model(**kwargs),
        build_sps(**kwargs),
        build_noise(**kwargs),
    )


def build_obs(**extras):  ##transient, aperture_type):
    """
    This functions is required by prospector and should return
    a dictionary defined by
    https://prospect.readthedocs.io/en/latest/dataformat.html.

    """
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


def build_model(obs=None, **extras):
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


def build_sps(zcontinuous=2, compute_vega_mags=False, **extras):
    sps = FastStepBasis(zcontinuous=zcontinuous, compute_vega_mags=compute_vega_mags)
    return sps


def build_noise(**extras):
    return None, None


# prior
def prior_from_train(ll_or_ul, x_train):
    """We will only need the lower & upper limits to be passed to sbi as 'priors'
    Note that I have not checked how this affects the sbi functions that are not used in the script.
    Here we simply read in the bounds from the training set
    """

    assert ll_or_ul in ["ll", "ul"]

    if ll_or_ul == "ll":
        res = []
        for i in range(x_train.shape[1]):
            res.append(np.min(x_train.T[i]))
    else:
        res = []
        for i in range(x_train.shape[1]):
            res.append(np.max(x_train.T[i]))

    return res


def toy_noise(flux, meds_sigs, stds_sigs, verbose=False, **extra):
    """toy noise; must be the same as the noise model used when generating the training set
    Here we use assume Gaussian noises
    flux: photometry
    meds_sigs: median of the magnitude bin
    stds_sigs: 1 standard deviation
    """
    return flux, meds_sigs(flux), np.clip(stds_sigs(flux), a_min=0.001, a_max=None)


# the following functions are used to set the max time spent per object
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def absdiff(mags, obsphot, obsphot_unc):
    """abs difference in photometry"""

    return np.abs(mags - obsphot)


def chi2dof(mags, obsphot, obsphot_unc, individual=False):
    """reduced chi^2"""

    if individual:
        return ((mags - obsphot) / obsphot_unc) ** 2
    else:
        chi2 = np.nansum(((mags - obsphot) / obsphot_unc) ** 2, axis=1)
        return chi2 / np.sum(np.isfinite(obsphot))


def chidof(mags, obsphot, obsphot_unc, individual=False):
    """reduced chi^2"""

    if individual:
        return (mags - obsphot) / obsphot_unc
    else:
        chi = np.nansum((mags - obsphot) / obsphot_unc, axis=1)
        return chi / np.sum(np.isfinite(obsphot))


def gauss_approx_missingband(obs, run_params, sbi_params, max_neighbors=200):
    """nearest neighbor approximation of missing bands;
    see sec. 4.1.2 for details
    """
    use_res = False
    y_train = sbi_params["y_train"]

    y_obs = np.copy(obs["mags_sbi"])
    sig_obs = np.copy(obs["mags_unc_sbi"])
    invalid_mask = np.copy(obs["missing_mask"])
    observed = np.concatenate([y_obs, sig_obs, [obs["redshift"]]])
    y_obs_valid_only = y_obs[~invalid_mask]
    valid_idx = np.where(~invalid_mask)[0]
    not_valid_idx = np.where(invalid_mask)[0]

    look_in_training = y_train[:, valid_idx]
    chi2_nei = chi2dof(
        mags=look_in_training, obsphot=y_obs[valid_idx], obsphot_unc=sig_obs[valid_idx]
    )

    _chi2_thres = run_params["ini_chi2"] * 1
    cnt = 0
    use_res = True
    while _chi2_thres <= run_params["max_chi2"]:
        idx_chi2_selected = np.where(chi2_nei <= _chi2_thres)[0]
        if len(idx_chi2_selected) >= 30:
            break
        else:
            _chi2_thres += 5
        cnt += 1

    if _chi2_thres > run_params["max_chi2"]:
        use_res = False
        chi2_selected = y_train[:, valid_idx]
        chi2_selected = chi2_selected[:max_neighbors]
        guess_ndata = y_train[:, not_valid_idx]
        guess_ndata = guess_ndata[:max_neighbors]
        idx_chi2_selected = np.argsort(chi2_nei)[0:max_neighbors]
        if run_params["verbose"]:
            print("Failed to find sufficient number of nearest neighbors!")
            print(
                "_chi2_thres {} > max_chi2 {}".format(
                    _chi2_thres, run_params["max_chi2"]
                ),
                len(guess_ndata),
            )
    else:
        chi2_selected = y_train[:, valid_idx][idx_chi2_selected]
        # get distribution of the missing band
        guess_ndata = y_train[:, not_valid_idx][idx_chi2_selected]
    dists = np.linalg.norm(y_obs_valid_only - chi2_selected, axis=1)
    neighs_weights = 1 / dists

    kdes = []
    for i in range(guess_ndata.shape[1]):
        _kde = stats.gaussian_kde(guess_ndata.T[i], 0.2, weights=neighs_weights)
        kdes.append(_kde)

    return kdes, use_res, idx_chi2_selected


def sbi_missingband(obs, run_params, sbi_params, seconditer=False):
    """used when observations have missing data;
    see sec. 4.1.2 of for details
    """

    signal.signal(signal.SIGALRM, timeout_handler)

    if run_params["verbose"]:
        print("sbi missing bands")
    # hack!
    sps = build_sps()
    ave_theta = []

    max_neighbors = 200
    hatp_x_y = sbi_params["hatp_x_y"]
    y_train = sbi_params["y_train"]
    y_obs = np.copy(obs["mags_sbi"])
    sig_obs = np.copy(obs["mags_unc_sbi"])
    invalid_mask = np.copy(obs["missing_mask"])
    observed = np.concatenate([y_obs, sig_obs, [obs["redshift"]]])
    y_obs_valid_only = y_obs[~invalid_mask]
    valid_idx = np.where(~invalid_mask)[0]
    not_valid_idx = np.where(invalid_mask)[0]
    st = time.time()

    # ------------------------------------------------
    # nearest neighbor approximation of missing bands;
    # see sec. 4.1 for details
    look_in_training = y_train[:, valid_idx]
    chi2_nei = chi2dof(
        mags=look_in_training, obsphot=y_obs[valid_idx], obsphot_unc=sig_obs[valid_idx]
    )
    chi_nei = chidof(
        mags=look_in_training, obsphot=y_obs[valid_idx], obsphot_unc=sig_obs[valid_idx]
    )

    _chi2_thres = run_params["ini_chi2"] * 1
    cnt = 0
    use_res = True
    while _chi2_thres <= run_params["max_chi2"]:
        # idx_chi2_selected = np.where(chi2_nei[redshift_idx] <= _chi2_thres)[0]
        # if len(idx_chi2_selected) >= 100 and np.abs(np.median(chi_nei[redshift_idx][idx_chi2_selected])) < 0.25:

        # let's not allow any matches with giant overall offsets
        idx_chi2_selected = np.where(chi2_nei <= _chi2_thres)[
            0
        ]  # & (np.abs(chi_nei) < 0.25))[0]
        if (
            len(idx_chi2_selected) >= max_neighbors
        ):  # and np.abs(np.median(chi_nei[idx_chi2_selected])) < 0.25:
            break
        else:
            _chi2_thres += 5
        cnt += 1

    if _chi2_thres > run_params["max_chi2"]:
        use_res = False
        chi2_selected = y_train[:, valid_idx]
        # chi2_selected = chi2_selected[redshift_idx][:100]
        chi2_selected = chi2_selected[:max_neighbors]
        guess_ndata = y_train[:, not_valid_idx]
        guess_ndata = guess_ndata[:max_neighbors]

        # if not seconditer:
        #    idx_chi2_selected = np.argsort(chi2_nei)
        #    diffs = absdiff(
        #        mags=look_in_training,
        #        obsphot=y_obs[valid_idx],
        #        obsphot_unc=sig_obs[valid_idx],
        #    )
        #    diffs_best = np.sum(diffs[idx_chi2_selected[0:100]], axis=0)
        #    worst_band = np.where(diffs_best == np.max(diffs_best))[0]
        #    obs["missing_mask"][worst_band] = True
        #    print("Failed to find sufficient number of nearest neighbors!")
        #    print(f"Trying again after dropping band {worst_band[0]}")
        #    obs["sbi_flag"] = "chi2 fail"
        #    return obs

        # idx_chi2_selected = np.argsort(chi2_nei[redshift_idx])[0:100]
        idx_chi2_selected = np.argsort(chi2_nei)[0:max_neighbors]

        if run_params["verbose"]:
            print("Failed to find sufficient number of nearest neighbors!")
            print(
                "_chi2_thres {} > max_chi2 {}".format(
                    _chi2_thres, run_params["max_chi2"]
                ),
                len(guess_ndata),
            )

    ### sometimes one bad photometry point messes up everything
    # std_out,iBad = np.array([]),np.array([],dtype=int)
    # for i in idx_chi2_selected[0:100]:
    #    std_out = np.append(std_out,[sigma_clipped_stats(y_obs[valid_idx]-y_train[redshift_idx][i][0:22][valid_idx[0:22]])[2]])
    #    iBad = np.append(iBad,np.where(np.abs(y_obs[valid_idx]-y_train[redshift_idx][i][0:22][valid_idx[0:22]]) > 3*std_out[-1])[0])
    # hack!
    # if len(iBad):
    #    iBad,counts = np.unique(iBad,return_counts=True)
    #    iBad = iBad[counts >= 10]; counts = counts[counts >= 10]
    #    iBad = iBad[np.argsort(counts)[::-1]][0:2] ## remove max two outliers, choose the ones that are most commonly outliers
    #    invalid_mask[valid_idx[iBad]] = True
    #    y_obs_valid_only = y_obs[~invalid_mask]
    #    valid_idx = np.where(~invalid_mask)[0]
    #    not_valid_idx = np.where(invalid_mask)[0]

    # chi2_selected = y_train[:, valid_idx][redshift_idx][idx_chi2_selected]
    # get distribution of the missing band
    # guess_ndata = y_train[:, not_valid_idx][redshift_idx][idx_chi2_selected]
    chi2_selected = y_train[:, valid_idx][idx_chi2_selected]
    # get distribution of the missing band

    guess_ndata = y_train[:, not_valid_idx][idx_chi2_selected]

    dists = np.linalg.norm(y_obs_valid_only - chi2_selected, axis=1)
    neighs_weights = 1 / dists

    kdes = []
    for i in range(guess_ndata.shape[1]):
        kde = stats.gaussian_kde(guess_ndata.T[i], 0.2, weights=neighs_weights)
        kdes.append(kde)
    # import pdb; pdb.set_trace()
    # ------------------------------------------------

    nbands = y_train.shape[1] // 2  # total number of bands
    not_valid_idx_unc = not_valid_idx + nbands

    all_x = []
    cnt = 0
    cnt_timeout = 0
    timeout_flag = False
    # ------------------------------------------------
    # draw monte carlo samples from the nearest neighbor approximation
    # later we will average over the monte-carlo posterior samples to attain the final posterior estimation
    # hack!
    while cnt < run_params["nmc"]:
        ### hack!
        signal.alarm(
            0
        )  # run_params["tmax_per_obj"])  # max time spent on one object in sec
        try:
            x = np.copy(observed)
            # D. Jones edit
            # idx_neighbor = np.random.choice(range(len(guess_ndata.T[0])))
            # while abs(y_train[idx_chi2_selected][idx_neighbor][-1] - observed[-1]) > 0.05:
            # idx_neighbor = np.random.choice(range(len(guess_ndata.T[0])))

            for j, idx in enumerate(not_valid_idx):  ##range(len(not_valid_idx)):
                x[not_valid_idx[j]] = np.random.choice(
                    guess_ndata.T[j]
                )  # np.clip(kdes[j].resample(size=1),0,23) ### I think these crazy non-detections are destroying the fit?
                # hack!
                # x[not_valid_idx[j]] = kdes[j].resample(size=1) ##np.clip(kdes[j].resample(size=1),0,23) ### I think these crazy non-detections are destroying the fit?
                # x[22:][not_valid_idx[j]] = y_train[idx_chi2_selected][idx_neighbor][22:][not_valid_idx[j]]

                # let's just randomly sample the neighbors instead of unpredictable toy noise model
                x[22:][not_valid_idx[j]] = y_train[idx_chi2_selected][
                    np.random.choice(range(len(guess_ndata.T[j])))
                ][22:][not_valid_idx[j]]
                # x[not_valid_idx_unc[j]] = toy_noise(
                #    flux=x[not_valid_idx[j]],
                #    meds_sigs=sbi_params["toynoise_meds_sigs"][idx],
                #    stds_sigs=sbi_params["toynoise_stds_sigs"][idx],
                #    verbose=run_params["verbose"],
                # )[1]

            # the noise model in the training isn't quite right
            # Pan-STARRS in particular seems a little off
            # we'll have to re-train at some point, but for now just pull
            # uncertainties from the training sample
            for idx, fname in zip(valid_idx, obs["filternames"][valid_idx]):
                # if 'PanSTARRS' in fname or '2MASS' in fname or 'SDSS' in fname or 'DES' in fname:
                chc = np.random.choice(range(len(y_train[idx_chi2_selected])))
                # hack!
                # x[22:-1][idx] = y_train[idx_chi2_selected][0][22:-1][idx]
                x[22:-1][idx] = y_train[idx_chi2_selected][chc][22:-1][idx]
                # x[22:-1][idx] = y_train[idx_chi2_selected][idx_neighbor][22:-1][idx]

            # import pdb; pdb.set_trace()
            # x = y_train[idx_chi2_selected][2150] ###guess_ndata.T[:,idx_neighbor]

            all_x.append(x)

            # ixtest = np.where(chi2_nei <= 100)[0]
            # ixtest = np.where(chi2_nei == np.min(chi2_nei))[0]
            # plt.errorbar(obs['wavelengths'],obs['mags'],obs['mags_unc'],fmt='.',zorder=100)
            # plt.errorbar(obs['wavelengths'],obs['mags_sbi'],obs['mags_unc_sbi'],fmt='.')
            # plt.errorbar(obs['wavelengths'],x[0:22],yerr=x[22:-1],fmt='.')
            ##plt.errorbar(obs['wavelengths'],y_train[ixtest[4]][:22],yerr=y_train[ixtest[4]][22:-1],fmt='.')
            # plt.errorbar(obs['wavelengths'],y_train[idx_chi2_selected[30]][:22],yerr=y_train[idx_chi2_selected[30]][22:-1],fmt='.',zorder=101)
            # plt.ylim([21,16])
            # plt.savefig('tmp.png',dpi=200)
            # x = y_train[idx_chi2_selected[30]]
            # x[1] += np.log10(0.7290672783420541)
            # import pdb; pdb.set_trace()

            # if we can't get one posterior sample in one second, we should move along
            # to the next MC sample
            do_continue = False
            for tmax, npost in zip(
                [1, run_params["tmax_per_iter"]], [1, run_params["nposterior"]]
            ):
                signal.alarm(tmax)  # max time spent on one object in sec
                try:
                    noiseless_theta = hatp_x_y.sample(
                        (npost,),
                        x=torch.as_tensor(x.astype(np.float32)).to(device),
                        show_progress_bars=False,
                    )
                except TimeoutException:
                    signal.alarm(0)
                    do_continue = True
                    break

            if do_continue:
                continue

            signal.alarm(0)

            noiseless_theta = noiseless_theta.detach().numpy()
            # if np.median(noiseless_theta[:,1]) > 9:

            ave_theta.append(noiseless_theta)

            ### debug!
            # transient = Transient.objects.get(name='PS15bwh')
            #### all these lines
            if not "hi":
                print("getting photometry")
                observations = build_obs()  # transient, "global")
                model_components = build_model(obs)
                model_mfrac = copy.deepcopy(model_components)  # ["model"])
                # sps = build_sps()
                for i in range(1):
                    print(i)
                    if i == 0:
                        (
                            best_spec,
                            best_phot,
                            mfrac,
                        ) = model_components.predict(  # ["model"]
                            noiseless_theta[i, :],
                            obs=observations,
                            sps=sps,  # model_components["sps"]
                        )
                        noiseless_theta[i, 1] -= np.log10(mfrac)
                        (
                            best_spec,
                            best_phot,
                            mfrac,
                        ) = model_components.predict(  # ["model"]
                            noiseless_theta[i, :],
                            obs=observations,
                            sps=sps,  # model_components["sps"]
                        )
                    else:
                        (
                            best_spec,
                            best_phot_s,
                            mfrac,
                        ) = model_components.predict(  # ["model"]
                            noiseless_theta[i, :],
                            obs=observations,
                            sps=sps,  # model_components["sps"]
                        )
                        noiseless_theta[i, 1] -= np.log10(mfrac)
                        (
                            best_spec,
                            best_phot,
                            mfrac,
                        ) = model_components.predict(  # ["model"]
                            noiseless_theta[i, :],
                            obs=observations,
                            sps=sps,  # model_components["sps"]
                        )
                        best_phot = np.vstack([best_phot, best_phot_s])
                print(-2.5 * np.log10(best_phot))  # np.median(best_phot,axis=0)))
                import pdb

                pdb.set_trace()
            # avg_theta = np.median(noiseless_theta,axis=0)
            # avg_theta[1] -= np.log10(mfrac)
            # best_spec2, best_phot2, mfrac2 = model_mfrac.predict(
            #    avg_theta, obs=observations, sps=build_sps()
            # )

            # theta_test = np.array([ 0.12240195,  8.47703558, -1.34368912, -3.72383187, -2.20004498,
            #                   3.03045638,  1.43847368,  2.05204451,  0.95743611,  0.55960846,
            #                   0.22913056,  1.28790968, -2.26064742,  1.67065051, -0.15686861,
            #                   1.51209086,  3.13108879, -0.39720158])
            # theta_test = np.array([ 0.08959557,  9.91042752, -0.16129237, -0.05564067, -0.47920846, 0.04121112,  0.03795635,  0.49733082,  0.02108837,  0.42442192, 0.17738165,  0.52207115, -0.82690121,  1.12715198, -1.3099881, 1.17192278, 10.65062175, -2.48147237])
            # obs, model, sps, noise = build_all(**run_params)
            # best_spec, best_phot, mfrac = model.predict(theta_test, obs=observations, sps=build_sps())

            #    age_interp, allsfhs_interp, allMWA, allsfrs = getSFH(
            # chain, theta_index=theta_index, rtn_chains=True, zred=zred
            # )

            ### (Pdb) sbi_params['theta_train'][idx_chi2_selected][0]
            # array([ 0.11929193, 10.39623445, -1.32188065,  0.18913014,  0.16209094,
            #         0.06441716, -0.19440235,  0.11935996,  0.37152376,  0.34494525,
            #        -0.42501956,  0.51024255, -2.06271797,  1.37400889, -1.21739342,
            #         3.32965967,  1.28220919, -1.79931691])
            # (Pdb) noiseless_theta[0]
            # array([ 0.11942666, 10.399073  , -1.2320234 ,  0.11172969,  0.34717456,
            #         0.3892678 , -0.6431147 ,  0.24748906, -0.5234739 ,  0.27822632,
            #        -0.52583283,  1.187651  , -1.2531726 ,  1.9812987 , -1.2108788 ,
            #         2.0470366 ,  2.0370784 , -2.9912286 ], dtype=float32)

            # import pdb; pdb.set_trace()

            cnt += 1
            # print(x)
            if run_params["verbose"]:
                if cnt % 10 == 0:
                    print("mc samples:", cnt)

        except TimeoutException:
            cnt_timeout += 1
        else:
            signal.alarm(0)

        # set max time
        et = time.time()
        elapsed_time = et - st  # in secs
        if elapsed_time / 60 > run_params["tmax_all"] or (
            cnt < run_params["nmc"] / 10
            and elapsed_time / 60 * 10 > run_params["tmax_all"]
        ):
            timeout_flag = True
            use_res = False
            break
    # ------------------------------------------------

    all_x = np.array(all_x)
    # import pdb; pdb.set_trace()
    all_x_flux = all_x.T[:nbands]
    all_x_unc = all_x.T[nbands:]
    y_guess = np.concatenate(
        [np.median(all_x_flux, axis=1), np.median(all_x_unc, axis=1), [obs["redshift"]]]
    )

    return ave_theta, y_guess, use_res, timeout_flag, cnt


def lim_of_noisy_guass(obs, run_params, sbi_params):
    """restrict the range over which we monte carlo the noise based on similar SEDs in the training set;
    see sec. 4.1.1 for details
    """

    use_res = 1

    y_train = sbi_params["y_train"]

    y_obs = np.copy(obs["mags_sbi"])
    sig_obs = np.copy(obs["mags_unc_sbi"])
    observed = np.concatenate([y_obs, sig_obs, [obs["redshift"]]])
    noisy_mask = np.copy(obs["noisy_mask"])

    noisy_idx = np.where(noisy_mask == True)[0]
    not_noisy_idx = np.where(noisy_mask == False)[0]

    look_in_training = y_train[:, noisy_idx]
    chi2_nei = chi2dof(
        mags=look_in_training, obsphot=y_obs[noisy_idx], obsphot_unc=sig_obs[noisy_idx]
    )

    _chi2_thres = run_params["ini_chi2"] * 1
    while True:
        idx_chi2_selected = np.where(chi2_nei <= _chi2_thres)[0]
        if len(idx_chi2_selected) >= 10:
            chi2_nei_selected = y_train[idx_chi2_selected]
            chi2_nei_selected = np.squeeze(chi2_nei_selected[:, noisy_idx])
            lims = [
                np.atleast_1d(np.min(chi2_nei_selected, axis=0)),
                np.atleast_1d(np.max(chi2_nei_selected, axis=0)),
            ]
            if np.all((lims[0] - y_obs[noisy_idx]) < 0) and np.all(
                (lims[1] - y_obs[noisy_idx]) > 0
            ):
                break
        _chi2_thres += 5
        if _chi2_thres > run_params["max_chi2"]:
            if run_params["verbose"]:
                print("Failed to find sufficient number of nearest neighbors!")
                print(
                    "Clipping the Gaussian, from which we MC noise, to be within the min & max of the magnitude at that band in the training set"
                )
            use_res = 0
            # choose the args for clipping norm by the min & max of the magnitude at that band in the training set
            lims = np.array(
                [
                    np.atleast_1d(np.min(y_train[:, noisy_idx], axis=0)),
                    np.atleast_1d(np.max(y_train[:, noisy_idx], axis=0)),
                ]
            )
            break

    return lims, use_res


def sbi_mcnoise(obs, run_params, sbi_params, max_neighbors=200):
    """used when observations have out-of-distribution uncertainties;
    see sec. 4.1.1 for details
    """
    signal.signal(signal.SIGALRM, timeout_handler)

    if run_params["verbose"]:
        print("sbi mc noise")

    ave_theta = []

    hatp_x_y = sbi_params["hatp_x_y"]
    y_train = sbi_params["y_train"]

    y_obs = np.copy(obs["mags_sbi"])
    sig_obs = np.copy(obs["mags_unc_sbi"])
    observed = np.concatenate([y_obs, sig_obs, [obs["redshift"]]])
    nbands = y_train.shape[1] // 2  # total number of bands

    noisy_mask = np.copy(obs["noisy_mask"])
    noisy_idx = np.where(noisy_mask == True)[0]
    not_noisy_idx = np.where(noisy_mask == False)[0]

    # start time
    st = time.time()

    lims, use_res = lim_of_noisy_guass(
        obs=obs, run_params=run_params, sbi_params=sbi_params
    )
    loc = y_obs[noisy_idx]
    scale = sig_obs[noisy_idx]

    ### temporary for getting errors, because error model not good enough
    chi2_nei = chi2dof(mags=y_train[:, :22], obsphot=y_obs, obsphot_unc=sig_obs)

    _chi2_thres = run_params["ini_chi2"] * 1
    while _chi2_thres <= run_params["max_chi2"]:
        idx_chi2_selected = np.where(chi2_nei <= _chi2_thres)[0]
        if len(idx_chi2_selected) >= 30:
            break
        else:
            _chi2_thres += 5

    if _chi2_thres > run_params["max_chi2"]:
        chi2_selected = y_train[:]
        idx_chi2_selected = np.argsort(chi2_nei)[0:max_neighbors]
        if run_params["verbose"]:
            print("Failed to find sufficient number of nearest neighbors!")
    else:
        chi2_selected = y_train[idx_chi2_selected]

    cnt = 0
    cnt_timeout = 0
    timeout_flag = False
    # ------------------------------------------------
    # draw monte carlo samples from a norm dist centered at x_obs and 1 sigma = 1 sigma uncertainty associated with x_obs
    # later we will average over the those "noisy" posterior samples to attain the final posterior estimation
    while cnt < run_params["nmc"]:
        samp_y_guess = np.copy(observed)
        samp_y_guess[noisy_idx] = stats.norm.rvs(loc=loc, scale=scale)
        # ensure positive uncertainties
        _nnflag = True
        for ii, this_noisy_flux in enumerate(samp_y_guess[noisy_idx]):
            # print(lims[0][ii], lims[1][ii])
            if this_noisy_flux > lims[0][ii] and this_noisy_flux < lims[1][ii]:
                _nnflag &= True
            else:
                _nnflag &= False

            if _nnflag:
                samp_y_guess[noisy_idx + nbands] = toy_noise(
                    flux=samp_y_guess[noisy_idx],
                    meds_sigs=sbi_params["toynoise_meds_sigs"][ii],
                    stds_sigs=sbi_params["toynoise_stds_sigs"][ii],
                    verbose=run_params["verbose"],
                )[1]
                # signal.alarm(run_params["tmax_per_obj"])

                for idx, fname in enumerate(obs["filternames"]):
                    chc = np.random.choice(range(len(y_train[idx_chi2_selected])))
                    samp_y_guess[22:-1][idx] = y_train[idx_chi2_selected][chc][22:-1][
                        idx
                    ]

                do_continue = False
                for tmax, npost in zip(
                    [run_params["tmax_per_iter"]], [run_params["nposterior"]]
                ):
                    signal.alarm(tmax)  # max time spent on one object in sec
                    try:
                        noiseless_theta = hatp_x_y.sample(
                            (run_params["nposterior"],),
                            x=torch.as_tensor(samp_y_guess).to(device),
                            show_progress_bars=False,
                        )
                    except TimeoutException:
                        signal.alarm(0)
                        do_continue = True
                        break

                if do_continue:
                    continue

                noiseless_theta = noiseless_theta.detach().numpy()

                ave_theta.append(noiseless_theta)

                cnt += 1
                if run_params["verbose"]:
                    if cnt % 10 == 0:
                        print("mc samples:", cnt)

            # except TimeoutException:
            #    cnt_timeout += 1
            # else:
            signal.alarm(0)

        # end time
        et = time.time()
        elapsed_time = et - st  # in secs
        if elapsed_time / 60 > run_params["tmax_all"] or (
            cnt < run_params["nmc"] / 10
            and elapsed_time / 60 * 10 > run_params["tmax_all"]
        ):
            timeout_flag = True
            use_res = False
            break
    # ------------------------------------------------

    return ave_theta, use_res, timeout_flag, cnt


def sbi_missing_and_noisy(obs, run_params, sbi_params):
    """used when observations have missing data and out-of-distribution uncertainties.
    fill in the missing bands first using the nearest neighbor approximation;
    then mc the noisy bands
    """
    signal.signal(signal.SIGALRM, timeout_handler)

    if run_params["verbose"]:
        print("sbi missing and noisy bands")

    ave_theta = []

    hatp_x_y = sbi_params["hatp_x_y"]
    y_train = sbi_params["y_train"]

    y_obs = np.copy(obs["mags_sbi"])
    sig_obs = np.copy(obs["mags_unc_sbi"])
    observed = np.concatenate([y_obs, sig_obs, [obs["redshift"]]])
    nbands = y_train.shape[1] // 2

    noisy_mask = np.copy(obs["noisy_mask"])
    noisy_idx = np.where(noisy_mask == True)[0]
    not_noisy_idx = np.where(noisy_mask == False)[0]

    invalid_mask = np.copy(obs["missing_mask"])
    y_obs_valid_only = y_obs[~invalid_mask]
    valid_idx = np.where(~invalid_mask)[0]
    not_valid_idx = np.where(invalid_mask)[0]
    not_valid_idx_unc = not_valid_idx + nbands

    # ------------------------------------------------
    kdes, use_res_missing, idx_chi2_selected = gauss_approx_missingband(
        obs, run_params, sbi_params
    )

    # start time
    st = time.time()

    lims, use_res_noisy = lim_of_noisy_guass(
        obs=obs, run_params=run_params, sbi_params=sbi_params
    )
    loc = y_obs[noisy_idx]
    scale = sig_obs[noisy_idx]

    cnt = 0
    cnt_timeout = 0
    timeout_flag = False
    while cnt < run_params["nmc"]:
        samp_y_guess = np.copy(observed)

        # first, fill in the missing bands
        for j in range(len(not_valid_idx)):
            # samp_y_guess[not_valid_idx[j]] = kdes[j].resample(size=1)
            # samp_y_guess[not_valid_idx_unc[j]] = toy_noise(
            #    flux=samp_y_guess[not_valid_idx[j]],
            #    meds_sigs=sbi_params["toynoise_meds_sigs"][not_valid_idx[j]],
            #    stds_sigs=sbi_params["toynoise_stds_sigs"][not_valid_idx[j]],
            #    verbose=run_params["verbose"],
            # )[1]
            samp_y_guess[not_valid_idx[j]] = y_train[idx_chi2_selected][
                np.random.choice(range(len(idx_chi2_selected)))
            ][not_valid_idx[j]]
            samp_y_guess[not_valid_idx_unc[j]] = y_train[idx_chi2_selected][
                np.random.choice(range(len(idx_chi2_selected)))
            ][not_valid_idx_unc[j]]

        # second, deal with OOD noise
        samp_y_guess[noisy_idx] = stats.norm.rvs(loc=loc, scale=scale)
        _nnflag = True
        for ii, this_noisy_flux in enumerate(samp_y_guess[noisy_idx]):
            if this_noisy_flux > lims[0][ii] and this_noisy_flux < lims[1][ii]:
                _nnflag &= True
            else:
                _nnflag &= False

            if _nnflag:
                samp_y_guess[noisy_idx[ii] + nbands] = y_train[idx_chi2_selected][
                    np.random.choice(range(len(idx_chi2_selected)))
                ][noisy_idx[ii] + nbands]
                # samp_y_guess[noisy_idx + nbands] = toy_noise(
                #    flux=samp_y_guess[noisy_idx[ii]],
                #    meds_sigs=sbi_params["toynoise_meds_sigs"][noisy_idx[ii]],
                #    stds_sigs=sbi_params["toynoise_stds_sigs"][noisy_idx[ii]],
                #    verbose=run_params["verbose"],
                # )[1]

            # the noise model in the training isn't quite right
            # Pan-STARRS in particular seems a little off
            # we'll have to re-train at some point, but for now just pull
            # uncertainties from the training sample
            for idx, fname in zip(valid_idx, obs["filternames"][valid_idx]):
                # if 'PanSTARRS' in fname or '2MASS' in fname or 'SDSS' in fname or 'DES' in fname:
                chc = np.random.choice(range(len(y_train[idx_chi2_selected])))
                samp_y_guess[22:][idx] = y_train[idx_chi2_selected][chc][22:][idx]

            # if we can't get one posterior sample in one second, we should move along
            # to the next MC sample
            do_continue = False
            for tmax, npost in zip(
                [1, run_params["tmax_per_iter"]], [1, run_params["nposterior"]]
            ):
                signal.alarm(tmax)  # max time spent on one object in sec
                try:
                    noiseless_theta = hatp_x_y.sample(
                        (npost,),
                        x=torch.as_tensor(samp_y_guess).to(device),
                        show_progress_bars=False,
                    )
                except TimeoutException:
                    signal.alarm(0)
                    do_continue = True
                    break

            if do_continue:
                continue

            signal.alarm(0)
            noiseless_theta = noiseless_theta.detach().numpy()

            ave_theta.append(noiseless_theta)

            cnt += 1
            if run_params["verbose"]:
                if cnt % 10 == 0:
                    print("mc samples:", cnt)

        # end time
        et = time.time()
        elapsed_time = et - st  # in secs
        if elapsed_time / 60 > run_params["tmax_all"] or (
            cnt < run_params["nmc"] / 10
            and elapsed_time / 60 * 10 > run_params["tmax_all"]
        ):
            timeout_flag = 1
            use_res = 0
            break
    # ------------------------------------------------

    if use_res_missing == 1 and use_res_noisy == 1 and timeout_flag == 0:
        use_res = 1

    return ave_theta, use_res, timeout_flag, cnt


def sbi_baseline(obs, run_params, sbi_params, max_neighbors=200):
    signal.signal(signal.SIGALRM, timeout_handler)

    if run_params["verbose"]:
        print("baseline sbi")

    flags = {
        "use_res": 0,  # True if sbi++ succeeds; False if otherwise.
        # below are for bookkeeping
        "timeout": 0,
        "use_res_missing": 0,  # True if sbi++ for missing bands succeeds
        "use_res_noisy": 0,  # True if sbi++ for noisy bands succeeds
        "noisy_data": False,  # True if sbi++ (noisy data) is called
        "missing_data": False,  # True if sbi++ (missing data) is called
        "nsamp_missing": -99,  # number of MC samples drawn
        "nsamp_noisy": -99,  # number of MC samples drawn
    }

    hatp_x_y = sbi_params["hatp_x_y"]
    y_train = sbi_params["y_train"]

    y_obs = np.copy(obs["mags"])
    sig_obs = np.copy(obs["mags_unc"])
    # copy the observed data to be used by sbi
    # missing data, if any, will be filled in later
    obs["mags_sbi"] = y_obs
    obs["mags_unc_sbi"] = sig_obs
    observed = np.concatenate([y_obs, sig_obs, [obs["redshift"]]])
    nbands = y_train.shape[1] // 2  # total number of bands

    flags["use_res"] = 1
    flags["timeout"] = False

    ### temporary for getting errors, because error model not good enough
    chi2_nei = chi2dof(mags=y_train[:, :22], obsphot=y_obs, obsphot_unc=sig_obs)

    _chi2_thres = run_params["ini_chi2"] * 1
    while _chi2_thres <= run_params["max_chi2"]:
        idx_chi2_selected = np.where(chi2_nei <= _chi2_thres)[0]
        if len(idx_chi2_selected) >= 30:
            break
        else:
            _chi2_thres += 5

    if _chi2_thres > run_params["max_chi2"]:
        chi2_selected = y_train[:]
        idx_chi2_selected = np.argsort(chi2_nei)[0:max_neighbors]
        if run_params["verbose"]:
            print("Failed to find sufficient number of nearest neighbors!")
    else:
        chi2_selected = y_train[idx_chi2_selected]

    # ------------------------------------------------
    # call baseline sbi to draw posterior samples
    signal.alarm(run_params["tmax_per_obj"])  # max time spent on one object in sec

    x = np.concatenate([y_obs, sig_obs, [obs["redshift"]]])

    for idx, fname in enumerate(obs["filternames"]):
        chc = np.random.choice(range(len(y_train[idx_chi2_selected])))
        x[22:-1][idx] = y_train[idx_chi2_selected][chc][22:-1][idx]

    try:
        ave_theta = hatp_x_y.sample(
            (run_params["np_baseline"],),
            x=torch.as_tensor(x.astype(np.float32)).to(device),
            show_progress_bars=False,
        )
        ave_theta = ave_theta.detach().numpy()
    except TimeoutException:
        flags["timeout"] = True
        ave_theta = [np.nan]
        if run_params["verbose"]:
            print("timeout!")
    else:
        signal.alarm(0)
    # ------------------------------------------------

    return ave_theta, obs, flags


def sbi_pp(obs, run_params, sbi_params, max_neighbors=200):
    """wrapper for sbi++; this should be the only function needed to be called outside this scipt under normal circumstances

    obs: a dictionary at least containing
        - "mags": observed photometry, unit must match the training set
        - "mags_unc": observed uncertainty, unit must match the training set
    run_params: a dictionary at least containing
        - ""

    """
    signal.signal(signal.SIGALRM, timeout_handler)

    flags = {
        "use_res": 0,  # True if sbi++ succeeds; False if otherwise.
        # below are for bookkeeping
        "timeout": 0,
        "use_res_missing": 0,  # True if sbi++ for missing bands succeeds
        "use_res_noisy": 0,  # True if sbi++ for noisy bands succeeds
        "noisy_data": False,  # True if sbi++ (noisy data) is called
        "missing_data": False,  # True if sbi++ (missing data) is called
        "nsamp_missing": -99,  # number of MC samples drawn
        "nsamp_noisy": -99,  # number of MC samples drawn
    }

    hatp_x_y = sbi_params["hatp_x_y"]
    y_train = sbi_params["y_train"]

    y_obs = np.copy(obs["mags"])
    sig_obs = np.copy(obs["mags_unc"])
    # copy the observed data to be used by sbi
    # missing data, if any, will be filled in later
    obs["mags_sbi"] = y_obs
    obs["mags_unc_sbi"] = sig_obs
    observed = np.concatenate([y_obs, sig_obs, [obs["redshift"]]])
    nbands = y_train.shape[1] // 2  # total number of bands

    # decide if we need to deal with missing bands
    obs["missing_mask"] = np.isnan(y_obs)
    missing_mask = np.isnan(y_obs)  # idx of missing bands
    # decide if we need to deal with noisy bands
    noisy_mask = np.zeros_like(y_obs, dtype=bool)
    for j in range(nbands):
        _toynoise = toy_noise(
            flux=y_obs[j],
            meds_sigs=sbi_params["toynoise_meds_sigs"][j],
            stds_sigs=sbi_params["toynoise_stds_sigs"][j],
            verbose=run_params["verbose"],
        )
        noisy_mask[j] = (sig_obs[j] - _toynoise[1]) / _toynoise[2] >= run_params[
            "noisy_sig"
        ]
        # if noisy_mask[j]:
        #    import pdb; pdb.set_trace()
    noisy_mask &= np.isfinite(y_obs)  # idx of noisy bands
    obs["noisy_mask"] = noisy_mask

    ave_theta = [np.nan]
    if np.any(missing_mask):
        flags["missing_data"] = True
    if np.any(noisy_mask):
        flags["noisy_data"] = True

    if not flags["missing_data"] and not flags["noisy_data"]:
        flags["use_res"] = 1
        flags["timeout"] = False
        if run_params["verbose"]:
            print("baseline sbi")

        ### temporary for getting errors, because error model not good enough
        chi2_nei = chi2dof(mags=y_train[:, :22], obsphot=y_obs, obsphot_unc=sig_obs)

        _chi2_thres = run_params["ini_chi2"] * 1
        while _chi2_thres <= run_params["max_chi2"]:
            idx_chi2_selected = np.where(chi2_nei <= _chi2_thres)[0]
            if len(idx_chi2_selected) >= 30:
                break
            else:
                _chi2_thres += 5

        if _chi2_thres > run_params["max_chi2"]:
            chi2_selected = y_train[:]
            idx_chi2_selected = np.argsort(chi2_nei)[0:max_neighbors]
            if run_params["verbose"]:
                print("Failed to find sufficient number of nearest neighbors!")
        else:
            chi2_selected = y_train[idx_chi2_selected]

        signal.alarm(run_params["tmax_per_obj"])  # max time spent on one object in sec

        x = np.concatenate([y_obs, sig_obs, [obs["redshift"]]])

        for idx, fname in enumerate(obs["filternames"]):
            chc = np.random.choice(range(len(y_train[idx_chi2_selected])))
            x[22:-1][idx] = y_train[idx_chi2_selected][chc][22:-1][idx]

        try:
            ave_theta = hatp_x_y.sample(
                (run_params["np_baseline"],),
                x=torch.as_tensor(x.astype(np.float32)).to(device),
                show_progress_bars=False,
            )
            ave_theta = ave_theta.detach().numpy()
        except TimeoutException:
            flags["timeout"] = True
            ave_theta = [np.nan]
        else:
            signal.alarm(0)

        return ave_theta, obs, flags

    if flags["missing_data"] and flags["noisy_data"]:
        (
            ave_theta,
            flags["use_res"],
            flags["timeout"],
            flags["nsamp_noisy"],
        ) = sbi_missing_and_noisy(obs=obs, run_params=run_params, sbi_params=sbi_params)
        if flags["timeout"]:
            for i, mask in enumerate(
                [
                    (obs["bands"] == "WISE_W3") | (obs["bands"] == "WISE_W4"),
                    (obs["bands"] == "WISE_W1") | (obs["bands"] == "WISE_W2"),
                    (obs["bands"] == "GALEX_NUV") | (obs["bands"] == "GALEX_FUV"),
                ]
            ):
                print("timeout!  trying without some filters")
                if i == 2:
                    run_params["tmax_all"] *= 3
                obs["missing_mask"][mask] = True
                (
                    ave_theta,
                    flags["use_res"],
                    flags["timeout"],
                    flags["nsamp_noisy"],
                ) = sbi_missing_and_noisy(
                    obs=obs, run_params=run_params, sbi_params=sbi_params
                )
                if not flags["timeout"]:
                    break

    # separate cases
    if flags["missing_data"] and not flags["noisy_data"]:
        res = sbi_missingband(
            obs=obs, run_params=run_params, sbi_params=sbi_params, seconditer=True
        )
        # if len(res) != 5:
        #    if sbi_flag == "chi2 fail":
        #        obs["missing_mask"] = res["missing_mask"][:]
        #        res = sbi_missingband(
        #            obs=obs, run_params=run_params, sbi_params=sbi_params, seconditer=True
        #        )
        # else:
        # if things timed out, then we should try dropping
        # some problematic filters
        (ave_theta, flags["use_res"], flags["nsamp_noisy"], flags["timeout"], cnt) = res
        if flags["timeout"]:
            for i, mask in enumerate(
                [
                    (obs["bands"] == "WISE_W3") | (obs["bands"] == "WISE_W4"),
                    (obs["bands"] == "WISE_W1") | (obs["bands"] == "WISE_W2"),
                    (obs["bands"] == "GALEX_NUV") | (obs["bands"] == "GALEX_FUV"),
                ]
            ):
                print("timeout!  trying without some filters")
                if i == 2:
                    run_params["tmax_all"] *= 3

                obs["missing_mask"][mask] = True
                (
                    ave_theta,
                    flags["use_res"],
                    flags["nsamp_noisy"],
                    flags["timeout"],
                    cnt,
                ) = sbi_missingband(
                    obs=obs,
                    run_params=run_params,
                    sbi_params=sbi_params,
                    seconditer=True,
                )
                if not flags["timeout"]:
                    break

            if len(res) != 5:
                raise RuntimeError(
                    "couldnt get good chi2 for nearest neighbors.  Aborting"
                )

        (
            ave_theta,
            y_guess,
            flags["use_res_missing"],
            flags["timeout"],
            flags["nsamp_missing"],
        ) = res

        flags["use_res"] = flags["use_res_missing"] * 1

    if not flags["missing_data"] and flags["noisy_data"]:
        # mc the noisy bands
        (
            ave_theta,
            flags["use_res_noisy"],
            flags["timeout"],
            ###            flags["nsamp_noisy"],
            cnt,
        ) = sbi_mcnoise(obs=obs, run_params=run_params, sbi_params=sbi_params)
        flags["use_res"] = flags["use_res_noisy"] * 1

    ave_theta = np.array(ave_theta)
    try:
        ave_theta = np.concatenate(ave_theta)
    except Exception as e:
        pass

    return ave_theta, obs, flags
