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

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from operator import itemgetter
import numpy as np
from numpy.random import normal, uniform
from scipy import stats
from scipy.interpolate import interp1d

# torch
import torch

torch.set_num_threads(1)
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
    while cnt < run_params["nmc"]:
        signal.alarm(
            0
        )  # run_params["tmax_per_obj"])  # max time spent on one object in sec -- disabled for now
        try:
            x = np.copy(observed)

            for j, idx in enumerate(not_valid_idx):
                x[not_valid_idx[j]] = np.random.choice(guess_ndata.T[j])
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
                chc = np.random.choice(range(len(y_train[idx_chi2_selected])))
                x[22:-1][idx] = y_train[idx_chi2_selected][chc][22:-1][idx]

            all_x.append(x)

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

            ave_theta.append(noiseless_theta)

            cnt += 1
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

        # if things timed out, then we should try dropping
        # some problematic filters
        # I think this is mostly deprecated now
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
