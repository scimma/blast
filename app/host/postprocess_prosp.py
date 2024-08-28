"""Transform from prospector outputs to physical parameters.
Bingjie Wang 7/13/23

#####
example:
mod_fsps = build_model() # the same as the function I wrote in train_sbi.py
sps = build_sps() # the same as in train_sbi.py
run_all(fname='test.h5', mod_fsps=mod_fsps, sps=sps, percents=[15.9,50,84.1])

To use this script on SBI posteriors, replace that block that reads in a Prospector .h5 file in run_all(). res['theta_index'] should be kept the same, and res['chain'] should be replaced with the SBI posteriors.

If you train the SBI on the stellar mass, instead the total mass formed (the default from prospector), then there is no need to run the block that does this transformation in run_all().
"""
import os
import random
import sys

import numpy as np
import prospect.io.read_results as reader
from astropy.cosmology import WMAP9 as cosmo
from astropy.io import fits
from astropy.table import Table
from dynesty.utils import resample_equal
from prospect.models.transforms import logsfr_ratios_to_masses
from prospect.plotting.corner import quantile


def theta_index(prior="p-alpha"):
    """Corresponding to the Prospector-alpha model"""
    # index = {'zred': slice(0, 1, None), 'logmass': slice(1, 2, None), 'logzsol': slice(2, 3, None), 'logsfr_ratios': slice(3, 9, None),
    #         'dust2': slice(9, 10, None), 'dust_index': slice(10, 11, None), 'dust1_fraction': slice(11, 12, None),
    #         'log_fagn': slice(12, 13, None), 'log_agn_tau': slice(13, 14, None), 'gas_logz': slice(14, 15, None),
    #         'duste_qpah': slice(15, 16, None), 'duste_umin': slice(16, 17, None), 'log_duste_gamma': slice(17, 18, None)}
    index = {
        "zred": slice(0, 1, None),
        "logmass": slice(1, 2, None),
        "logzsol": slice(2, 3, None),
        "logsfr_ratios": slice(3, 9, None),
        "dust2": slice(9, 10, None),
        "dust_index": slice(10, 11, None),
        "dust1_fraction": slice(11, 12, None),
        "log_fagn": slice(12, 13, None),
        "log_agn_tau": slice(13, 14, None),
        "gas_logz": slice(14, 15, None),
        "duste_qpah": slice(15, 16, None),
        "duste_umin": slice(16, 17, None),
        "log_duste_gamma": slice(17, 18, None),
    }

    return index


def getPercentiles(
    chain, quantity="zred", theta_index=None, percents=[15.9, 50.0, 84.1]
):
    """get the 16/50/84th percentile for a scalar output
    that does not need transform functions
    (e.g., mass, dust, etc).
    """
    try:
        npix = chain[theta_index[quantity]].shape[0]
    except ValueError:
        print('"' + quantity + '" does not exist in the output.')
        return

    p = np.percentile(chain[:, theta_index[quantity]], q=percents)
    return p.T


def z_to_agebins(zred=None, agebins=None, nbins_sfh=7, amin=7.1295, **extras):
    """new agebins defined in Wang+2024
    https://ui.adsabs.harvard.edu/abs/2024ApJS..270...12W/abstract
    """
    tuniv = cosmo.age(zred).value * 1e9
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


def stepInterp(ab, val, ts):
    """ab: agebins vector
    val: the original value (sfr, etc) that we want to interpolate
    ts: new values we want to interpolate to"""
    newval = np.zeros_like(ts) + np.nan
    for i in range(0, len(ab)):
        newval[(ts >= ab[i, 0]) & (ts < ab[i, 1])] = val[i]
    newval[-1] = 0
    return newval


def get_mwa(agebins, sfr):
    ages = 10**agebins
    dt = np.abs(ages[:, 1] - ages[:, 0])
    return np.sum(np.mean(ages, axis=1) * sfr * dt) / np.sum(sfr * dt) / 1e9  # in Gyr


def getSFH(
    chain,
    nagebins=7,
    sfrtimes=[10, 30, 100],
    tbins=100,
    theta_index=None,
    rtn_chains=False,
    percents=[15.9, 50, 84.1],
    zred=None,
):
    """get the 16/50/84th percentile of the SFH for each pixel.

    Parameters
    ___________

    chain : chain object as defined in pirate.io
    nagebins : number of agebins
    sfrtimes : timescales that we want the output SFR averaged over (in Myr)
    tbins : how many log-scaled timebins to interpolate results onto

    Returns
    __________

    age_interp : list of ages -- lookback time in Gyr
    sfh : 3 x npix x len(age_interp) array that gives the 16/50/84% interval of the SFH at each lookback time
    mwa : 3 x npix array-- 16/50/84th percentile mass-weighted age
    sfr : 3 x npix x n_sfrtimes -- 16/50/84% SFR over each timescale
    """

    # run the transforms to get the SFH at each step in the chain for every pixel:
    # (pixel, agebin, chain step)
    nsamples = chain.shape[0]
    allsfhs = np.zeros((nsamples, nagebins))
    allagebins = np.zeros((nsamples, nagebins, 2))
    allMWA = np.zeros((nsamples))

    # make the SFR for each draw in each pixel
    for iteration in range(nsamples):
        # get agebins and time spacing
        allagebins[iteration, :, :] = z_to_agebins(
            zred=zred
        )  # chain[:,theta_index['zred']][iteration][0])
        dt = 10 ** allagebins[iteration, :, 1] - 10 ** allagebins[iteration, :, 0]

        # get mass per bin
        masses = logsfr_ratios_to_masses(
            logsfr_ratios=chain[:, theta_index["logsfr_ratios"]][iteration, :],
            agebins=allagebins[iteration, :, :],
            logmass=chain[:, theta_index["logmass"]][iteration][0],
        )

        # convert to sfr
        allsfhs[iteration, :] = masses / dt

        # go ahead and get the mass-weighted age too
        allMWA[iteration] = get_mwa(allagebins[iteration, :, :], allsfhs[iteration, :])

    # interpolate everything onto the same time grid
    # define the grid as going from lookback time = 0 to the oldest time in all the samples
    # with 1000 log-spaced samples in between (might want to update this later!)
    allagebins_ago = 10**allagebins / 1e9
    age_interp = (np.logspace(1, np.log10(np.max(allagebins_ago * 1e9)), tbins) / 1e9)  # lookback time in Gyr
    age_interp_linear = np.linspace(0, 100, 51) / 1e3 # for <sfr> only; Myr -> Gyr
    allsfhs_interp = np.zeros((nsamples, len(age_interp)))
    allsfhs_interp_linear = np.zeros((nsamples, len(age_interp_linear)))
    allsfrs = np.zeros((nsamples, len(sfrtimes)))
    for iteration in range(nsamples):
        allsfhs_interp[iteration, :] = stepInterp(allagebins_ago[iteration, :], allsfhs[iteration, :], age_interp)

        allsfhs_interp_linear[iteration, :] = stepInterp(allagebins_ago[iteration,:], allsfhs[iteration,:], age_interp_linear)
        nan_idx = np.isnan(allsfhs_interp_linear[iteration, :])
        foo = allsfhs_interp_linear[iteration, ~nan_idx]
        allsfhs_interp_linear[iteration, nan_idx] = foo[0]

        # also get SFR averaged over all the timescales we want
        for i, time in enumerate(sfrtimes):
            allsfrs[iteration, i] = np.mean(allsfhs_interp_linear[iteration, (age_interp_linear<=time*1e-3)])

    if rtn_chains:
        return (age_interp, allsfhs_interp, allMWA, allsfrs, allsfhs)
    else:
        # sfr and MWA percentiles
        sfh = np.percentile(allsfhs_interp, percents, axis=0)
        mwa = np.percentile(allMWA, percents)
        sfr = np.percentile(allsfrs, percents, axis=0)
        allsfh_nointerp = np.percentile(allsfhs, percents, axis=0)
        return (age_interp, sfh.T, mwa, sfr.T, allsfh_nointerp.T)


def get_all_outputs_and_chains(
    res=None,
    keys=None,
    run_params=None,
    percents=[15.9, 50, 84.1],
    nsamp=1000,
    zred=None,
):
    """get all the outputs;
    nsamp: number of posterior samples drawn
    """

    # load the output file and get the unweighted chain
    chain = res["chain"]
    theta_index = res["theta_index"]

    # get the basic quantities
    percentiles = {}
    for key in keys:
        percentiles[key] = getPercentiles(chain, key, theta_index)

    ###
    train_pars = np.array(
        [
            0.11929193,
            10.39623445,
            -1.32188065,
            0.18913014,
            0.16209094,
            0.06441716,
            -0.19440235,
            0.11935996,
            0.37152376,
            0.34494525,
            -0.42501956,
            0.51024255,
            -2.06271797,
            1.37400889,
            -1.21739342,
            3.32965967,
            1.28220919,
            -1.79931691,
        ]
    )
    age_interp, allsfhs_interp, allMWA, allsfrs, allsfhs_nointerp = getSFH(
        chain, theta_index=theta_index, rtn_chains=True, zred=zred
    )
    # import pdb; pdb.set_trace()
    # sfr and MWA percentiles
    # rtn_chains is defaulted to False: so need to transpose sfh and sfr
    allsfhs_interp[np.isnan(allsfhs_interp)] = 0
    sfh = np.percentile(allsfhs_interp, percents, axis=0)
    mwa = np.percentile(allMWA, percents)
    sfr = np.percentile(allsfrs, percents, axis=0)
    sfh_noint = np.percentile(allsfhs_nointerp, percents, axis=0)
    
    # each of these keys is a (xpix x ypix x nparam x 16/50/84%) map
    percentiles["age_interp"] = age_interp
    percentiles["sfh"] = sfh.T
    percentiles["mwa"] = mwa
    percentiles["sfr"] = sfr.T
    percentiles["sfh_binned"] = sfh_noint.T
    
    # saved chains are subsampled, so that we can plot stellar mass on the corner plot
    chain_len = res["chain"].shape[0]
    if nsamp > chain_len:
        sub_idx = random.sample(range(res["chain"].shape[0]), chain_len)
    else:
        sub_idx = random.sample(range(res["chain"].shape[0]), nsamp)

    chain = res["chain"][sub_idx]
    chains = {
        "age_interp": age_interp,
        "sfh": allsfhs_interp,
        "mwa": allMWA[sub_idx],
        "sfr": allsfrs[sub_idx, :],
    }

    for _k in keys:
        chains[_k] = np.concatenate(chain[:, theta_index[_k]])

    return percentiles, chains, sub_idx


def run_all(
    fname,
    unw_fname,
    perc_fname,
    zred,
    prior="p-alpha",
    mod_fsps=None,
    sps=None,
    percents=[15.9, 50, 84.1],
    use_weights=True,
    obs=None,
    **extra
):
    # XXX read in prospector outputs
    if obs is None:
        res, obs, _ = reader.results_from(fname, dangerous=False)
    else:
        res, _, _ = reader.results_from(fname, dangerous=False)

    res["theta_index"] = theta_index(prior)
    # If sampling using dynesty, we resample the chains so that each has an equal weight.
    if use_weights:
        res["chain"] = resample_equal(res["chain"], res["weights"])  # unweighted_chain

    """
    I think the chains output by SBI are the same as the Prospector ones, unless you change the paramters constituting the training set.
    If so, then you could also use this same script to postprocess the SBI posteriors -- just replace the res['chain'] with the SBI posteriors.
    """

    # zred_idx = 0
    mass_idx = 1
    # scalar outputs that do not need transform functions
    keys = [
        "logzsol",
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
    percentiles, chains, sub_idx = get_all_outputs_and_chains(res, keys=keys, zred=zred)

    # ---------- total mass formed -> stellar mass
    stellarmass = []
    ssfr = []
    modphots_all = []
    modspecs_all = []

    for i, _subidx in enumerate(sub_idx):
        modspec, modmags, sm = mod_fsps.predict(
            res["chain"][int(_subidx)], sps=sps, obs=obs
        )
        modphots_all.append(modmags)  # model photometry
        modspecs_all.append(modspec)  # model spectrum
        _mass = res["chain"][int(_subidx)][mass_idx]

        stellarmass.append(
            _mass
        )  ## mass is already converted to surviving mass in training

        ssfr.append(
            chains["sfr"][i] / 10 ** stellarmass[-1]
        )  # sfr chains are already sub-sampled
    stellarmass = np.array(stellarmass)
    ssfr = np.array(ssfr)
    modphots_all = np.array(modphots_all)

    percentiles["stellar_mass"] = np.percentile(stellarmass, percents)
    percentiles["ssfr"] = np.percentile(ssfr, percents, axis=0).T
    percentiles["modphot"] = np.percentile(modphots_all, percents, axis=0).T
    percentiles["modspec"] = np.percentile(modspecs_all, percents, axis=0).T

    # XXX save percentiles to files
    # perc_fname = fname.replace('mcmc', 'perc')
    # perc_fname = perc_fname.replace('.h5', '.npz')
    np.savez(perc_fname, percentiles=percentiles, theta_lbs=res["theta_labels"])

    # XXX save chains to files
    # unw_fname = fname.replace('mcmc', 'chain')
    # unw_fname = unw_fname.replace('.h5', '.npz')
    chains["stellar_mass"] = stellarmass
    chains["ssfr"] = ssfr
    np.savez(unw_fname, chains=chains)

    # load as
    # fnpz = np.load(perc_fname, allow_pickle=True)
    # perc = fnpz['percentiles'][()]
