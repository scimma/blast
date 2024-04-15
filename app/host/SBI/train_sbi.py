#!/usr/bin/env python
# D. Jones - 5/26/23
# modified by B. Wang - 6/7/23
"""Implementation of SBI++ training for Blast"""
import math
import os
import pickle

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.cosmology import WMAP9 as cosmo
from django.db.models import Q
from django_cron import CronJobBase
from django_cron import Schedule
from host.models import *
from host.photometric_calibration import ab_mag_to_mJy
from host.photometric_calibration import mJy_to_maggies
from host.prospector import build_model
from host.prospector import build_obs
from prospect.fitting import fit_model as fit_model_prospect
from prospect.io import write_results as writer
from prospect.models import priors
from prospect.models import SpecModel
from prospect.models.sedmodel import PolySpecModel
from prospect.models.templates import TemplateLibrary
from prospect.sources import CSPSpecBasis
from prospect.sources import FastStepBasis
from prospect.utils.obsutils import fix_obs
from sbi import inference as Inference
from sbi import utils as Ut

_outfile = os.environ.get("OUTFILE")

# torch

all_filters = Filter.objects.filter(~Q(name="DES_i") & ~Q(name="DES_Y"))
massmet = np.loadtxt("host/SBI/priors/gallazzi_05_massmet.txt")

nhidden = 500  # architecture
nblocks = 15  # architecture

fanpe = "host/SBI/SBI_model.pt"  # name for the .pt file where the trained model will be saved
fsumm = "host/SBI/SBI_model_summary.p"  # name for the .p file where the training summary will be saved; useful if want to check the convergence, etc.

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


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


def scale(mass):
    upper_84 = np.interp(mass, massmet[:, 0], massmet[:, 3])
    lower_16 = np.interp(mass, massmet[:, 0], massmet[:, 2])
    return upper_84 - lower_16


def loc(mass):
    return np.interp(mass, massmet[:, 0], massmet[:, 1])


def draw_thetas(flat=False):

    if flat:
        zred = priors.FastUniform(a=0.0, b=0.2 + 1e-3).sample()
        logmass = priors.FastUniform(a=7.0, b=12.5).sample()
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
        zred = priors.FastUniform(a=0.0, b=0.2 + 1e-3).sample()
        logmass = priors.FastUniform(a=7.0, b=12.5).sample()
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

    def do(self, do_phot=True, do_train=False):

        # parameters
        needed_size = 150000
        run_params = {"ichunk": 0, "needed_size": needed_size}
        run_params["add_duste"] = True
        run_params["add_igm"] = True
        run_params["add_neb"] = True
        run_params["dynesty"] = False
        run_params["optmization"] = False
        run_params["emcee"] = False

        if do_phot:
            obs, model, sps, noise = self.build_all(**run_params)

            run_params["sps_libraries"] = sps.ssp.libraries
            run_params["param_file"] = __file__

            # get the minimum, maximum magnitudes
            cat_min, cat_max, cat_full = {}, {}, {}
            for f in all_filters:
                mag, snr = np.loadtxt(
                    f"host/SBI/snrfiles/{f.name}_magvsnr.txt", unpack=True
                )

                cat_min[f.name] = np.min(mag)
                cat_max[f.name] = np.max(mag)
                cat_full[f.name] = (mag, snr)

            ### start putting together the synthetic data
            list_thetas = []
            list_mfrac = []
            list_phot = []
            while len(list_phot) < needed_size:
                if not len(list_phot) % 3:
                    theta = draw_thetas(flat=True)
                else:
                    theta = draw_thetas(flat=False)

                # call prospector
                # generate the model SED at given theta
                spec, phot, mfrac = model.predict(theta, obs=obs, sps=sps)

                predicted_mags = -2.5 * np.log10(phot)
                theta[1] += np.log10(mfrac)

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
                    phot_err = phot[i] / snr
                    phot_random = np.random.normal(phot[i], phot_err)
                    phot_random_mags = maggies_to_asinh(phot_random)
                    phot_err_mags = 2.5 / np.log(10) * phot_err / phot[i]

                    list_phot_single = np.append(list_phot_single, [phot_random_mags])
                    list_phot_errs_single = np.append(
                        list_phot_errs_single, [phot_err_mags]
                    )
                # adding in the redshift here
                list_phot.append(
                    np.concatenate(
                        (list_phot_single, list_phot_errs_single, [theta[0]]),
                    )
                )
                print(len(list_phot))

            save_phot = True
            if save_phot:
                hf_phot = h5py.File(_outfile, "w")
                hf_phot.create_dataset("wphot", data=obs["phot_wave"])
                hf_phot.create_dataset("phot", data=list_phot)
                hf_phot.create_dataset("mfrac", data=list_mfrac)
                hf_phot.create_dataset("theta", data=list_thetas)

                try:
                    hf_phot.close()
                except AttributeError:
                    pass

        if do_train:
            data = h5py.File("host/SBI/sbi_phot_1.h5", "r")
            list_thetas, list_phot, list_mfrac, list_phot_wave = (
                np.array(data["theta"]),
                np.array(data["phot"]),
                np.array(data["mfrac"]),
                np.array(data["wphot"]),
            )
            for i in range(2, 11):
                data2 = h5py.File(f"host/SBI/sbi_phot_{i}.h5", "r")
                list_thetas = np.append(list_thetas, np.array(data2["theta"]), axis=0)
                list_phot = np.append(list_phot, np.array(data2["phot"]), axis=0)
                list_mfrac = np.append(list_mfrac, np.array(data2["mfrac"]), axis=0)
                list_phot_wave = np.append(
                    list_phot_wave, np.array(data2["wphot"]), axis=0
                )
                hf_phot = h5py.File("host/SBI/sbi_phot.h5", "w")
                hf_phot.create_dataset("wphot", data=list_phot_wave)
                hf_phot.create_dataset("phot", data=list_phot)
                hf_phot.create_dataset("mfrac", data=list_mfrac)
                hf_phot.create_dataset("theta", data=list_thetas)
                hf_phot.close()

            x_train = np.array(list_thetas)
            y_train = np.array(list_phot)
            # now do the training
            anpe = Inference.SNPE(
                density_estimator=Ut.posterior_nn(
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
            torch.save(p_x_y_estimator.state_dict(), fanpe)

            # save training summary
            pickle.dump(anpe._summary, open(fsumm, "wb"))
            print(anpe._summary)

        print("Finished.")


if __name__ == "__main__":
    ts = TrainSBI()
    ts.main()
