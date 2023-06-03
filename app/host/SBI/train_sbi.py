#!/usr/bin/env python
# D. Jones - 5/26/23
"""Implementation of SBI++ training for Blast"""
import math
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
from prospect.fitting import lnprobfn
from prospect.io import write_results as writer
from prospect.models import priors
from prospect.models import priors_beta as pb
from prospect.models import SpecModel
from prospect.models import transforms
from prospect.models.templates import TemplateLibrary
from prospect.sources import CSPSpecBasis
from prospect.utils.obsutils import fix_obs
from sbi import inference as Inference
from sbi import utils as Ut
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.stats import t

# torch

all_filters = Filter.objects.filter(~Q(name="DES_i") & ~Q(name="DES_Y"))
massmet = np.loadtxt("host/SBI/priors/gallazzi_05_massmet.txt")
z_age, age = np.loadtxt("host/SBI/priors/wmap9_z_age.txt", unpack=True)
f_age_z = interp1d(age, z_age)
z_b19, tl_b19, sfrd_b19 = np.loadtxt(
    "host/SBI/priors/behroozi_19_sfrd.txt", unpack=True
)
spl_z_sfrd = UnivariateSpline(z_b19, sfrd_b19, s=0, ext=1)
spl_tl_sfrd = UnivariateSpline(tl_b19, sfrd_b19, s=0, ext=1)  # tl in yrs

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


def build_model(observations=None, **extras):
    """
    Construct all model components
    """

    model_params = TemplateLibrary["parametric_sfh"]
    model_params.update(TemplateLibrary["nebular"])
    model_params["zred"]["init"] = 0.1

    return SpecModel(model_params)


def build_sps(zcontinuous=2, compute_vega_mags=False, **extras):
    sps = CSPSpecBasis(
        zcontinuous=zcontinuous, compute_vega_mags=compute_vega_mags
    )  # special to remove redshifting issue
    return sps


def build_noise(**extras):
    return None, None


def draw_thetas():

    # draw from the mass function at the above zred
    mass = priors.LogUniform(mini=100000000.0, maxi=1000000000000.0).sample()

    # given mass from above, draw logzsol
    logzsol = priors.FastUniform(a=-2, b=0.19).sample()

    # dust
    dust2 = priors.FastUniform(a=0.0, b=2.0).sample()

    # tage
    tage = priors.FastUniform(a=0.001, b=13.8).sample()

    # tau
    tau = priors.LogUniform(mini=0.1, maxi=30).sample()

    return np.concatenate(
        [
            np.atleast_1d(mass),
            np.atleast_1d(logzsol),
            np.atleast_1d(dust2),
            np.atleast_1d(tage),
            np.atleast_1d(tau),
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

    def do(self):

        # parameters
        needed_size = 150000
        run_params = {"ichunk": 0, "needed_size": needed_size}
        run_params["add_duste"] = True
        run_params["add_igm"] = True
        run_params["add_neb"] = True
        run_params["dynesty"] = False
        run_params["optmization"] = False
        run_params["emcee"] = False

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
            theta = draw_thetas()

            # call prospector
            # generate the model SED at given theta
            zred = priors.FastUniform(a=0.0, b=0.2).sample()
            model.params["zred"] = np.atleast_1d(zred)
            spec, phot, mfrac = model.predict(theta, obs=obs, sps=sps)
            predicted_mags = -2.5 * np.log10(phot)

            flag = True
            for i, f in enumerate(all_filters):
                # probably gonna have some unit issues here
                flag &= (predicted_mags[i] >= cat_min[f.name]) & (
                    predicted_mags[i] <= cat_max[f.name]
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
            list_phot.append(np.append(list_phot_single, list_phot_errs_single))
            print(len(list_phot))

        save_phot = True
        if save_phot:
            hf_phot = h5py.File("host/SBI/sbi_phot.h5", "w")
            hf_phot.create_dataset("wphot", data=obs["phot_wave"])
            hf_phot.create_dataset("phot", data=list_phot)
            hf_phot.create_dataset("mfrac", data=list_mfrac)
            hf_phot.create_dataset("theta", data=list_thetas)

            try:
                hf_phot.close()
            except (AttributeError):
                pass

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
