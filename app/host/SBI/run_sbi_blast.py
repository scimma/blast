import os

from django.conf import settings

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import numpy as np
import math
from scipy.interpolate import interp1d

# torch
import torch
from sbi import utils as Ut
from sbi import inference as Inference
from host.models import Filter
from django.db.models import Q

# all the functions implementing SBI++ are contained in `sbi_pp.py`
from host.SBI import sbi_pp
import h5py
import pickle

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

run_params = {
    "nmc": 50,  # number of MC samples
    "nposterior": 50,  # number of posterior samples per MC drawn
    "np_baseline": 500,  # number of posterior samples used in baseline SBI
    "ini_chi2": 5,  # chi^2 cut usedi in the nearest neighbor search
    "max_chi2": 5000,  # the maximum chi^2 to reach in case we Incremently increase the chi^2
    # in the case of insufficient neighbors
    "noisy_sig": 3,  # deviation from the noise model, above which the measuremnt is taked as OOD
    "tmax_per_obj": 120000,  # max time spent on one object / mc sample in secs
    "tmax_all": 600000,  # max time spent on all mc samples in mins
    "outdir": "output",  # output directory
    "verbose": True,
    "tmax_per_iter": 60,
}

sbi_params = {
    "anpe_fname_global": f"{settings.SBIPP_ROOT}/SBI_model_global.pt",  # trained sbi model
    "train_fname_global": f"{settings.SBIPP_PHOT_ROOT}/sbi_phot_global.h5",  # training set
    "anpe_fname_local": f"{settings.SBIPP_ROOT}/SBI_model_local.pt",  # trained sbi model
    "train_fname_local": f"{settings.SBIPP_PHOT_ROOT}/sbi_phot_local.h5",  # training set
    "nhidden": 500,  # architecture of the trained density estimator
    "nblocks": 15,  # architecture of the trained density estimator
}

all_filters = Filter.objects.filter(~Q(name="DES_i") & ~Q(name="DES_Y"))
uv_filters = ["GALEX_NUV", "GALEX_FUV", "SDSS_u", "DES_u"]
opt_filters = [
    "SDSS_g",
    "SDSS_r",
    "SDSS_i",
    "SDSS_z",
    "PanSTARRS_g",
    "PanSTARRS_r",
    "PanSTARRS_i",
    "PanSTARRS_z",
    "PanSTARRS_y",
    "DES_g",
    "DES_r",
]
ir_filters = [
    "WISE_W1",
    "WISE_W2",
    "WISE_W3",
    "WISE_W4",
    "2MASS_J",
    "2MASS_H",
    "2MASS_K",
]


# training set
def run_training_set():
    for _fit_type in ["global", "local"]:
        data = h5py.File(sbi_params[f"train_fname_{_fit_type}"], "r")
        x_train = np.array(data["theta"])  # physical parameters
        y_train = np.array(data["phot"])  # fluxes & uncertainties

        # we will only need the lower & upper limits to be passed to sbi as "priors"
        # here we simply read in the bounds from the training set
        prior_low = sbi_pp.prior_from_train("ll", x_train=x_train)
        prior_high = sbi_pp.prior_from_train("ul", x_train=x_train)
        lower_bounds = torch.tensor(prior_low).to(device)
        upper_bounds = torch.tensor(prior_high).to(device)
        prior = Ut.BoxUniform(low=lower_bounds, high=upper_bounds, device=device)

        # density estimater
        anpe = Inference.SNPE(
            prior=prior,
            density_estimator=Ut.posterior_nn(
                "maf",
                hidden_features=sbi_params["nhidden"],
                num_transforms=sbi_params["nblocks"],
            ),
            device=device,
        )
        x_tensor = torch.as_tensor(x_train.astype(np.float32)).to(device)
        y_tensor = torch.as_tensor(y_train.astype(np.float32)).to(device)
        anpe.append_simulations(x_tensor, y_tensor)
        p_x_y_estimator = anpe._build_neural_net(x_tensor, y_tensor)
        p_x_y_estimator.load_state_dict(
            torch.load(
                sbi_params[f"anpe_fname_{_fit_type}"], map_location=torch.device(device)
            )
        )
        anpe._x_shape = Ut.x_shape_from_simulation(y_tensor)
        if _fit_type == "global":
            hatp_x_y_global = anpe.build_posterior(
                p_x_y_estimator, sample_with="rejection"
            )
            y_train_global = y_train[:]
            x_train_global = x_train[:]
        elif _fit_type == "local":
            hatp_x_y_local = anpe.build_posterior(
                p_x_y_estimator, sample_with="rejection"
            )
            y_train_local = y_train[:]
            x_train_local = x_train[:]

    print("""Storing training sets as data files...""")
    with open(
        os.path.join(settings.SBI_TRAINING_ROOT, "hatp_x_y_global.pkl"), "wb"
    ) as handle:
        pickle.dump(hatp_x_y_global, handle)
    with open(
        os.path.join(settings.SBI_TRAINING_ROOT, "y_train_global.pkl"), "wb"
    ) as handle:
        pickle.dump(y_train_global, handle)
    with open(
        os.path.join(settings.SBI_TRAINING_ROOT, "x_train_global.pkl"), "wb"
    ) as handle:
        pickle.dump(x_train_global, handle)
    with open(
        os.path.join(settings.SBI_TRAINING_ROOT, "hatp_x_y_local.pkl"), "wb"
    ) as handle:
        pickle.dump(hatp_x_y_local, handle)
    with open(
        os.path.join(settings.SBI_TRAINING_ROOT, "y_train_local.pkl"), "wb"
    ) as handle:
        pickle.dump(y_train_local, handle)
    with open(
        os.path.join(settings.SBI_TRAINING_ROOT, "x_train_local.pkl"), "wb"
    ) as handle:
        pickle.dump(x_train_local, handle)


try:
    print("""Loading training sets from data files...""")
    with open(
        os.path.join(settings.SBI_TRAINING_ROOT, "hatp_x_y_global.pkl"), "rb"
    ) as handle:
        hatp_x_y_global = pickle.load(handle)
    with open(
        os.path.join(settings.SBI_TRAINING_ROOT, "y_train_global.pkl"), "rb"
    ) as handle:
        y_train_global = pickle.load(handle)
    with open(
        os.path.join(settings.SBI_TRAINING_ROOT, "x_train_global.pkl"), "rb"
    ) as handle:
        x_train_global = pickle.load(handle)
    with open(
        os.path.join(settings.SBI_TRAINING_ROOT, "hatp_x_y_local.pkl"), "rb"
    ) as handle:
        hatp_x_y_local = pickle.load(handle)
    with open(
        os.path.join(settings.SBI_TRAINING_ROOT, "y_train_local.pkl"), "rb"
    ) as handle:
        y_train_local = pickle.load(handle)
    with open(
        os.path.join(settings.SBI_TRAINING_ROOT, "x_train_local.pkl"), "rb"
    ) as handle:
        x_train_local = pickle.load(handle)
    print("""Training sets loaded.""")
except Exception as err:
    print(f"""Error loading training sets: {err}. Regenerating...""")
    run_training_set()
    print("""Training sets generated.""")


def maggies_to_asinh(x):
    """asinh magnitudes"""
    a = 2.50 * np.log10(np.e)
    mu = 35.0
    return -a * math.asinh((x / 2.0) * np.exp(mu / a)) + mu


def fit_sbi_pp(observations, n_filt_cuts=True, fit_type="global"):
    np.random.seed(100)  # make results reproducible

    # toy noise model
    meds_sigs, stds_sigs = [], []

    for f in all_filters:
        toy_noise_x, toy_noise_y = np.loadtxt(
            f"host/SBI/snrfiles/{f.name}_magvsnr.txt", dtype=float, unpack=True
        )
        meds_sigs += [
            interp1d(
                toy_noise_x,
                1.0857 * 1 / toy_noise_y,
                kind="slinear",
                fill_value="extrapolate",  # (0.01,1.0),
                bounds_error=False,
            )
        ]
        stds_sigs += [
            interp1d(
                toy_noise_x,
                1.0857 * 1 / toy_noise_y,
                kind="slinear",
                fill_value="extrapolate",  # (0.01,1.0),
                bounds_error=False,
            )
        ]
    sbi_params["toynoise_meds_sigs"] = meds_sigs
    sbi_params["toynoise_stds_sigs"] = stds_sigs

    # a testing object of which the noises are OOD
    mags, mags_unc, filternames, wavelengths = (
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
    )

    has_uv, has_opt, has_ir = False, False, False
    for f in all_filters:
        if f.name in observations["filternames"]:
            iflt = np.array(observations["filternames"]) == f.name
            mags = np.append(mags, maggies_to_asinh(observations["maggies"][iflt]))
            mags_unc = np.append(
                mags_unc,
                2.5
                / np.log(10)
                * observations["maggies_unc"][iflt]
                / observations["maggies"][iflt],
            )
            if f.name in uv_filters:
                has_uv = True
            elif f.name in opt_filters:
                has_opt = True
            elif f.name in ir_filters:
                has_ir = True
        else:
            mags = np.append(mags, np.nan)
            mags_unc = np.append(mags_unc, np.nan)
        filternames = np.append(filternames, f.name)
        wavelengths = np.append(wavelengths, f.transmission_curve().wave_effective)

    obs = {}
    obs[
        "mags"
    ] = mags  ##np.array([maggies_to_asinh(p) for p in observations['maggies']])
    obs[
        "mags_unc"
    ] = mags_unc  ##2.5/np.log(10)*observations['maggies_unc']/observations['maggies']
    obs["redshift"] = observations["redshift"]
    obs["wavelengths"] = wavelengths
    obs["filternames"] = filternames

    if n_filt_cuts and not has_opt and (not has_ir or not has_uv):
        print("not enough filters for reliable/fast inference")
        return {}, 1

    # prepare to pass the reconstructed model to sbi_pp
    if fit_type == "global":
        sbi_params["y_train"] = y_train_global
        sbi_params["theta_train"] = x_train_global
        sbi_params["hatp_x_y"] = hatp_x_y_global
    elif fit_type == "local":
        sbi_params["y_train"] = y_train_local
        sbi_params["hatp_x_y"] = hatp_x_y_local
        sbi_params["theta_train"] = x_train_local

    # Run SBI++
    chain, obs, flags = sbi_pp.sbi_pp(
        obs=obs, run_params=run_params, sbi_params=sbi_params
    )

    # pathological format as we're missing some stuff that prospector usually spits out
    output = {"sampling": [{"samples": chain[:, :], "eff": 100}, 0]}
    return output, 0
