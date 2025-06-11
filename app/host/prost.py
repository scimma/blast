import os
from shutil import rmtree
from django.conf import settings
from .models import Host
from astropy.coordinates import SkyCoord
# Prost dependencies
import pandas as pd
from scipy.stats import gamma, halfnorm, uniform
# from astropy.cosmology import LambdaCDM
from host.astro_prost.helpers import SnRateAbsmag
from host.astro_prost.associate import associate_sample


def run_prost(transient, output_dir_root=settings.PROST_OUTPUT_ROOT):
    """
    Finds the information about the host galaxy given the position of the supernova.
    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        On Sky position of the source to be matched.
    :name : str, default='No name'
        Name of the the object.
    Returns
    -------
    :host_information : ~astropy.coordinates.SkyCoord`
        Host position
    """
    transient_position = SkyCoord(
        ra=transient.ra_deg, dec=transient.dec_deg, unit="deg"
    )

    # Define and create output file root directory
    output_dir = os.path.join(output_dir_root, transient.name)
    os.makedirs(output_dir, exist_ok=True)

    # define priors for properties
    priorfunc_z = halfnorm(loc=0.0001, scale=0.5)
    priorfunc_offset = uniform(loc=0, scale=5)
    priorfunc_absmag = uniform(loc=-30, scale=20)

    # cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

    likefunc_offset = gamma(a=0.75)
    likefunc_absmag = SnRateAbsmag(a=-25, b=20)

    priors = {
        "offset": priorfunc_offset,
        "absmag": priorfunc_absmag
    }
    likes = {
        "offset": likefunc_offset,
        "absmag": likefunc_absmag
    }

    transient_catalog = pd.DataFrame(
        {'IAUID': [transient.name],
         'RA': [transient_position.ra.deg],
         'Dec': [transient_position.dec.deg]
         }
    )
    # add the redshift info from the transient if it exists
    if transient.redshift is not None:
        priors['redshift'] = priorfunc_z
        transient_catalog['redshift'] = transient.redshift

    catalogs = ["glade", "decals", "panstarrs", "skymapper"]
    transient_coord_cols = ("RA", "Dec")
    # transient_name_col = "IAUID"
    parallel = False
    save = True
    progress_bar = False
    cat_cols = False

    try:
        hosts = associate_sample(
            transient_catalog,
            coord_cols=transient_coord_cols,
            priors=priors,
            likes=likes,
            catalogs=catalogs,
            parallel=parallel,
            save=save,
            save_path=output_dir,
            progress_bar=progress_bar,
            cat_cols=cat_cols,
            verbose=0,
        )
        host = Host(
            ra_deg=hosts["host_ra"][0],
            dec_deg=hosts["host_dec"][0],
            name=hosts["host_name"][0],
        )
        if hosts['host_redshift_info'][0] == 'SPEC':
            host.redshift = hosts["host_redshift_mean"][0]
            host.redshift_err = hosts["host_redshift_std"][0]
        elif hosts['host_redshift_info'][0] == 'PHOT' and hosts['best_cat'][0] != 'panstarrs':
            host.photometric_redshift = hosts["host_redshift_mean"][0]
            host.photometric_redshift_err = hosts["host_redshift_std"][0]

    except Exception as err:
        error = err
    else:
        error = None
    finally:
        # Cleanup Prost file cache
        rmtree(output_dir, ignore_errors=True)
        if error:
            raise error

    return host
