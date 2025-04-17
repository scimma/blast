import os
from shutil import rmtree
from astro_ghost.ghostHelperFunctions import getGHOST
from astro_ghost.ghostHelperFunctions import getTransientHosts
from astro_ghost.photoz_helper import calc_photoz
from astropy.coordinates import SkyCoord
from django.conf import settings

from .models import Host

# prost dependencies
import pandas as pd
from scipy.stats import gamma, halfnorm, uniform
from astropy.cosmology import LambdaCDM
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.coordinates import SkyCoord
from astropy import units as u
from astro_prost.helpers import SnRateAbsmag
from astro_prost.associate import associate_sample

def run_prost(transient):
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


    verbose = 1

    # define priors for properties
    priorfunc_z = halfnorm(loc=0.0001, scale=0.5)
    priorfunc_offset = uniform(loc=0, scale=5)
    priorfunc_absmag = uniform(loc=-30, scale=20)

    cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

    transient_names = ['2024aeyj']

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
        {'IAUID':[transient.name],
         'RA':[transient_position.ra.deg],
         'Dec':[transient_position.dec.deg]
         }
    )
    # add the redshift info from the transient
    # if it exists
    if transient.redshift is not None:
        priors['redshift'] = priorfunc_z
        transient_catalog['redshift'] = transient.redshift
    
    catalogs = ["glade", "decals", "panstarrs"]
    transient_coord_cols = ("RA", "Dec")
    transient_name_col = "IAUID"
    verbose = 0
    parallel = False
    save = True
    progress_bar = False
    cat_cols = False

    hosts = associate_sample(
        transient_catalog,
        coord_cols=transient_coord_cols,
        priors=priors,
        likes=likes,
        catalogs=catalogs,
        parallel=parallel,
        save=save,
        progress_bar=progress_bar,
        cat_cols=cat_cols,
    )
    import pdb; pdb.set_trace()
    host = Host(
        ra_deg=hosts["host_ra"][0],
        dec_deg=hosts["host_dec"][0],
        name=hosts["host_name"][0],
    )
    if hosts['best_cat'][0] != 'panstarrs' :
        host.redshift = host_data["host_redshift_mean"][0]
        
    #if host_data["NED_redshift"][0] == host_data["NED_redshift"][0]:


    #if "photo_z" in host_data.keys() and host_data["photo_z"][0] == host_data["photo_z"][0]:
    #    host.photometric_redshift = host_data["photo_z"][0]

    
    return host
