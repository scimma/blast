import os
from shutil import rmtree
from django.conf import settings
from django.db.models import Q
from .models import Host
from astropy.coordinates import SkyCoord
# Prost dependencies
import pandas as pd
from scipy.stats import gamma, halfnorm, uniform
# from astropy.cosmology import LambdaCDM
from astro_prost.helpers import SnRateAbsmag
from astro_prost.associate import associate_sample
from host.host_utils import ARCSEC_DEC_IN_DEG, ARCSEC_RA_IN_DEG
from host.log import get_logger
logger = get_logger(__name__)


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

    # If host matching throws an unhandled exception, raise it to let the TaskRunner
    # catch it and mark the task status as failed.
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
    finally:
        # Cleanup scratch file cache
        # Note: Depending on the nature of the volume backing PROST_OUTPUT_ROOT,
        #       over time temp files may accumulate that are not deleted
        #       due to processes aborted before this block is executed. More robust
        #       garbage collection may be necessary.
        rmtree(output_dir, ignore_errors=True)
    try:
        name = hosts["host_name"][0]
        obj_id = hosts["host_objID"][0]
        ra_deg = hosts["host_ra"][0]
        dec_deg = hosts["host_dec"][0]
    except KeyError:
        return None
    # If the name field is empty, use the object ID as the name instead
    if not name:
        name = obj_id
    # If a match was found, use a name and cone search to avoid duplicating Host objects
    existing_host, conflict = find_existing_host(transient, name, ra_deg, dec_deg)
    # If no existing host was found, create a new Host object
    if not existing_host:
        host = Host(ra_deg=ra_deg, dec_deg=dec_deg, name=name)
        if hosts['host_redshift_info'][0] == 'SPEC':
            host.redshift = hosts["host_redshift_mean"][0]
            host.redshift_err = hosts["host_redshift_std"][0]
        elif hosts['host_redshift_info'][0] == 'PHOT' and hosts['best_cat'][0] != 'panstarrs':
            host.photometric_redshift = hosts["host_redshift_mean"][0]
            host.photometric_redshift_err = hosts["host_redshift_std"][0]
        return host
    # If there is an existing host in the same location, return it
    if not conflict:
        return existing_host
    # If there is an existing host in a different location, generate a unique name for the match
    host_search = True
    idx = 0
    while host_search:
        idx += 1
        new_host_name = f'''{name if name else "host"}_{idx}'''
        host_search = Host.objects.filter(name__exact=new_host_name)
    new_host = Host(ra_deg=ra_deg, dec_deg=dec_deg, name=new_host_name)
    return new_host


def find_existing_host(transient, host_name, ra_deg, dec_deg):
    """Identify existing hosts nearby the same location"""
    # TODO: Deduplicate code in this function by refactoring host.host_utils.process_transient_dataset()
    cone_search = (Q(ra_deg__gte=ra_deg - ARCSEC_RA_IN_DEG)
                   & Q(ra_deg__lte=ra_deg + ARCSEC_RA_IN_DEG)
                   & Q(dec_deg__gte=dec_deg - ARCSEC_DEC_IN_DEG)
                   & Q(dec_deg__lte=dec_deg + ARCSEC_DEC_IN_DEG))
    proximate_hosts = Host.objects.filter(cone_search)
    if proximate_hosts:
        logger.info(f'''{len(proximate_hosts)} existing hosts were found within an arcsecond of '''
                    f'''host "{host_name}" returned by host matcher.''')
    host = None
    conflict = False
    # If there is an existing proximate host for an unnamed host, claim this is the same host
    if not host_name and proximate_hosts:
        return proximate_hosts[0], conflict
    if not host_name:
        return host, conflict
    # Find existing hosts with the same name
    host_search = Host.objects.filter(name__exact=host_name)
    if host_search:
        # If the host name matches, require that the position overlaps
        proximity_search = host_search.filter(cone_search)
        # Consider the import a failure if there is an inconsistent host definition
        if not proximity_search:
            logger.warning(f'[{transient.name}] Host with matching name "{host_name}" '
                           f'exists, but it is in a different location.')
            conflict = True
            return host_search[0], conflict
        # If the name and location match, claim this is the same host
        return proximity_search[0], conflict
    return None, False
