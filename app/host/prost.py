import os
from shutil import rmtree
from django.conf import settings
from django.db.models import Q
from .models import Host
from astropy.coordinates import SkyCoord
from astropy import units as u
# Prost dependencies
import pandas as pd
from scipy.stats import gamma, halfnorm, uniform
# from astropy.cosmology import LambdaCDM
from astro_prost.helpers import SnRateAbsmag
from astro_prost.associate import associate_sample
from host.host_utils import ARCSEC_DEC_IN_DEG, ARCSEC_RA_IN_DEG
from host.log import get_logger
logger = get_logger(__name__)


def update_redshifts(host, host_redshift_info, host_redshift_mean, host_redshift_std, catalog_name):
    """Update redshift null values only; do not change existing values"""
    if host_redshift_info == 'SPEC':
        host.redshift = host.redshift if host.redshift else host_redshift_mean
        host.redshift_err = host.redshift_err if host.redshift_err else host_redshift_std
    elif host_redshift_info == 'PHOT' and catalog_name != 'panstarrs':
        host.photometric_redshift = host.photometric_redshift if host.photometric_redshift else host_redshift_mean
        host.photometric_redshift_err = host.photometric_redshift_err if host.photometric_redshift_err else host_redshift_std  # noqa: E501
    return host


def name_from_coords(ra_deg, dec_deg):
    """
    Generate a name from position coordinates using the International Celestial Reference Frame (ICRF)
    and standard J2000 epoch notation.
    """
    host_coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit='deg')
    name = (f'''J{host_coord.ra.to_string(unit=u.hour, precision=2, sep='', pad=True)}'''
            f'''{host_coord.dec.to_string(unit=u.degree, precision=2, sep='', alwayssign=True, pad=True)}''')
    return name


def run_prost(transient, output_dir_root=settings.PROST_OUTPUT_ROOT):
    """Uses Prost to identify the likely host galaxy for the input transient."""
    transient_catalog = pd.DataFrame({
        'IAUID': [transient.name],
        'RA': [transient.sky_coord.ra.deg],
        'Dec': [transient.sky_coord.dec.deg]
    })
    # define priors for properties
    priors = {
        "offset": uniform(loc=0, scale=5),
        "absmag": uniform(loc=-30, scale=20)
    }
    # add the redshift info from the transient if it exists
    if transient.redshift is not None:
        priors['redshift'] = halfnorm(loc=0.0001, scale=0.5)
        transient_catalog['redshift'] = transient.redshift

    # Define and create output file root directory
    output_dir = os.path.join(output_dir_root, transient.name)
    os.makedirs(output_dir, exist_ok=True)

    # If host matching throws an unhandled exception, raise it to let the TaskRunner
    # catch it and mark the task status as failed.
    try:
        hosts = associate_sample(
            transient_catalog,
            catalogs=["glade", "decals", "panstarrs", "skymapper"],
            coord_cols=("RA", "Dec"),
            priors=priors,
            likes={
                "offset": gamma(a=0.75),
                "absmag": SnRateAbsmag(a=-25, b=20),
            },
            verbose=0,
            parallel=False,
            save=True,
            save_path=output_dir,
            cat_cols=False,
            progress_bar=False,
        )
    finally:
        # Cleanup scratch file cache
        # Note: Depending on the nature of the volume backing PROST_OUTPUT_ROOT,
        #       over time temp files may accumulate that are not deleted
        #       due to processes aborted before this block is executed. More robust
        #       garbage collection may be necessary.
        rmtree(output_dir, ignore_errors=True)
    result = {'host': None, 'error': '', 'new': False}
    try:
        catalog_name = hosts["best_cat"][0]
        catalog_release = hosts["best_cat_release"][0]
    except KeyError as err:
        result['error'] = f'Host matcher did not return expected catalog info fields: {err}'
        return result
    # If no catalog info was provided, there was no match.
    if not catalog_name or not catalog_release:
        return result
    # If a match was found, the following fields should be available:
    try:
        name = hosts["host_name"][0]
        object_id = hosts["host_objID"][0]
        ra_deg = hosts["host_ra"][0]
        dec_deg = hosts["host_dec"][0]
        host_redshift_info = hosts['host_redshift_info'][0]
        host_redshift_mean = hosts["host_redshift_mean"][0]
        host_redshift_std = hosts["host_redshift_std"][0]
    except KeyError as err:
        result['error'] = f'Host matcher did not return expected data fields: {err}'
        return result
    # If a name is not supplied, generate a name based on the position.
    if not name:
        name = name_from_coords(ra_deg, dec_deg)
    # If the host already exists in the database, use it instead of creating a new Host object.
    cone_search = (Q(ra_deg__gte=ra_deg - ARCSEC_RA_IN_DEG)
                   & Q(ra_deg__lte=ra_deg + ARCSEC_RA_IN_DEG)
                   & Q(dec_deg__gte=dec_deg - ARCSEC_DEC_IN_DEG)
                   & Q(dec_deg__lte=dec_deg + ARCSEC_DEC_IN_DEG))
    #
    # CASE A: There is an existing object with the same catalog info (object ID, catalog name & release)
    #
    catalog_info_search = (Q(object_id__exact=object_id)
                           & Q(catalog_name__exact=catalog_name)
                           & Q(catalog_release__exact=catalog_release))
    matching_catalog_info = Host.objects.filter(catalog_info_search)
    if matching_catalog_info:
        matching_position = matching_catalog_info.filter(cone_search)
        # CASE A1: Existing object is at same position
        #          If existing object name is empty, populate with match's name.
        #          Return existing object.
        if matching_position:
            logger.debug(f'''{len(matching_position)} existing hosts were found within an arcsecond of '''
                         f'''host "{name}" ("{object_id}" from catalog "{catalog_name}", release "{catalog_release}")'''
                         ''' returned by host matcher.''')
            host = matching_position[0]
            # Update the host name field if it is empty
            host.name = host.name if host.name else name
            result['host'] = host
            return result
        # CASE A2: Existing object with same non-empty object ID is at different position.
        #          This should not occur. Treat as error and fail.
        if object_id:
            result['error'] = 'Existing Host object with matching catalog info has different position.'
            return result
        # CASE A3: Existing object has a blank object ID and is at different position.
        #          This should be treated as a new Host object.
        else:
            host = Host(ra_deg=ra_deg, dec_deg=dec_deg, name=name, object_id=object_id,
                        catalog_name=catalog_name, catalog_release=catalog_release)
            result['new'] = True
            result['host'] = update_redshifts(host, host_redshift_info, host_redshift_mean, host_redshift_std,
                                              catalog_name)
            return result
    #
    # CASE B: There is an existing object at the same position but not with the same catalog info.
    #
    matching_position = Host.objects.filter(cone_search)
    if matching_position:
        logger.debug(f'''{len(matching_position)} existing hosts were found within an arcsecond of '''
                     f'''host "{name}" ("{object_id}" from catalog "{catalog_name}", release "{catalog_release}") '''
                     '''returned by host matcher.''')
        no_catalog_info = matching_position.filter((Q(catalog_name__exact='') | Q(catalog_name__isnull=True))
                                                   & (Q(catalog_release__exact='') | Q(catalog_release__isnull=True)))
        # CASE B1: Existing object has no catalog info.
        #          If existing object name is empty, populate with match's name.
        #          Populate with match's catalog info and return the object.
        if no_catalog_info:
            host = no_catalog_info[0]
            host.name = host.name if host.name else name
            host.object_id = object_id
            host.catalog_name = catalog_name
            host.catalog_release = catalog_release
            result['host'] = update_redshifts(host, host_redshift_info, host_redshift_mean, host_redshift_std,
                                              catalog_name)
            return result

        # CASE B2: Existing object with different object ID from same catalog release is at the same position.
        #          This should not occur. Treat as error and fail.
        catalog_match = matching_position.filter(Q(catalog_name__exact=catalog_name)
                                                 & Q(catalog_release__exact=catalog_release))
        if catalog_match:
            result['error'] = 'Existing Host object with matching position and catalog release has different object ID.'
            return result
    else:
        logger.debug('''No existing hosts were found within an arcsecond of '''
                     f'''host "{name}" ("{object_id}" from catalog "{catalog_name}", release "{catalog_release}") '''
                     '''returned by host matcher.''')
    #
    # CASE C: There is no existing host with consistent catalog info and position.
    #
    host = Host(ra_deg=ra_deg, dec_deg=dec_deg, name=name, object_id=object_id,
                catalog_name=catalog_name, catalog_release=catalog_release)
    result['new'] = True
    result['host'] = update_redshifts(host, host_redshift_info, host_redshift_mean, host_redshift_std, catalog_name)
    return result
