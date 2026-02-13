import json
from datetime import datetime, timezone
from django.core import serializers

import os
import math
import time
import warnings
from collections import namedtuple
from xml.parsers.expat import ExpatError

import astropy.units as u
import numpy as np
import yaml
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.ipac.ned import Ned
from astroquery.sdss import SDSS

cosmo = FlatLambdaCDM(H0=70, Om0=0.315)

from django.conf import settings
from django.db.models import Q
from dustmaps.sfd import SFDQuery

# Use correct dustmap data directory
from dustmaps.config import config
config.reset()
config["data_dir"] = settings.DUSTMAPS_DATA_ROOT
from photutils.aperture import aperture_photometry
from photutils.aperture import EllipticalAperture
from photutils.background import Background2D
from photutils.segmentation import detect_sources
from photutils.segmentation import SourceCatalog
from photutils.utils import calc_total_error
from photutils.background import LocalBackground
from photutils.background import MeanBackground, SExtractorBackground
from astropy.stats import SigmaClip

from .photometric_calibration import flux_to_mag
from .photometric_calibration import flux_to_mJy_flux
from .photometric_calibration import fluxerr_to_magerr
from .photometric_calibration import fluxerr_to_mJy_fluxerr

from host.models import Aperture
from host.models import Host
from host.models import AperturePhotometry
from host.models import Cutout
from host.models import Filter
from host.models import SEDFittingResult
from host.models import TaskRegister
from host.models import Transient
from host.models import Survey
from host.models import StarFormationHistoryResult

from .object_store import ObjectStore
from .models import TaskLock
from uuid import uuid4
from shutil import rmtree
from app.celery import app
from .models import Status
from .base_tasks import initialize_all_tasks_status

from host.log import get_logger
logger = get_logger(__name__)

uuid_regex = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
ARCSEC_DEC_IN_DEG = 0.0002778  # 1 arcsecond declination in degrees
ARCSEC_RA_IN_DEG = 0.004167  # 1 arcsecond right ascension in degrees
FILTER_NAMES = [
    "PanSTARRS_g",
    "PanSTARRS_r",
    "PanSTARRS_i",
    "SDSS_r",
    "SDSS_i",
    "SDSS_g",
    "DES_r",
    "DES_i",
    "DES_g",
    "2MASS_H",
]


def survey_list(survey_metadata_path):
    """
    Build a list of survey objects from a metadata file.
    Parameters
    ----------
    :survey_metadata_path : str
        Path to a yaml data file containing survey metadata
    Returns
    -------
    :list of surveys: list[Survey]
        List of survey objects
    """
    with open(survey_metadata_path, "r") as stream:
        survey_metadata = yaml.safe_load(stream)

    # get first survey from the metadata in order to infer the data field names
    survey_name = list(survey_metadata.keys())[0]
    data_fields = list(survey_metadata[survey_name].keys())

    # create a named tuple class with all the survey data fields as attributes
    # including the survey name
    Survey = namedtuple("Survey", ["name"] + data_fields)

    survey_list = []
    for name in survey_metadata:
        field_dict = {field: survey_metadata[name][field] for field in data_fields}
        field_dict["name"] = name
        survey_list.append(Survey(**field_dict))

    return survey_list


def build_source_catalog(image, background, threshhold_sigma=3.0, npixels=10):
    """
    Constructs a source catalog given an image and background estimation
    Parameters
    ----------
    :image :  :class:`~astropy.io.fits.HDUList`
        Fits image to construct source catalog from.
    :background : :class:`~photutils.background.Background2D`
        Estimate of the background in the image.
    :threshold_sigma : float default=2.0
        Threshold sigma above the baseline that a source has to be to be
        detected.
    :n_pixels : int default=10
        The length of the size of the box in pixels used to perform segmentation
        and de-blending of the image.
    Returns
    -------
    :source_catalog : :class:`photutils.segmentation.SourceCatalog`
        Catalog of sources constructed from the image.
    """

    image_data = image[0].data
    background_subtracted_data = image_data - background.background
    threshold = threshhold_sigma * background.background_rms

    segmentation = detect_sources(
        background_subtracted_data, threshold, npixels=npixels
    )
    if segmentation is None:
        return None
    # deblended_segmentation = deblend_sources(
    #     background_subtracted_data, segmentation, npixels=npixels
    # )
    logger.debug(segmentation)
    return SourceCatalog(background_subtracted_data, segmentation)


def match_source(position, source_catalog, wcs):
    """
    Match the source in the source catalog to the host position
    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        On Sky position of the source to be matched.
    :source_catalog : :class:`~photutils.segmentation.SourceCatalog`
        Catalog of sources.
    :wcs : :class:`~astropy.wcs.WCS`
        World coordinate system to match the sky position to the
        source catalog.
    Returns
    -------
    :source : :class:`~photutils.segmentation.SourceCatalog`
        Catalog containing the one matched source.
    """

    host_x_pixel, host_y_pixel = wcs.world_to_pixel(position)
    source_x_pixels, source_y_pixels = (
        source_catalog.xcentroid,
        source_catalog.ycentroid,
    )
    closest_source_index = np.argmin(
        np.hypot(host_x_pixel - source_x_pixels, host_y_pixel - source_y_pixels)
    )

    return source_catalog[closest_source_index]


def elliptical_sky_aperture(source_catalog, wcs, aperture_scale=3.0):
    """
    Constructs an elliptical sky aperture from a source catalog
    Parameters
    ----------
    :source_catalog: :class:`~photutils.segmentation.SourceCatalog`
        Catalog containing the source to get aperture information from.
    :wcs : :class:`~astropy.wcs.WCS`
        World coordinate system of the source catalog.
    :aperture_scale: float default=3.0
        Scale factor to increase the size of the aperture
    Returns
    -------
    :sky_aperture: :class:`~photutils.aperture.SkyEllipticalAperture`
        Elliptical sky aperture of the source in the source catalog.
    """
    center = (source_catalog.xcentroid, source_catalog.ycentroid)
    semi_major_axis = source_catalog.semimajor_sigma.value * aperture_scale
    semi_minor_axis = source_catalog.semiminor_sigma.value * aperture_scale
    orientation_angle = source_catalog.orientation.to(u.rad).value
    pixel_aperture = EllipticalAperture(
        center, semi_major_axis, semi_minor_axis, theta=orientation_angle
    )
    pixel_aperture = source_catalog.kron_aperture
    return pixel_aperture.to_sky(wcs)


def do_aperture_photometry(image, sky_aperture, filter):
    """
    Performs Aperture photometry
    """
    image_data = image[0].data
    wcs = WCS(image[0].header)

    # get the background
    try:
        background = estimate_background(image, filter.name)
    except ValueError:
        # indicates poor image data
        return {
            "flux": None,
            "flux_error": None,
            "magnitude": None,
            "magnitude_error": None,
        }

    # is the aperture inside the image?
    bbox = sky_aperture.to_pixel(wcs).bbox
    if (
        bbox.ixmin < 0
        or bbox.iymin < 0
        or bbox.ixmax > image_data.shape[1]
        or bbox.iymax > image_data.shape[0]
    ):
        return {
            "flux": None,
            "flux_error": None,
            "magnitude": None,
            "magnitude_error": None,
        }

    # if the image pixels are all zero, let's assume this is masked
    # even GALEX FUV should have *something*
    phot_table_maskcheck = aperture_photometry(image_data, sky_aperture, wcs=wcs)
    if phot_table_maskcheck["aperture_sum"].value[0] == 0:
        return {
            "flux": None,
            "flux_error": None,
            "magnitude": None,
            "magnitude_error": None,
        }

    background_subtracted_data = image_data - background.background

    # I think we need a local background subtraction for WISE
    # the others haven't given major problems
    if "WISE" in filter.name:
        aper_pix = sky_aperture.to_pixel(wcs)
        lbg = LocalBackground(aper_pix.a, aper_pix.a * 2)
        local_background = lbg(
            background_subtracted_data, aper_pix.positions[0], aper_pix.positions[1]
        )
        background_subtracted_data -= local_background

    if filter.image_pixel_units == "counts/sec":
        error = calc_total_error(
            background_subtracted_data,
            background.background_rms,
            float(image[0].header["EXPTIME"]),
        )

    else:
        error = calc_total_error(
            background_subtracted_data, background.background_rms, 1.0
        )

    phot_table = aperture_photometry(
        background_subtracted_data, sky_aperture, wcs=wcs, error=error
    )
    uncalibrated_flux = phot_table["aperture_sum"].value[0]
    if "2MASS" not in filter.name:
        uncalibrated_flux_err = phot_table["aperture_sum_err"].value[0]
    else:
        # 2MASS is annoying
        # https://wise2.ipac.caltech.edu/staff/jarrett/2mass/3chan/noise/
        n_pix = (
            np.pi
            * sky_aperture.a.value
            * sky_aperture.b.value
            * filter.pixel_size_arcsec**2.0
        )
        uncalibrated_flux_err = np.sqrt(
            uncalibrated_flux / (10 * 6)
            + 4 * n_pix * 1.7**2.0 * np.median(background.background_rms) ** 2
        )

    # check for correlated errors
    aprad, err_adjust = filter.correlation_model()
    if aprad is not None:
        image_aperture = sky_aperture.to_pixel(wcs)

        err_adjust_interp = np.interp(
            (image_aperture.a + image_aperture.b) / 2.0, aprad, err_adjust
        )
        uncalibrated_flux_err *= err_adjust_interp

    if filter.magnitude_zero_point_keyword is not None:
        zpt = image[0].header[filter.magnitude_zero_point_keyword]
    elif filter.image_pixel_units == "counts/sec":
        zpt = filter.magnitude_zero_point
    else:
        zpt = filter.magnitude_zero_point + 2.5 * np.log10(image[0].header["EXPTIME"])

    flux = flux_to_mJy_flux(uncalibrated_flux, zpt)
    flux = flux * 10 ** (-0.4 * filter.ab_offset)
    flux_error = fluxerr_to_mJy_fluxerr(uncalibrated_flux_err, zpt)
    flux_error = flux_error * 10 ** (-0.4 * filter.ab_offset)

    magnitude = flux_to_mag(uncalibrated_flux, zpt)
    magnitude_error = fluxerr_to_magerr(uncalibrated_flux, uncalibrated_flux_err)
    if magnitude != magnitude:
        magnitude, magnitude_error = None, None
    if flux != flux or flux_error != flux_error:
        flux, flux_error = None, None
    # wave_eff = filter.transmission_curve().wave_effective
    return {
        "flux": flux,
        "flux_error": flux_error,
        "magnitude": magnitude,
        "magnitude_error": magnitude_error,
    }


def get_dust_maps(position):
    """Gets milkyway reddening value"""

    ebv = SFDQuery()(position)
    # see Schlafly & Finkbeiner 2011 for the 0.86 correction term
    return 0.86 * ebv


def get_local_aperture_size(redshift, apr_kpc=2):
    """find the size of a 2 kpc radius in arcsec"""

    dadist = cosmo.angular_diameter_distance(redshift).value
    apr_arcsec = apr_kpc / (
        dadist * 1000 * (np.pi / 180.0 / 3600.0)
    )  # 2 kpc aperture radius is this many arcsec

    return apr_arcsec


def check_local_radius(redshift, image_fwhm_arcsec):
    """Checks whether filter image FWHM is larger than
    the aperture size"""

    dadist = cosmo.angular_diameter_distance(redshift).value
    apr_arcsec = 2 / (
        dadist * 1000 * (np.pi / 180.0 / 3600.0)
    )  # 2 kpc aperture radius is this many arcsec

    return "true" if apr_arcsec > image_fwhm_arcsec else "false"


def check_global_contamination(global_aperture_phot, aperture_primary):
    """Checks whether aperture is contaminated by multiple objects"""
    warnings.simplefilter("ignore")
    is_contam = False
    aperture = global_aperture_phot.aperture
    # check both the image used to generate aperture
    # and the image used to measure photometry
    for cutout_fits_path in [
        global_aperture_phot.aperture.cutout.fits.name,
        aperture_primary.cutout.fits.name,
    ]:
        # UV photons are too sparse, segmentation map
        # builder cannot easily handle these
        if "/GALEX/" in cutout_fits_path:
            continue

        # Download FITS file to local scratch space
        s3 = ObjectStore()
        local_tmp_path = os.path.join('/tmp', cutout_fits_path.strip('/').replace('/', '__'))
        if os.path.exists(local_tmp_path):
            # Use a unique name to avoid collisions with concurrent processes
            local_tmp_path = f'''{local_tmp_path}.{str(uuid4())}'''
        object_key = os.path.join(settings.S3_BASE_PATH, cutout_fits_path.strip('/'))
        s3.download_object(path=object_key, file_path=local_tmp_path)
        assert os.path.isfile(local_tmp_path)

        try:
            # copy the steps to build segmentation map
            image = fits.open(local_tmp_path)
            wcs = WCS(image[0].header)
            background = estimate_background(image)
            catalog = build_source_catalog(
                image, background, threshhold_sigma=5, npixels=15
            )

            # catalog is None is no sources are detected in the image
            # so we don't have to worry about contamination in that case
            if catalog is None:
                continue

            source_data = match_source(aperture.sky_coord, catalog, wcs)

            mask_image = (
                aperture.sky_aperture.to_pixel(wcs)
                .to_mask()
                .to_image(np.shape(image[0].data))
            )
            obj_ids = catalog._segment_img.data[np.where(mask_image == True)]  # noqa: E712
            source_obj = source_data._labels

            # let's look for contaminants
            unq_obj_ids = np.unique(obj_ids)
            if len(unq_obj_ids[(unq_obj_ids != 0) & (unq_obj_ids != source_obj)]):
                is_contam = True
        finally:
            try:
                # Delete FITS file from local file cache
                os.remove(local_tmp_path)
            except FileNotFoundError:
                pass
    return is_contam


def select_best_cutout(transient_name):
    cutouts = Cutout.objects.filter(transient__name__exact=transient_name).filter(~Q(fits=""))
    # Choose the cutout from the available filters using the priority define in select_cutout_aperture()
    cutout = None
    cutout_set = select_cutout_aperture(cutouts).filter(~Q(fits=""))
    if len(cutout_set):
        cutout = cutout_set[0]
    return cutout


def select_cutout_aperture(cutouts, choice=0):
    """
    Select cutout for aperture by searching through the available filters.
    """
    # Start iterating through the subset of filters in the list staring at the index specified by "choice".
    filter_choice = FILTER_NAMES[choice:]
    for filter_name in FILTER_NAMES[choice:]:
        cutout_qs = cutouts.filter(filter__name=filter_name).filter(~Q(fits=""))
        if cutout_qs.exists():
            logger.debug(f'''Cutouts for filter "{filter_name}": {[str(cutout) for cutout in cutout_qs]}''')
            filter_choice = filter_name
            break

    return cutouts.filter(filter__name=filter_choice)


def select_aperture(transient):
    '''Select the best Aperture object for the input transient.
       Returns a QuerySet of Aperture objects.'''
    # Find all cutouts that have an associated FITS image file
    cutouts = Cutout.objects.filter(transient=transient).filter(~Q(fits=""))
    # If no cutout images are found, return an empty Aperture set
    if not len(cutouts):
        return Aperture.objects.none()
    cutout_for_aperture = select_cutout_aperture(cutouts)
    if len(cutout_for_aperture):
        return Aperture.objects.filter(type__exact="global", transient=transient, cutout=cutout_for_aperture[0])
    return Aperture.objects.none()


def estimate_background(image, filter_name=None):
    """
    Estimates the background of an image
    Parameters
    ----------
    :image : :class:`~astropy.io.fits.HDUList`
        Image to have the background estimated of.
    Returns
    -------
    :background : :class:`~photutils.background.Background2D`
        Background estimate of the image
    """
    image_data = image[0].data
    box_size = int(0.1 * np.sqrt(image_data.size))

    # GALEX needs mean, not median - median just always comes up with zero
    if filter_name is not None and "GALEX" in filter_name:
        bkg = MeanBackground(SigmaClip(sigma=3.0))
    else:
        bkg = SExtractorBackground(sigma_clip=None)

    try:
        return Background2D(image_data, box_size=box_size, bkg_estimator=bkg)
    except ValueError:
        return Background2D(
            image_data, box_size=box_size, exclude_percentile=50, bkg_estimator=bkg
        )


def construct_aperture(image, position):
    """
    Construct an elliptical aperture at the position in the image
    Parameters
    ----------
    :image : :class:`~astropy.io.fits.HDUList`
    Returns
    -------
    """
    wcs = WCS(image[0].header)
    background = estimate_background(image)

    # found an edge case where deblending isn't working how I'd like it to
    # so if it's not finding the host, play with the default threshold
    def get_source_data(threshhold_sigma):
        catalog = build_source_catalog(
            image, background, threshhold_sigma=threshhold_sigma
        )
        source_data = match_source(position, catalog, wcs)

        source_ra, source_dec = wcs.wcs_pix2world(
            source_data.xcentroid, source_data.ycentroid, 0
        )
        source_position = SkyCoord(source_ra, source_dec, unit=u.deg)
        source_separation_arcsec = position.separation(source_position).arcsec
        return source_data, source_separation_arcsec

    iter = 0
    source_separation_arcsec = 100
    while source_separation_arcsec > 5 and iter < 5:
        source_data, source_separation_arcsec = get_source_data(5 * (iter + 1))
        iter += 1
    # look for sub-threshold sources
    # if we still can't find the host
    if source_separation_arcsec > 5:
        source_data, source_separation_arcsec = get_source_data(2)

    # make sure we know this failed
    if source_separation_arcsec > 5:
        return None

    return elliptical_sky_aperture(source_data, wcs)


def query_ned(position):
    """Get a Galaxy's redshift from NED if it is available."""

    timeout = settings.QUERY_TIMEOUT
    time_start = time.time()
    logger.debug('''Aquiring NED query lock...''')
    while timeout > time.time() - time_start:
        if TaskLock.objects.request_lock('ned_query'):
            break
        logger.debug('''Waiting to aquire NED query lock...''')
        time.sleep(1)

    galaxy_data = {"redshift": None}
    try:
        result_table = Ned.query_region(position, radius=1.0 * u.arcsec)
        result_table = result_table[result_table["Redshift"].mask == False]  # noqa: E712
        redshift = result_table["Redshift"].value
        if len(redshift):
            pos = SkyCoord(result_table["RA"].value, result_table["DEC"].value, unit=u.deg)
            sep = position.separation(pos).arcsec
            iBest = np.where(sep == np.min(sep))[0][0]
            galaxy_data = {"redshift": redshift[iBest]}
        assert not math.isnan(galaxy_data['redshift'])
    except ExpatError as err:
        logger.error(f"Too many requests to NED: {err}")
        raise RuntimeError("Too many requests to NED")
    except Exception as err:
        logger.warning(f'''Error obtaining redshift from NED: {err}''')
    finally:
        # Release the NED query lock
        logger.debug('''Releasing NED query lock...''')
        TaskLock.objects.release_lock('ned_query')

    return galaxy_data


def query_sdss(position):
    """Get a Galaxy's redshift from SDSS if it is available"""

    timeout = settings.QUERY_TIMEOUT
    time_start = time.time()
    logger.debug('''Aquiring SDSS query lock...''')
    while timeout > time.time() - time_start:
        if TaskLock.objects.request_lock('SDSS_query'):
            break
        logger.debug('''Waiting to aquire SDSS query lock...''')
        time.sleep(1)
    galaxy_data = {"redshift": None}
    try:
        result_table = SDSS.query_region(position, spectro=True, radius=1.0 * u.arcsec)
        redshift = result_table["z"].value
        assert not math.isnan(redshift[0])
        galaxy_data["redshift"] = redshift[0]
    except Exception as err:
        logger.warning(f'''Error obtaining redshift from SDSS: {err}''')
    finally:
        # Release the SDSS query lock
        logger.debug('''Releasing SDSS query lock...''')
        TaskLock.objects.release_lock('sdss_query')

    return galaxy_data


def construct_all_apertures(position, image_dict):
    apertures = {}

    for name, image in image_dict.items():
        try:
            aperture = construct_aperture(image, position)
            apertures[name] = aperture
        except Exception:
            logger.warning(f"Could not fit aperture to {name} imaging data")

    return apertures


def pick_largest_aperture(position, image_dict):
    """
    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        On Sky position of the source which aperture is to be measured.
    :image_dic: dict[str:~astropy.io.fits.HDUList]
        Dictionary of images from different surveys, key is the the survey
        name.
    Returns
    -------
    :largest_aperture: dict[str:~photutils.aperture.SkyEllipticalAperture]
        Dictionary of contain the image with the largest aperture, key is the
         name of the survey.
    """

    apertures = {}

    for name, image in image_dict.items():
        try:
            aperture = construct_aperture(image, position)
            apertures[name] = aperture
        except Exception:
            logger.warning(f"Could not fit aperture to {name} imaging data")

    aperture_areas = {}
    for image_name in image_dict:
        aperture_semi_major_axis = apertures[image_name].a
        aperture_semi_minor_axis = apertures[image_name].b
        aperture_area = np.pi * aperture_semi_minor_axis * aperture_semi_major_axis
        aperture_areas[image_name] = aperture_area

    max_size_name = max(aperture_areas, key=aperture_areas.get)
    return {max_size_name: apertures[max_size_name]}


def get_directory_size(directory):
    """Returns the `directory` size in bytes."""
    total = 0
    # Avoid returning zero length for symlinks
    directory = os.path.realpath(directory)
    try:
        # print("[+] Getting the size of", directory)
        for entry in os.scandir(directory):
            if entry.is_file():
                # if it's a file, use stat() function
                total += entry.stat().st_size
            elif entry.is_dir():
                # if it's a directory, recursively call this function
                try:
                    total += get_directory_size(entry.path)
                except FileNotFoundError:
                    pass
    except NotADirectoryError:
        # if `directory` isn't a directory, get the file size then
        return os.path.getsize(directory)
    except PermissionError:
        # if for whatever reason we can't open the folder, return 0
        return 0
    return total


def get_job_scratch_prune_lock_id(scratch_root):
    '''Create or parse a lock file containing a unique ID for the scratch root directory.
       In conjunction with the TaskLock global mutex, this lock file prevents more than
       a single process from attempting to prune scratch files therein.'''
    # Determine worker ID from "prune_lock.yaml" file
    semaphore_path = os.path.join(scratch_root, 'prune_lock.yaml')
    try:
        with open(semaphore_path, 'r') as meta_file:
            metadata = yaml.load(meta_file, Loader=yaml.SafeLoader)
        worker_id = metadata['id']
    except Exception as err:
        logger.warning(f'Assuming missing or invalid scratch metadata file; creating new one. (Error: {err})')
        worker_id = str(uuid4())
        with open(semaphore_path, 'w') as meta_file:
            yaml.dump({'id': worker_id}, meta_file)
    return f'prune_scratch_files_{worker_id}'


def calculate_units(size):
    '''Return the best units to express the file size along with the rounded integer in those units.'''
    units = 'bytes'
    size_in_units = size
    if size > 1024**4:
        units = 'TiB'
        size_in_units = round(size / 1024**4, 0)
    elif size > 1024**3:
        units = 'GiB'
        size_in_units = round(size / 1024**3, 0)
    elif size > 1024**2:
        units = 'MiB'
        size_in_units = round(size / 1024**2, 0)
    return units, size_in_units


def wait_for_free_space(force_prune=False):
    # Wait until enough scratch space is available before launching the workflow tasks.
    for scratch_root in [settings.CUTOUT_ROOT, settings.SED_OUTPUT_ROOT]:
        while True:
            if not force_prune:
                # Calculate size of /scratch to determine free space. If JOB_SCRATCH_MAX_SIZE is finite, calculate free
                # space using the supplied value; otherwise, attempt to calculate using statvfs.
                scratch_total = settings.JOB_SCRATCH_MAX_SIZE
                if scratch_total:
                    scratch_used = get_directory_size(scratch_root)
                    scratch_free = scratch_total - scratch_used
                else:
                    # See https://pubs.opengroup.org/onlinepubs/009695399/basedefs/sys/statvfs.h.html
                    # and https://docs.python.org/3/library/os.html#os.statvfs
                    statvfs = os.statvfs(scratch_root)
                    # Capacity of filesystem in bytes
                    scratch_total = statvfs.f_frsize * statvfs.f_blocks
                    # Number of free bytes available to non-privileged process.
                    scratch_free = statvfs.f_frsize * statvfs.f_bavail
                free_percentage = round(100.0 * scratch_free / scratch_total)
                logger.debug(f'''Job scratch free space is {scratch_free} bytes ({free_percentage}%) '''
                             f'''for a scratch volume capacity of {scratch_total} bytes.''')
                # If there is sufficient free scratch space, stop waiting
                if scratch_free > settings.JOB_SCRATCH_FREE_SPACE:
                    return
                # If there is insufficient free space, attempt to delete orphaned data
                logger.info(f'''Insufficient free scratch space {round(scratch_free / 1024**2)} MiB '''
                            f'''({free_percentage}%). Pruning scratch files...''')
            lock_id = get_job_scratch_prune_lock_id(scratch_root)
            if TaskLock.objects.request_lock(lock_id):
                prune_workflow_scratch_dirs(scratch_root)
                TaskLock.objects.release_lock(lock_id)
                break
            else:
                # Give the existing prune operation time to complete before recalculating free disk space.
                logger.debug('''Waiting for running prune operation to complete...''')
                time.sleep(20)


def inspect_worker_tasks():
    inspect = app.control.inspect()
    all_worker_tasks = []
    # Query the task workers to collect the name and arguments for all the queued and active tasks.
    for inspect_func in [inspect.active, inspect.scheduled, inspect.reserved]:
        try:
            items = inspect_func(safe=True).items()
        except AttributeError:
            items = []
        tasks = [task for worker, worker_tasks in items for task in worker_tasks]
        all_worker_tasks.extend([{'name': task['name'], 'args': str(task['args'])} for task in tasks])
    return all_worker_tasks


def prune_workflow_scratch_dirs(root_path, dry_run=False):
    # List scratch directories first to ensure that a subsequently launched transient
    # workflow's scratch files cannot be accidentally deleted.
    scratch_dirs = os.listdir(root_path)
    all_tasks = [task for task in inspect_worker_tasks()]
    total_size = 0
    for transient_name in scratch_dirs:
        scratch_path = os.path.join(os.path.realpath(root_path), transient_name)
        dir_size = get_directory_size(scratch_path)
        total_size += dir_size
        # If there is an active or queued task with the transient's name as the argument,
        # the transient workflow is still processing.
        if [task for task in all_tasks if task['args'].find(transient_name) >= 0]:
            logger.debug(f'''["{transient_name}"] is queued or active. Skipping.''')
            continue
        # If the workflow is not queued or active, purge the scratch files.
        log_msg = f'''["{transient_name}"] Purging scratch files ({dir_size} bytes): "{scratch_path}"'''
        if dry_run:
            logger.info(f'''{log_msg} [dry-run]''')
        else:
            logger.info(log_msg)
            rmtree(scratch_path, ignore_errors=True)


def reset_workflow_if_not_processing(transient, worker_tasks, reset_failed=False):
    # If there is an active or queued task with the transient's name as the argument,
    # the transient workflow is still processing.
    if [task for task in worker_tasks if task['args'].find(transient.name) >= 0]:
        logger.debug(f'''Workflow for transient "{transient.name}" is queued or running.''')
        return False
    logger.debug(f'''Detected stalled workflow for transient "{transient.name}". '''
                 '''Resetting "processing" statuses to "not processed"...''')
    # Reset any workflow tasks with errant "processing" status to "not processed" so they will be executed.
    processing_status = Status.objects.get(message__exact="processing")
    not_processed_status = Status.objects.get(message__exact="not processed")
    processing_tasks = [task for task in TaskRegister.objects.filter(transient__exact=transient)
                        if task.status == processing_status]
    failed_tasks = []
    if reset_failed:
        failed_tasks = [task for task in TaskRegister.objects.filter(transient__exact=transient)
                        if task.status.type == 'error']
    for task in processing_tasks + failed_tasks:
        task.status = not_processed_status
        task.save()
    return True


def create_or_update_aperture(query, data):
    aperture_name = data['name']
    aperture_query = Aperture.objects.filter(**query)
    if aperture_query:
        logger.debug(f'''Aperture object "{data['name']}" exists. Updating with new data...''')
        Aperture.objects.filter(name__exact=aperture_name).update(**data)
        aperture = Aperture.objects.get(**query)
    else:
        logger.debug(f'''Aperture object "{data['name']}" does not exist. Creating it and '''
                     '''manually associating it with related objects...''')
        aperture, created = Aperture.objects.get_or_create(**data)
        assert created
        SEDFittingResult.objects.filter(aperture__name__exact=aperture.name).update(aperture=aperture)
        AperturePhotometry.objects.filter(aperture__name__exact=aperture.name).update(aperture=aperture)
        StarFormationHistoryResult.objects.filter(aperture__name__exact=aperture.name).update(aperture=aperture)
    return aperture


def delete_transient(transient_name='', transient=None):
    err_msg = ''
    if not transient:
        try:
            transient = Transient.objects.get(name__exact=transient_name)
        except Transient.DoesNotExist:
            err_msg = 'Transient does not exist in the database.'
            return err_msg
    try:
        # TODO: This manual deletion of associated objects should not be necessary, due to the use of
        #       "models.ForeignKey(X, on_delete=models.CASCADE)" in the model definitions. However, without
        #       this logic, there are foriegn key constraint violations in the database transactions.
        for aperture in Aperture.objects.filter(transient=transient):
            # Delete SEDFittingResult objects but not their associated data files
            SEDFittingResult.objects.filter(aperture__name__exact=aperture.name).delete()
            AperturePhotometry.objects.filter(aperture__name__exact=aperture.name).delete()
            StarFormationHistoryResult.objects.filter(aperture__name__exact=aperture.name).delete()
            aperture.delete()
        # Delete Cutout objects but not their associated data files
        for cutout in Cutout.objects.filter(transient=transient):
            cutout.delete()
        transient.delete()
    except Exception as err:
        err_msg = str(err)
    return err_msg


def export_transient_info(transient_name=''):
    '''Export all data associated with a transient sufficient to import into another Blast instance.'''
    def prune_fields(data_object, model_name):
        if model_name == 'transient':
            data_object['fields'].pop('tasks_initialized')
            data_object['fields'].pop('progress')
            data_object['fields'].pop('added_by')
        return data_object

    transient_data = {
        'metadata': {
            'app_version': f'v{settings.APP_VERSION}',
            'export_time': datetime.now(timezone.utc).isoformat(),
        },
        'transient': {},
        'host': None,
        'apertures': [],
        'cutouts': [],
        'filters': json.loads(serializers.serialize("json", Filter.objects.all())),
        'surveys': json.loads(serializers.serialize("json", Survey.objects.all())),
        'workflow_tasks': [],
    }
    try:
        transient_obj = Transient.objects.filter(name__exact=transient_name)
    except Transient.DoesNotExist:
        return {}
    transient_obj = Transient.objects.filter(name__exact=transient_name)
    transient = json.loads(serializers.serialize("json", transient_obj))
    if not transient_obj:
        return {}
    transient_obj = transient_obj[0]
    transient = transient[0]
    assert isinstance(transient, dict)
    # Export intrinsic transient data
    transient_data['transient'] = prune_fields(transient, 'transient')
    # Export host information
    if transient_obj.host:
        transient_data['host'] = json.loads(serializers.serialize("json", [transient_obj.host]))[0]
    # Export cutout image data
    cutouts = json.loads(serializers.serialize("json", Cutout.objects.filter(transient__name__exact=transient_name)))
    assert isinstance(cutouts, list)
    transient_data['cutouts'] = cutouts
    # Export aperture-related data
    apertures = json.loads(serializers.serialize(
        "json", Aperture.objects.filter(transient__name__exact=transient_name)))
    assert isinstance(apertures, list)
    transient_data['apertures'] = apertures
    for aperture in transient_data['apertures']:
        aperture['sedfittingresults'] = json.loads(serializers.serialize(
            "json", SEDFittingResult.objects.filter(aperture__name__exact=aperture['fields']['name']))),
        if len(aperture['sedfittingresults']) == 1 and isinstance(aperture['sedfittingresults'][0], list):
            aperture['sedfittingresults'] = aperture['sedfittingresults'][0]
        aperture['aperturephotometry'] = json.loads(serializers.serialize(
            "json", AperturePhotometry.objects.filter(aperture__name__exact=aperture['fields']['name']))),
        if len(aperture['aperturephotometry']) == 1 and isinstance(aperture['aperturephotometry'][0], list):
            aperture['aperturephotometry'] = aperture['aperturephotometry'][0]
        aperture['starformationhistoryresult'] = json.loads(serializers.serialize(
            "json", StarFormationHistoryResult.objects.filter(aperture__name__exact=aperture['fields']['name']))),
        if len(aperture['starformationhistoryresult']) == 1 and isinstance(aperture['starformationhistoryresult'][0], list):  # noqa
            aperture['starformationhistoryresult'] = aperture['starformationhistoryresult'][0]
    for tr in TaskRegister.objects.filter(transient__name=transient_name):
        transient_data['workflow_tasks'].append({
            'task_name': tr.task.name,
            'status': {
                'type': tr.status.type,
                'message': tr.status.message,
            },
            'user_warning': tr.user_warning,
            'last_modified': tr.last_modified,
            'last_processing_time_seconds': tr.last_processing_time_seconds,
        })
    return transient_data


def import_transient_info(transient_data_json):
    '''Import all data associated with a transient from a Blast export file.'''
    logger.debug(transient_data_json)
    transient_data = json.load(transient_data_json)
    # Construct the import list
    if isinstance(transient_data, dict):
        datasets_to_import = [transient_data]
    elif isinstance(transient_data, list):
        datasets_to_import = transient_data
    assert isinstance(datasets_to_import, list)

    imported_transient_names = []
    import_failures = []

    def record_import_error(transient_name, err_msg=''):
        import_failures.append({
            'transient_name': transient_name,
            'err_msg': err_msg,
        })

    def process_transient_dataset(dataset):
        # Verify that the transient is not already present (by name)
        transient_name = dataset['transient']['fields']['name']
        if Transient.objects.filter(name__exact=transient_name):
            record_import_error(transient_name, f'Transient "{transient_name}" already exists')
            return
        # Verify that Survey objects exist and are identical.
        for survey in dataset['surveys']:
            survey_name = survey['fields']['name']
            try:
                Survey.objects.get(name__exact=survey_name)
            except Survey.DoesNotExist:
                record_import_error(transient_name, f'[{transient_name}] Survey "{survey_name}" does not exist.')
                return
        # Verify that Filter objects exist and are identical.
        for filter in dataset['filters']:
            filter_name = filter['fields']['name']
            try:
                filter_obj = Filter.objects.get(name__exact=filter_name)
            except Filter.DoesNotExist:
                record_import_error(transient_name,
                                    f'[{transient_name}] Filter "{filter_name}" does not exist.')
                return
            try:
                logger.debug(f'''Importing filter:\n{json.dumps(filter['fields'], indent=2)}''')
                debug_filter_obj_dict = {
                    'survey': filter_obj.survey.name,
                    'kcorrect_name': filter_obj.kcorrect_name,
                    'sedpy_id': filter_obj.sedpy_id,
                    'hips_id': filter_obj.hips_id,
                    'vosa_id': filter_obj.vosa_id,
                    'image_download_method': filter_obj.image_download_method,
                    'pixel_size_arcsec': filter_obj.pixel_size_arcsec,
                    'image_fwhm_arcsec': filter_obj.image_fwhm_arcsec,
                    'wavelength_eff_angstrom': filter_obj.wavelength_eff_angstrom,
                    'wavelength_min_angstrom': filter_obj.wavelength_min_angstrom,
                    'wavelength_max_angstrom': filter_obj.wavelength_max_angstrom,
                    'vega_zero_point_jansky': filter_obj.vega_zero_point_jansky,
                    'magnitude_zero_point': filter_obj.magnitude_zero_point,
                    'ab_offset': filter_obj.ab_offset,
                    'magnitude_zero_point_keyword': filter_obj.magnitude_zero_point_keyword,
                    'image_pixel_units': filter_obj.image_pixel_units,
                }
                logger.debug(f'''Existing filter:\n{json.dumps(debug_filter_obj_dict, indent=2)}''')
                # The survey names associated with the existing and importing filter should match
                assert filter_obj.survey.name == [survey['fields']['name'] for survey in dataset['surveys']
                                                  if survey['pk'] == filter['fields']['survey']][0]
                assert filter_obj.kcorrect_name == filter['fields']['kcorrect_name']
                assert filter_obj.sedpy_id == filter['fields']['sedpy_id']
                assert filter_obj.hips_id == filter['fields']['hips_id']
                assert filter_obj.vosa_id == filter['fields']['vosa_id']
                assert filter_obj.image_download_method == filter['fields']['image_download_method']
                assert filter_obj.pixel_size_arcsec == filter['fields']['pixel_size_arcsec']
                assert filter_obj.image_fwhm_arcsec == filter['fields']['image_fwhm_arcsec']
                assert filter_obj.wavelength_eff_angstrom == filter['fields']['wavelength_eff_angstrom']
                assert filter_obj.wavelength_min_angstrom == filter['fields']['wavelength_min_angstrom']
                assert filter_obj.wavelength_max_angstrom == filter['fields']['wavelength_max_angstrom']
                assert filter_obj.vega_zero_point_jansky == filter['fields']['vega_zero_point_jansky']
                assert filter_obj.magnitude_zero_point == filter['fields']['magnitude_zero_point']
                assert filter_obj.ab_offset == filter['fields']['ab_offset']
                assert filter_obj.magnitude_zero_point_keyword == filter['fields']['magnitude_zero_point_keyword']
                assert filter_obj.image_pixel_units == filter['fields']['image_pixel_units']
            except AssertionError as err:
                record_import_error(transient_name, f'[{transient_name}] Filter named '
                                    f'"{filter_obj.name}" exists but is not identical to the import: {err}')
                return
        # Verify that if the Host exists (by name), that it is identical.
        # TODO: How concerned should we be about duplicates? Should we perform a cone search instead of assuming
        #       perfect coordinate matching? Should the redshift values be updated from the imported data if they are
        #       missing?
        host_name = dataset['host']['fields']['name']
        ra_deg = dataset['host']['fields']['ra_deg']
        dec_deg = dataset['host']['fields']['dec_deg']
        cone_search = (Q(ra_deg__gte=ra_deg - ARCSEC_RA_IN_DEG)
                       & Q(ra_deg__lte=ra_deg + ARCSEC_RA_IN_DEG)
                       & Q(dec_deg__gte=dec_deg - ARCSEC_DEC_IN_DEG)
                       & Q(dec_deg__lte=dec_deg + ARCSEC_DEC_IN_DEG))
        proximate_hosts = Host.objects.filter(cone_search)
        if proximate_hosts:
            logger.info(f'''{len(proximate_hosts)} existing hosts were found within an arcsecond of '''
                        f'''importing host "{host_name}".''')
        host = None
        # If there is an existing proximate host for an unnamed host, claim this is the same host
        if not host_name and proximate_hosts:
            host = proximate_hosts[0]
        elif host_name:
            # Find existing hosts with the same name
            host_search = Host.objects.filter(name__exact=host_name)
            if host_search:
                # If the host name matches, require that the position overlaps
                proximity_search = host_search.filter(cone_search)
                # Consider the import a failure if there is an inconsistent host definition
                if not proximity_search:
                    record_import_error(transient_name,
                                        f'[{transient_name}] Host with matching name "{host_name}" '
                                        f'exists, but it is in a different location.')
                    return
                # If the name and location match, claim this is the same host
                host = proximity_search[0]
        # If no host match was found, create a new Host object
        if not host:
            host = Host.objects.create(
                ra_deg=dataset['host']['fields']['ra_deg'],
                dec_deg=dataset['host']['fields']['dec_deg'],
                name=dataset['host']['fields']['name'],
                redshift=dataset['host']['fields']['redshift'],
                redshift_err=dataset['host']['fields']['redshift_err'],
                photometric_redshift=dataset['host']['fields']['photometric_redshift'],
                photometric_redshift_err=dataset['host']['fields']['photometric_redshift_err'],
                milkyway_dust_reddening=dataset['host']['fields']['milkyway_dust_reddening'],
                software_version=dataset['host']['fields']['software_version'],
            )
        # Verify that the Cutout objects do not exist (by name).
        for cutout in dataset['cutouts']:
            cutout_name = cutout['fields']['name']
            if Cutout.objects.filter(name__exact=cutout_name).exists():
                record_import_error(transient_name,
                                    f'[{transient_name}] Cutout "{cutout_name}" exists.')
                return
        # Verify that each Aperture object does not exist.
        for aperture in dataset['apertures']:
            aperture_name = aperture['fields']['name']
            if Aperture.objects.filter(name__exact=aperture_name).exists():
                record_import_error(transient_name,
                                    f'[{transient_name}] Aperture "{aperture_name}" exists.')
                return
            # Ignore any orphaned StarFormationHistoryResult objects that may exist because they will not interfere.
            # Ignore any orphaned AperturePhotometry objects that may exist because they will not interfere.
            # Ignore any orphaned SEDFittingResult objects that may exist because they will not interfere.

        # Create Transient object
        transient = Transient.objects.create(
            ra_deg=dataset['transient']['fields']['ra_deg'],
            dec_deg=dataset['transient']['fields']['dec_deg'],
            name=dataset['transient']['fields']['name'],
            display_name=dataset['transient']['fields']['display_name'],
            tns_id=dataset['transient']['fields']['tns_id'],
            tns_prefix=dataset['transient']['fields']['tns_prefix'],
            public_timestamp=dataset['transient']['fields']['public_timestamp'],
            host=host,
            redshift=dataset['transient']['fields']['redshift'],
            spectroscopic_class=dataset['transient']['fields']['spectroscopic_class'],
            photometric_class=dataset['transient']['fields']['photometric_class'],
            milkyway_dust_reddening=dataset['transient']['fields']['milkyway_dust_reddening'],
            processing_status=dataset['transient']['fields']['processing_status'],
            software_version=dataset['transient']['fields']['software_version'],
        )
        # Create Cutout objects
        for cutout in dataset['cutouts']:
            filter_name = [filter['fields']['name'] for filter in dataset['filters']
                           if filter['pk'] == cutout['fields']['filter']][0]
            Cutout.objects.create(
                name=cutout['fields']['name'],
                filter=Filter.objects.get(name__exact=filter_name),
                transient=transient,
                fits=cutout['fields']['fits'],
                message=cutout['fields']['message'],
                software_version=cutout['fields']['software_version'],
                cropped=cutout['fields']['cropped'],
            )
        # For each Aperture object,
            # Create Aperture object
            # Create StarFormationHistoryResult objects
            # Create AperturePhotometry objects
            # Create SEDFittingResult objects
        for aperture in dataset['apertures']:
            logger.debug(f'''Aperture cutout pk: {aperture['fields']['cutout']}''')
            cutout_name_search = [cutout['fields']['name'] for cutout in dataset['cutouts']
                                  if cutout['pk'] == aperture['fields']['cutout']]
            if cutout_name_search:
                cutout_obj = Cutout.objects.get(name__exact=cutout_name_search[0])
            else:
                cutout_obj = None
            aperture_obj = Aperture.objects.create(
                ra_deg=aperture['fields']['ra_deg'],
                dec_deg=aperture['fields']['dec_deg'],
                name=aperture['fields']['name'],
                cutout=cutout_obj,
                transient=transient,
                orientation_deg=aperture['fields']['orientation_deg'],
                semi_major_axis_arcsec=aperture['fields']['semi_major_axis_arcsec'],
                semi_minor_axis_arcsec=aperture['fields']['semi_minor_axis_arcsec'],
                type=aperture['fields']['type'],
                software_version=aperture['fields']['software_version'],
            )
            for aperturephotometry in aperture['aperturephotometry']:
                filter_name = [filter['fields']['name'] for filter in dataset['filters']
                               if filter['pk'] == aperturephotometry['fields']['filter']][0]
                AperturePhotometry.objects.create(
                    aperture=aperture_obj,
                    filter=Filter.objects.get(name__exact=filter_name),
                    transient=transient,
                    flux=aperturephotometry['fields']['flux'],
                    flux_error=aperturephotometry['fields']['flux_error'],
                    magnitude=aperturephotometry['fields']['magnitude'],
                    magnitude_error=aperturephotometry['fields']['magnitude_error'],
                    is_validated=aperturephotometry['fields']['is_validated'],
                    software_version=aperturephotometry['fields']['software_version'],
                )
            # Compile a dictionary of StarFormationHistoryResult objects indexed by their primary key values for
            # subsequent association with SEDFittingResult objects.
            sfh_objs = {}
            for starformationhistoryresult in aperture['starformationhistoryresult']:
                sfh_objs[starformationhistoryresult['pk']] = StarFormationHistoryResult.objects.create(
                    aperture=aperture_obj,
                    transient=transient,
                    logsfr_16=starformationhistoryresult['fields']['logsfr_16'],
                    logsfr_50=starformationhistoryresult['fields']['logsfr_50'],
                    logsfr_84=starformationhistoryresult['fields']['logsfr_84'],
                    logsfr_tmin=starformationhistoryresult['fields']['logsfr_tmin'],
                    logsfr_tmax=starformationhistoryresult['fields']['logsfr_tmax'],
                    software_version=starformationhistoryresult['fields']['software_version'],
                )
            for sedfittingresults in aperture['sedfittingresults']:
                # logsfh = []
                # for sfh_pk in sedfittingresults['fields']['logsfh']:
                #     logsfh.append([sfh for key, sfh in sfh_objs.items() if key == sfh_pk][0])
                # logsfh = [[sfh for key, sfh in sfh_objs.items() if key == sfh_pk][0]
                #           for sfh_pk in sedfittingresults['fields']['logsfh']]
                sedfittingresult = SEDFittingResult.objects.create(
                    aperture=aperture_obj,
                    transient=transient,
                    posterior=sedfittingresults['fields']['posterior'],
                    log_mass_16=sedfittingresults['fields']['log_mass_16'],
                    log_mass_50=sedfittingresults['fields']['log_mass_50'],
                    log_mass_84=sedfittingresults['fields']['log_mass_84'],
                    mass_surviving_ratio=sedfittingresults['fields']['mass_surviving_ratio'],
                    log_sfr_16=sedfittingresults['fields']['log_sfr_16'],
                    log_sfr_50=sedfittingresults['fields']['log_sfr_50'],
                    log_sfr_84=sedfittingresults['fields']['log_sfr_84'],
                    log_ssfr_16=sedfittingresults['fields']['log_ssfr_16'],
                    log_ssfr_50=sedfittingresults['fields']['log_ssfr_50'],
                    log_ssfr_84=sedfittingresults['fields']['log_ssfr_84'],
                    log_age_16=sedfittingresults['fields']['log_age_16'],
                    log_age_50=sedfittingresults['fields']['log_age_50'],
                    log_age_84=sedfittingresults['fields']['log_age_84'],
                    log_tau_16=sedfittingresults['fields']['log_tau_16'],
                    log_tau_50=sedfittingresults['fields']['log_tau_50'],
                    log_tau_84=sedfittingresults['fields']['log_tau_84'],
                    logzsol_16=sedfittingresults['fields']['logzsol_16'],
                    logzsol_50=sedfittingresults['fields']['logzsol_50'],
                    logzsol_84=sedfittingresults['fields']['logzsol_84'],
                    dust2_16=sedfittingresults['fields']['dust2_16'],
                    dust2_50=sedfittingresults['fields']['dust2_50'],
                    dust2_84=sedfittingresults['fields']['dust2_84'],
                    dust_index_16=sedfittingresults['fields']['dust_index_16'],
                    dust_index_50=sedfittingresults['fields']['dust_index_50'],
                    dust_index_84=sedfittingresults['fields']['dust_index_84'],
                    dust1_fraction_16=sedfittingresults['fields']['dust1_fraction_16'],
                    dust1_fraction_50=sedfittingresults['fields']['dust1_fraction_50'],
                    dust1_fraction_84=sedfittingresults['fields']['dust1_fraction_84'],
                    log_fagn_16=sedfittingresults['fields']['log_fagn_16'],
                    log_fagn_50=sedfittingresults['fields']['log_fagn_50'],
                    log_fagn_84=sedfittingresults['fields']['log_fagn_84'],
                    log_agn_tau_16=sedfittingresults['fields']['log_agn_tau_16'],
                    log_agn_tau_50=sedfittingresults['fields']['log_agn_tau_50'],
                    log_agn_tau_84=sedfittingresults['fields']['log_agn_tau_84'],
                    gas_logz_16=sedfittingresults['fields']['gas_logz_16'],
                    gas_logz_50=sedfittingresults['fields']['gas_logz_50'],
                    gas_logz_84=sedfittingresults['fields']['gas_logz_84'],
                    duste_qpah_16=sedfittingresults['fields']['duste_qpah_16'],
                    duste_qpah_50=sedfittingresults['fields']['duste_qpah_50'],
                    duste_qpah_84=sedfittingresults['fields']['duste_qpah_84'],
                    duste_umin_16=sedfittingresults['fields']['duste_umin_16'],
                    duste_umin_50=sedfittingresults['fields']['duste_umin_50'],
                    duste_umin_84=sedfittingresults['fields']['duste_umin_84'],
                    log_duste_gamma_16=sedfittingresults['fields']['log_duste_gamma_16'],
                    log_duste_gamma_50=sedfittingresults['fields']['log_duste_gamma_50'],
                    log_duste_gamma_84=sedfittingresults['fields']['log_duste_gamma_84'],
                    chains_file=sedfittingresults['fields']['chains_file'],
                    percentiles_file=sedfittingresults['fields']['percentiles_file'],
                    model_file=sedfittingresults['fields']['model_file'],
                    software_version=sedfittingresults['fields']['software_version'],
                )
                # Collect subset of StarFormationHistoryResult objects matching the list of primary key values
                sedfittingresult.logsfh.set([[sfh for key, sfh in sfh_objs.items() if key == sfh_pk][0]
                                             for sfh_pk in sedfittingresults['fields']['logsfh']])
        initialize_all_tasks_status(transient)
        all_trs = TaskRegister.objects.filter(transient__name=transient_name)
        for tr in dataset['workflow_tasks']:
            task_name = tr['task_name']
            try:
                tr_obj = all_trs.get(task__name=task_name)
            except TaskRegister.DoesNotExist:
                record_import_error(transient_name,
                                    f'[{transient_name}] Workflow task "{task_name}" does not exist.')
                return
            tr_obj.status = Status.objects.get(message=tr['status']['message'], type=tr['status']['type'])
            tr_obj.user_warning = tr['user_warning']
            tr_obj.last_modified = tr['last_modified']
            tr_obj.last_processing_time_seconds = tr['last_processing_time_seconds']
            tr_obj.save()
        # TODO: Install data files.
        imported_transient_names.append(transient.name)

    # Construct the database objects for each transient.
    for dataset in datasets_to_import:
        # Use a nested function to support aborting upon error within nested for loops.
        process_transient_dataset(dataset)

    # Delete database objects associated with failed imports
    for import_failure in import_failures:
        delete_transient(transient_name=import_failure['transient_name'])

    return imported_transient_names, import_failures
