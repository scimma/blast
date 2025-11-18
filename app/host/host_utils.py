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

from .models import Cutout
from .models import Aperture
from .object_store import ObjectStore
from pathlib import Path
from .models import TaskLock
from uuid import uuid4
from shutil import rmtree
from app.celery import app
from .models import Status
from .models import TaskRegister

from host.log import get_logger
logger = get_logger(__name__)

uuid_regex = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
ARCSEC_DEC_IN_DEG = 0.0002778  # 1 arcsecond declination in degrees
ARCSEC_RA_IN_DEG = 0.004167  # 1 arcsecond right ascension in degrees


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
    for local_fits_path in [
        global_aperture_phot.aperture.cutout.fits.name,
        aperture_primary.cutout.fits.name,
    ]:
        # UV photons are too sparse, segmentation map
        # builder cannot easily handle these
        if "/GALEX/" in local_fits_path:
            continue

        # Download FITS file local file cache
        if not os.path.isfile(local_fits_path):
            s3 = ObjectStore()
            object_key = os.path.join(settings.S3_BASE_PATH, local_fits_path.strip('/'))
            s3.download_object(path=object_key, file_path=local_fits_path)
        assert os.path.isfile(local_fits_path)
        # Create a lock file to prevent concurrent processes from deleting the data file prematurely
        lock_path = f'''{local_fits_path}.check_global_contamination.lock'''
        Path(lock_path).touch(exist_ok=True)
        assert os.path.isfile(lock_path)

        err_to_raise = None
        try:
            # copy the steps to build segmentation map
            image = fits.open(local_fits_path)
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
        except Exception as err:
            err_to_raise = err
            pass
        finally:
            os.remove(lock_path)
            if not [Path(local_fits_path).parent.glob('*.lock')]:
                try:
                    # Delete FITS file from local file cache
                    os.remove(local_fits_path)
                except FileNotFoundError:
                    pass
            if err_to_raise:
                raise err_to_raise
    return is_contam


def select_cutout_aperture(cutouts, choice=0):
    """
    Select cutout for aperture by searching through the available filters.
    """
    filter_names = [
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

    # Start iterating through the subset of filters in the list staring at the index specified by "choice".
    filter_choice = filter_names[choice:]
    for filter_name in filter_names[choice:]:
        cutout_qs = cutouts.filter(filter__name=filter_name).filter(~Q(fits=""))
        if cutout_qs.exists():
            logger.debug(f'''Cutouts for filter "{filter_name}": {[str(cutout) for cutout in cutout_qs]}''')
            filter_choice = filter_name
            break

    return cutouts.filter(filter__name=filter_choice)


def select_aperture(transient):
    cutouts = Cutout.objects.filter(transient=transient).filter(~Q(fits=""))
    if len(cutouts):
        cutout_for_aperture = select_cutout_aperture(cutouts)
    if len(cutouts) and len(cutout_for_aperture):
        global_aperture = Aperture.objects.filter(
            type__exact="global", transient=transient, cutout=cutout_for_aperture[0]
        )
    else:
        global_aperture = Aperture.objects.none()

    return global_aperture


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


def wait_for_free_space():
    # Wait until enough scratch space is available before launching the workflow tasks.
    for scratch_root in [settings.CUTOUT_ROOT, settings.SED_OUTPUT_ROOT]:
        while True:
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
