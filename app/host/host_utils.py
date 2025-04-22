from datetime import datetime, timezone, timedelta
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
# TODO: Where is this config variable used? Is this obsolete code?
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
from .models import ExternalRequest
from .object_store import ObjectStore

from host.log import get_logger
logger = get_logger(__name__)


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
    flux = flux*10 ** (-0.4 * filter.ab_offset)
    flux_error = fluxerr_to_mJy_fluxerr(uncalibrated_flux_err, zpt)
    flux_error = flux_error*10 ** (-0.4 * filter.ab_offset)

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


def get_local_aperture_size(redshift,apr_kpc=2):
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

        # copy the steps to build segmentation map
        # Download FITS file local file cache
        if not os.path.isfile(local_fits_path):
            s3 = ObjectStore()
            object_key = os.path.join(settings.S3_BASE_PATH, local_fits_path.strip('/'))
            s3.download_object(path=object_key, file_path=local_fits_path)
        assert os.path.isfile(local_fits_path)
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

        # Delete FITS file from local file cache
        os.remove(local_fits_path)

    return is_contam


def select_cutout_aperture(cutouts, choice=0):
    """
    Select cutout for aperture
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

    # choice = 0
    # edited to allow initial offset
    filter_choice = filter_names[choice]

    while not cutouts.filter(filter__name=filter_choice).filter(~Q(fits="")).exists():
        choice += 1
        filter_choice = filter_names[choice]

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

    qs = ExternalRequest.objects.filter(name="NED")
    if not len(qs):
        ExternalRequest.objects.create(
            name="NED", last_query=datetime.now(timezone.utc)
        )
        try:
            result_table = Ned.query_region(position, radius=1.0 * u.arcsec)
        except ExpatError:
            raise RuntimeError("too many requests to NED")
    else:
        count = 0
        NED_TIME_SLEEP = 2
        current_time = datetime.now(timezone.utc)
        last_query = qs[0].last_query
        while (
            current_time - last_query < timedelta(seconds=NED_TIME_SLEEP)
            and count < NED_TIME_SLEEP * 100
        ):
            print(f"NED rate limit avoidance ({last_query}: sleeping iteration #{count})")
            time.sleep(NED_TIME_SLEEP)
            current_time = datetime.now(timezone.utc)
            count += 1
        else:
            try:
                result_table = Ned.query_region(position, radius=1.0 * u.arcsec)
            except ExpatError:
                raise RuntimeError("too many requests to NED")
            er = ExternalRequest.objects.get(name="NED")
            er.last_query = datetime.now(timezone.utc)
            er.save()

    result_table = result_table[result_table["Redshift"].mask == False]  # noqa: E712

    redshift = result_table["Redshift"].value

    if len(redshift):
        pos = SkyCoord(result_table["RA"].value, result_table["DEC"].value, unit=u.deg)
        sep = position.separation(pos).arcsec
        iBest = np.where(sep == np.min(sep))[0][0]

        galaxy_data = {"redshift": redshift[iBest]}
    else:
        galaxy_data = {"redshift": None}

    return galaxy_data


def query_sdss(position):
    """Get a Galaxy's redshift from SDSS if it is available"""
    result_table = SDSS.query_region(position, spectro=True, radius=1.0 * u.arcsec)

    if result_table is not None and "z" in result_table.keys():
        redshift = result_table["z"].value
        if len(redshift) > 0:
            if not math.isnan(redshift[0]):
                galaxy_data = {"redshift": redshift[0]}
            else:
                galaxy_data = {"redshift": None}
        else:
            galaxy_data = {"redshift": None}
    else:
        galaxy_data = {"redshift": None}

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
