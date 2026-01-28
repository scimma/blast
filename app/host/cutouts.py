import os
import re
import time
from io import BytesIO

import astropy.table as at
import astropy.units as u
import astropy.utils.data
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.units import Quantity
from astropy.wcs import WCS
from astroquery.hips2fits import hips2fits
from astroquery.mast import Observations
from astroquery.sdss import SDSS
from astroquery.skyview import SkyView
from django.conf import settings
from dl import authClient as ac
from dl import queryClient as qc
from dl import storeClient as sc
from pyvo.dal import sia
from host.object_store import ObjectStore
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import Cutout
from .models import Filter

from host.log import get_logger
logger = get_logger(__name__)

DOWNLOAD_SLEEP_TIME = int(os.environ.get("DOWNLOAD_SLEEP_TIME", "0"))
DOWNLOAD_MAX_TRIES = int(os.environ.get("DOWNLOAD_MAX_TRIES", "1"))

# from host import SkyServer


def getRADecBox(ra, dec, size):
    RAboxsize = DECboxsize = size

    # get the maximum 1.0/cos(DEC) term: used for RA cut
    minDec = dec - 0.5 * DECboxsize
    if minDec <= -90.0:
        minDec = -89.9
    maxDec = dec + 0.5 * DECboxsize
    if maxDec >= 90.0:
        maxDec = 89.9

    invcosdec = max(
        1.0 / np.cos(dec * np.pi / 180.0),
        1.0 / np.cos(minDec * np.pi / 180.0),
        1.0 / np.cos(maxDec * np.pi / 180.0),
    )

    ramin = ra - 0.5 * RAboxsize * invcosdec
    ramax = ra + 0.5 * RAboxsize * invcosdec
    decmin = dec - 0.5 * DECboxsize
    decmax = dec + 0.5 * DECboxsize

    if ra < 0.0:
        ra += 360.0
    if ra >= 360.0:
        ra -= 360.0

    if ramin != None:
        if (ra - ramin) < -180:
            ramin -= 360.0
            ramax -= 360.0
        elif (ra - ramin) > 180:
            ramin += 360.0
            ramax += 360.0
    return (ramin, ramax, decmin, decmax)


def download_and_save_cutouts(
    transient,
    fov=Quantity(0.1, unit="deg"),
    cutout_base_path=settings.CUTOUT_ROOT,
    overwrite=settings.CUTOUT_OVERWRITE,
    filter_set=None
):
    """
    Download all available imaging from a list of surveys
    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :survey_list : list[Survey]
        List of surveys to download data from
    :fov : :class:`~astropy.units.Quantity`,
    default=Quantity(0.1,unit='deg')
        Field of view of the cutout image, angular length of one of the sides
        of the square cutout. Angular astropy quantity. Default is angular
        length of 0.2 degrees.
    Returns
    -------
    :images dictionary : dict[str: :class:`~astropy.io.fits.HDUList`]
        Dictionary of images with the survey names as keys and fits images
        as values.
    """

    s3 = ObjectStore()

    status_failed = "failed"
    status_succeeded = "processed"

    assert isinstance(overwrite, bool)

    def download_filter_data(filter):
        # TODO: The "canonical" base path "/data/cutout_cdn" is now hard-coded in the object keys
        #       of 50,000+ transient datasets in the production instance of Blast as of 2026/01/13.
        #       For consistency we need to keep that object key base path for new transients, but
        #       the local temporary base file path can be different.
        trans_root_path = os.path.join('/tmp', transient.name)
        temp_save_dir = os.path.join(trans_root_path, filter.survey.name)
        temp_fits_path = os.path.join(temp_save_dir, f"{filter.name}.fits")
        logger.debug(f'''FITS file temp local path: {temp_fits_path}''')
        canonical_fits_path = temp_fits_path.replace('/tmp', cutout_base_path)
        object_key = os.path.join(settings.S3_BASE_PATH, canonical_fits_path.strip('/'))
        logger.debug(f'''FITS file object_key: {object_key}''')
        cutout_name = f"{transient.name}_{filter.name}"
        # Fetch or create the associated cutout object in the database.
        cutout_object, created = Cutout.objects.get_or_create(name=cutout_name, filter=filter, transient=transient)

        # If we know there is no image to download, exit.
        if cutout_object.message == "No image found":
            return status_succeeded

        # Does cutout file exist in the S3 bucket?
        cutout_file_exists = s3.object_exists(object_key)
        fits = None
        if overwrite or not cutout_file_exists or created:
            logger.debug(f'Downloading cutout "{cutout_name}"...')
            fits, status, err = cutout(transient.sky_coord, filter, fov=fov, download_max_tries=2,
                                       download_sleep_time=5)
            # If a download error occurred, report it and exit.
            if status == 1:
                logger.error(f'Download error for "{cutout_name}": {err}')
                cutout_object.message = "Download error"
                cutout_object.fits = ""
                cutout_object.save()
                return status_failed
        if fits:
            upload_succeeded = False
            # Write FITS file to local cache
            os.makedirs(temp_save_dir, exist_ok=True)
            fits.writeto(temp_fits_path, overwrite=True)
            assert os.path.isfile(temp_fits_path)
            # Upload file to bucket and delete local copy
            try:
                s3.put_object(path=object_key, file_path=temp_fits_path)
                upload_succeeded = s3.object_exists(object_key)
                assert upload_succeeded
            except AssertionError as err:
                logger.error(f'Error uploading cutout download: {err}')
            finally:
                os.remove(temp_fits_path)
            if upload_succeeded:
                cutout_object.fits.name = canonical_fits_path
                cutout_object.save()
                return status_succeeded
            else:
                cutout_object.message = "Upload failed"
                cutout_object.fits.name = ""
                cutout_object.save()
                return status_failed
        # If the FITS file exists now (whether it was (re)downloaded a moment ago or not),
        # update the database object. Otherwise record that no image was found.
        # This supports manually uploaded data files.
        if s3.object_exists(object_key):
            cutout_object.fits.name = canonical_fits_path
            cutout_object.message = ""
            cutout_object.save()
        else:
            # There is no image available.
            cutout_object.message = "No image found"
            cutout_object.fits.name = ""
            cutout_object.save()
        return status_succeeded

    try:
        results = []
        if filter_set is None:
            filter_set = Filter.objects.all()
        with ThreadPoolExecutor(max_workers=len(filter_set)) as executor:
            future_to_filter = {executor.submit(download_filter_data, filter): filter for filter in filter_set}
            for future in as_completed(future_to_filter):
                filter = future_to_filter[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (filter, exc))
                    result = status_failed
                logger.debug(f'''"{filter}" result: "{result}"''')
                results.append(result)

        # If any of the cutout downloads failed, mark the entire task as failed
        if not results or [result for result in results if result != status_succeeded]:
            return status_failed
        else:
            return status_succeeded
    except Exception as err:
        logger.error(err)
        return status_failed


def panstarrs_image_filename(position, image_size=None, filter=None):
    """Query panstarrs service to get a list of image names

    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :size : int: cutout image size in pixels.
    :filter: str: Panstarrs filter (g r i z y)
    Returns
    -------
    :filename: str: file name of the cutout
    """

    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = (
        f"{service}?ra={position.ra.degree}&dec={position.dec.degree}"
        f"&size={image_size}&format=fits&filters={filter}"
    )

    ### was having SSL errors with pandas, so let's run it through requests
    ### optionally, can edit to do this in an unsafe way
    r = requests.get(url, stream=True)
    r.raw.decode_content = True
    filename_table = pd.read_csv(r.raw, sep="\s+")["filename"]
    return filename_table[0] if len(filename_table) > 0 else None


def hips_cutout(position, survey, image_size=None):
    """
    Download fits image from hips2fits service.

    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :image_size : int: cutout image size in pixels.
    Returns
    -------
    :cutout : :class:`~astropy.io.fits.HDUList` or None
    """
    fov = Quantity(survey.pixel_size_arcsec * image_size, unit="arcsec")

    fits_image = hips2fits.query(
        hips=survey.hips_id,
        ra=position.ra,
        dec=position.dec,
        width=image_size,
        height=image_size,
        fov=fov,
        projection="TAN",
        format="fits",
    )

    # if the position is outside of the survey footprint
    if np.all(np.isnan(fits_image[0].data)):
        fits_image = None
    return fits_image


def panstarrs_cutout(position, image_size=None, filter=None):
    """
    Download Panstarrs cutout from their own service

    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :image_size: int: size of cutout image in pixels
    :filter: str: Panstarrs filter (g r i z y)
    Returns
    -------
    :cutout : :class:`~astropy.io.fits.HDUList` or None
    """

    filename = panstarrs_image_filename(position, image_size=image_size, filter=filter)
    if filename is not None:
        service = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
        fits_url = (
            f"{service}ra={position.ra.degree}&dec={position.dec.degree}"
            f"&size={image_size}&format=fits&red={filename}"
        )
        try:
            r = requests.get(fits_url, stream=True)
        except Exception as e:
            time.sleep(5)
            r = requests.get(fits_url, stream=True)
        fits_image = fits.open(BytesIO(r.content))

    else:
        fits_image = None

    return fits_image


def galex_cutout(position, image_size=None, filter=None):
    """
    Download GALEX cutout from MAST

    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :image_size: int: size of cutout image in pixels
    :filter: str: Panstarrs filter (g r i z y)
    Returns
    -------
    :cutout : :class:`~astropy.io.fits.HDUList` or None
    """

    obs = Observations.query_criteria(
        coordinates=position,
        radius="0.2 deg",
        obs_collection="GALEX",
        filters=filter,
        distance=0
    )
    #obs = obs[
    #    (obs["obs_collection"] == "GALEX")
    #    & (obs["filters"] == filter)
    #    & (obs["distance"] == 0)
    #]

    # avoid masked regions
    center = SkyCoord(obs["s_ra"], obs["s_dec"], unit=u.deg)
    sep = position.separation(center).deg
    obs = obs[sep < 0.55]

    if len(obs) > 1:
        obs = obs[obs["t_exptime"] == max(obs["t_exptime"])]

    if len(obs):
        fits_image = fits.open(
            obs["dataURL"][0]
            .replace("-exp.fits.gz", "-int.fits.gz")
            .replace("-gsp.fits.gz", "-int.fits.gz")
            .replace("-rr.fits.gz", "-int.fits.gz")
            .replace("-cnt.fits.gz", "-int.fits.gz")
            .replace("-fcat.ds9reg", "-int.fits.gz")
            .replace("-xd-mcat.fits.gz", f"-{filter[0].lower()}d-int.fits.gz"),
            cache=None,
        )

        wcs = WCS(fits_image[0].header)
        cutout = Cutout2D(fits_image[0].data, position, image_size, wcs=wcs)
        fits_image[0].data = cutout.data
        fits_image[0].header.update(cutout.wcs.to_header())
        if not np.any(fits_image[0].data):
            fits_image = None
    else:
        fits_image = None

    return fits_image


def WISE_cutout(position, image_size=None, filter=None):
    """
    Download WISE image cutout from IRSA

    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :image_size: int: size of cutout image in pixels
    :filter: str: Panstarrs filter (g r i z y)
    Returns
    -------
    :cutout : :class:`~astropy.io.fits.HDUList` or None
    """

    band_to_wavelength = {
        "W1": "3.4e-6",
        "W2": "4.6e-6",
        "W3": "1.2e-5",
        "W4": "2.2e-5",
    }

    url = f"https://irsa.ipac.caltech.edu/SIA?COLLECTION=wise_allwise&POS=circle+{position.ra.deg}+{position.dec.deg}+0.002777&RESPONSEFORMAT=CSV&BAND={band_to_wavelength[filter]}&FORMAT=image/fits"
    r = requests.get(url)
    url = None
    for t in r.text.split(","):
        if t.startswith("https"):
            url = t[:]
            break

    # remove the AWS crap messing up the CSV format
    line_out = ""
    for line in r.text.split("\n"):
        try:
            idx1 = line.index("{")
        except ValueError:
            line_out += line[:] + "\n"
            continue
        idx2 = line.index("}")
        newline = line[0 : idx1 + 1] + line[idx2:] + "\n"
        line_out += newline

    data = at.Table.read(line_out, format="ascii.csv")
    exptime = data["t_exptime"][0]

    if url is not None:
        fits_image = fits.open(url, cache=None)

        wcs = WCS(fits_image[0].header)
        cutout = Cutout2D(fits_image[0].data, position, image_size, wcs=wcs)
        fits_image[0].data = cutout.data
        fits_image[0].header.update(cutout.wcs.to_header())
        fits_image[0].header["EXPTIME"] = exptime

    else:
        fits_image = None

    return fits_image


def DES_cutout(position, image_size=None, filter=None):
    """
    Download DES image cutout from NOIRLab

    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :image_size: int: size of cutout image in pixels
    :filter: str: Panstarrs filter (g r i z y)
    Returns
    -------
    :cutout : :class:`~astropy.io.fits.HDUList` or None
    """

    DEF_ACCESS_URL = "https://datalab.noirlab.edu/sia/ls_dr9"
    svc_ls_dr9 = sia.SIAService(DEF_ACCESS_URL)

    imgTable = svc_ls_dr9.search(
        (position.ra.deg, position.dec.deg),
        (image_size / np.cos(position.dec.deg * np.pi / 180), image_size),
        verbosity=2,
    ).to_table()

    valid_urls = []
    for img in imgTable:
        if "-depth-" in img["access_url"] and img["obs_bandpass"].startswith(filter):
            valid_urls += [img["access_url"]]
            logger.debug(f'''DES image URL: {valid_urls[-1]}''')

    if len(valid_urls):
        # we need both the depth and the image
        time.sleep(1)
        try:
            fits_image = fits.open(
                valid_urls[0].replace("-depth-", "-image-"), cache=None
            )
        except Exception as e:
            ### found some bad links...
            return None
        if np.shape(fits_image[0].data)[0] == 1 or np.shape(fits_image[0].data)[1] == 1:
            # no idea what's happening here but this is a mess
            return None
        try:
            depth_image = fits.open(valid_urls[0])
        except Exception as e:
            # wonder if there's some issue with other tasks clearing the cache
            time.sleep(5)
            depth_image = fits.open(valid_urls[0])
        wcs_depth = WCS(depth_image[0].header)
        xc, yc = wcs_depth.wcs_world2pix(position.ra.deg, position.dec.deg, 0)

        # this is ugly - just assuming the exposure time at the
        # location of interest is uniform across the image
        if np.shape(depth_image[0].data) == (1, 1):
            exptime = depth_image[0].data[0][0]
        else:
            exptime = depth_image[0].data[int(yc), int(xc)]
        if exptime == 0:
            fits_image = None
        else:
            wcs = WCS(fits_image[0].header)
            cutout = Cutout2D(fits_image[0].data, position, image_size, wcs=wcs)
            fits_image[0].data = cutout.data
            fits_image[0].header.update(cutout.wcs.to_header())
            fits_image[0].header["EXPTIME"] = exptime
    else:
        fits_image = None

    return fits_image


def TWOMASS_cutout(position, image_size=None, filter=None):
    """
    Download 2MASS image cutout from IRSA

    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :image_size: int: size of cutout image in pixels
    :filter: str: Panstarrs filter (g r i z y)
    Returns
    -------
    :cutout : :class:`~astropy.io.fits.HDUList` or None
    """

    irsaquery = f"https://irsa.ipac.caltech.edu/cgi-bin/2MASS/IM/nph-im_sia?POS={position.ra.deg},{position.dec.deg}&SIZE=0.01"
    response = requests.get(url=irsaquery)

    fits_image = None
    for line in response.content.decode("utf-8").split("<TD><![CDATA["):
        if re.match(f"https://irsa.*{filter.lower()}i.*fits", line.split("]]>")[0]):
            fitsurl = line.split("]]")[0]

            fits_image = fits.open(fitsurl, cache=None)
            wcs = WCS(fits_image[0].header)

            if position.contained_by(wcs):
                break

    if fits_image is not None:
        cutout = Cutout2D(fits_image[0].data, position, image_size, wcs=wcs)
        fits_image[0].data = cutout.data
        fits_image[0].header.update(cutout.wcs.to_header())

    else:
        fits_image = None

    return fits_image


def SDSS_cutout(position, image_size=None, filter=None):
    """
    Download SDSS image cutout from astroquery

    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :image_size: int: size of cutout image in pixels
    :filter: str: Panstarrs filter (g r i z y)
    Returns
    -------
    :cutout : :class:`~astropy.io.fits.HDUList` or None
    """

    sdss_baseurl = "https://data.sdss.org/sas"
    print(position)

    xid = SDSS.query_region(position, radius=0.05 * u.deg)
    if xid is None or len(xid) == 0:
        return None
    
    image_pos = SkyCoord(xid['ra'],xid['dec'],unit=u.deg)
    sep = position.separation(image_pos)
    iSep = np.where(sep == np.min(sep))[0]
    

    # old (better, but deprecated) version
    #url = f"https://dr12.sdss.org/fields/raDec?ra={position.ra.deg}&dec={position.dec.deg}"
    #print(url)
    #rt = requests.get(url)



    regex = "<dt>run<\/dt>.*<dd>.*<\/dd>"
    run = xid['run'][iSep][0] #re.findall("<dt>run</dt>\n.*<dd>([0-9]+)</dd>", rt.text)[0]
    rerun = xid['rerun'][iSep][0] #re.findall("<dt>rerun</dt>\n.*<dd>([0-9]+)</dd>", rt.text)[0]
    camcol = xid['camcol'][iSep][0] #re.findall("<dt>camcol</dt>\n.*<dd>([0-9]+)</dd>", rt.text)[0]
    field = xid['field'][iSep][0] #re.findall("<dt>field</dt>\n.*<dd>([0-9]+)</dd>", rt.text)[0]

    # a little latency so that we don't look like a bot to SDSS?
    time.sleep(1)
    link = SDSS.IMAGING_URL_SUFFIX.format(
        base=sdss_baseurl,
        run=int(run),
        dr=16,
        instrument="eboss",
        rerun=int(rerun),
        camcol=int(camcol),
        field=int(field),
        band=filter,
    )

    fits_image = fits.open(link, cache=None)

    wcs = WCS(fits_image[0].header)
    cutout = Cutout2D(fits_image[0].data, position, image_size, wcs=wcs)
    fits_image[0].data = cutout.data
    fits_image[0].header.update(cutout.wcs.to_header())

    # else:
    #    fits_image = None

    return fits_image


download_function_dict = {
    "PanSTARRS": panstarrs_cutout,
    "GALEX": galex_cutout,
    "2MASS": TWOMASS_cutout,
    "WISE": WISE_cutout,
    "DES": DES_cutout,
    "SDSS": SDSS_cutout,
}


def cutout(transient, survey, fov=Quantity(0.1, unit="deg"), download_max_tries=DOWNLOAD_MAX_TRIES,
           download_sleep_time=DOWNLOAD_SLEEP_TIME):
    """
    Download image cutout data from a survey.
    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :survey : :class: Survey
        Named tuple containing metadata for the survey the image is to be
        downloaded from.
    :fov : :class:`~astropy.units.Quantity`,
    default=Quantity(0.2,unit='deg')
        Field of view of the cutout image, angular length of one of the sides
        of the square cutout. Angular astropy quantity. Default is angular
        length of 0.2 degrees.
    Returns
    -------
    :cutout : :class:`~astropy.io.fits.HDUList` or None
        Image cutout in fits format or if the image cannot be download due to a
        `ReadTimeoutError` None will be returned.
    """
    # need to make sure the cache doesn't overfill
    astropy.utils.data.clear_download_cache()
    num_pixels = int(fov.to(u.arcsec).value / survey.pixel_size_arcsec)

    status = 1
    n_iter = 0
    error = None
    assert download_max_tries > 0
    assert download_sleep_time >= 0
    while status == 1 and n_iter < download_max_tries:
        if n_iter > 0:
            logger.info(f'Retrying download ({n_iter}/{download_max_tries}) "{survey}"...')
        if survey.image_download_method == "hips":
            try:
                fits = hips_cutout(transient, survey, image_size=num_pixels)
                status = 0
            except Exception as err:
                print(f"Conection timed out, could not download {survey.name} data")
                fits = None
                status = 1
                error = err
        else:
            survey_name, filter = survey.name.split("_")
            try:
                fits = download_function_dict[survey_name](
                    transient, filter=filter, image_size=num_pixels
                )
                status = 0
            except Exception as err:
                print(f"Could not download {survey.name} data")
                print(f"exception: {err}")
                fits = None
                status = 1
                error = err
        n_iter += 1
        if status == 1:
            time.sleep(download_sleep_time)

    return fits, status, error
