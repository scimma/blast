import glob
import os

from astro_ghost.ghostHelperFunctions import getGHOST
from astro_ghost.ghostHelperFunctions import getTransientHosts
from astro_ghost.photoz_helper import calc_photoz
from astropy.coordinates import SkyCoord
from django.conf import settings

from .models import Host


def run_ghost(transient, output_dir=settings.GHOST_OUTPUT_ROOT):
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
    getGHOST(real=False, verbose=1)
    transient_position = SkyCoord(
        ra=transient.ra_deg, dec=transient.dec_deg, unit="deg"
    )

    # dumb hack for ghost
    try:
        float(transient.name)
        transient_name = "sn" + str(transient.name)
    except Exception as e:
        transient_name = transient.name

    # Due to a bug in astro_ghost version 2.2.2.dev19, we must ensure that
    # the output path has a trailing slash.
    output_dir = os.path.join(output_dir, '')

    # some issues with Pan-STARRS downloads
    host_data = getTransientHosts(
        transientCoord=[transient_position],
        transientName=[transient_name],
        verbose=1,
        savepath=output_dir,
        starcut="gentle",
        ascentMatch=False,
    )

    # photo-z only implemented for dec > -30
    if transient_position.dec.deg > -30:
        # still getting random photo-z bugs
        # but this shouldn't be a show-stopper
        try:
            host_data = calc_photoz(
                host_data,
                dust_path=settings.GHOST_DUST_PATH,
                model_path=settings.GHOST_PHOTOZ_PATH,
            )
        except Exception as err:
            print(f"warning : photo-z step failed: {err}")

    # clean up after GHOST...
    # dir_list = glob.glob('transients_*/*/*')
    # for dir in dir_list: os.remove(dir)

    # for level in ['*/*/', '*/']:
    #    dir_list = glob.glob('transients_' + level)
    #    for dir in dir_list: os.rmdir(dir)
    if len(host_data) == 0:
        host = None
    else:
        host = Host(
            ra_deg=host_data["raMean"][0],
            dec_deg=host_data["decMean"][0],
            name=host_data["TransientName"][0],
        )

        if host_data["NED_redshift"][0] == host_data["NED_redshift"][0]:
            host.redshift = host_data["NED_redshift"][0]

        if (
            "photo_z" in host_data.keys()
            and host_data["photo_z"][0] == host_data["photo_z"][0]
        ):
            host.photometric_redshift = host_data["photo_z"][0]

    return host
