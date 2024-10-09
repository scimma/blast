from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import astropy.units as u
import numpy as np
import os

from host.models import Cutout
from host.models import Aperture
from host.models import Transient
from host.host_utils import get_local_aperture_size


def trim_images(transient):

    cutouts = Cutout.objects.filter(transient=transient)  # ,filter__name='DES_r')
    for c in cutouts:
        if c.fits.name and os.path.exists(c.fits.name):
            trim_image(c)

    transient = Transient.objects.get(pk=transient.pk)
    transient.image_trim_status = "processed"
    transient.save()


def trim_image(cutout):
    transient = cutout.transient
    hdu = fits.open(cutout.fits.name)
    wcs = WCS(hdu[0].header)
    center_x, center_y = wcs.wcs_world2pix(transient.ra_deg, transient.dec_deg, 0)
    offset_ra, offset_dec = wcs.wcs_pix2world(center_x + 10, center_y, 0)
    arcsec2pix = SkyCoord(transient.ra_deg, transient.dec_deg, unit=u.deg).separation(
        SkyCoord(offset_ra, offset_dec, unit=u.deg)
    ).arcsec / 10.

    if transient.best_redshift is None:
        # no host, no redshift: 5 kpc at z = 0.01
        size_arcsec = get_local_aperture_size(0.01, 5)
        size_pix_5kpc = 2 * size_arcsec / arcsec2pix
    else:
        # 5 kpc at whatever the redshift is
        size_arcsec = get_local_aperture_size(transient.best_redshift, 5)
        size_pix_5kpc = 2 * size_arcsec / arcsec2pix

    if transient.host:
        # we need to include the host position
        # +50 pixels(?)
        host_center_x, host_center_y = \
            wcs.wcs_world2pix(transient.host.ra_deg, transient.host.dec_deg, 0)

        # if we have a host
        aperture_qs = Aperture.objects.filter(cutout=cutout, type='global')
        if len(aperture_qs):
            # we need a square cutout with the transient at the center
            # and includes the minimum pixel of the aperture radius
            aperture = aperture_qs[0]
            bbox = aperture.sky_aperture.to_pixel(wcs).bbox
            size = np.max([abs(bbox.ixmax - center_x) * 2 + 100, abs(bbox.ixmin - center_x) * 2 + 100,
                           abs(bbox.iymax - center_y) * 2 + 100, abs(bbox.iymin - center_y) * 2 + 100])

        else:

            size = np.max([abs(host_center_x - center_x) * 2 + 100,
                           abs(host_center_y - center_y) * 2 + 100])
    else:
        size = size_pix_5kpc
    # minimum size is 100x100 pixels
    if size < size_pix_5kpc:
        size = size_pix_5kpc
    if size < 100:
        size = 100

    cutout_new = Cutout2D(
        hdu[0].data,
        (center_x, center_y),
        (size, size),
        wcs=wcs
    )
    hdu[0].data = cutout_new.data
    hdu[0].header.update(cutout_new.wcs.to_header())
    hdu.writeto(cutout.fits.name, overwrite=True)
