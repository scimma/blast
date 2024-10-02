from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D

def clean_transient_images(transient):
    
    cutouts = Cutout.objects.filter(transient=transient)
    for c in cutouts:

        trim_image(c)

    transient.image_trim_status = "processed"
    transient.save()
        
def trim_image(cutout):

    hdu = fits.open(cutout.fits.name)
    wcs = WCS(hdu[0].header)
    center_x,center_y = wcs.wcs_world2pix(transient.ra_deg,transient.dec_deg,0)
    offset_ra,offset_dec = wcs.wcs_pix2world(center_x+10,center_y,0)
    arcsec2pix = SkyCoord(transient.ra_deg,transient.dec_deg,unit=u.deg).separation(
        SkyCoord(offset_ra,offset_dec,unit=u.deg)
    ).arcsec/10.
    
    if transient.host is None and transient.best_redshift is None:
        # no host, no redshift: 5 kpc at z = 0.01
        size_arcsec = get_local_aperture_size(0.01,5)
        size_pix_5kpc = 2*size_arcsec/arcsec2pix
    elif transient.host is None:
        # 5 kpc at whatever the redshift is
        size_arcsec = get_local_aperture_size(transient.best_redshift,5)
        size_pix_5kpc = 2*size_arcsec/arcsec2pix

    if transient.host:
        # if we have a host
        aperture_qs = Aperture.objects.filter(cutout=cutout,type='global')
        if len(aperture_qs):
            # we need a square cutout with the transient at the center
            # and includes the minimum pixel of the aperture radius
            aperture = aperture_qs[0]
            if host_center_x > center_x:
                xsize = (np.max(aperture.bbox[:]) - center_x)*2 + 100
            else:
                xsize = (np.min(aperture.bbox[:]) - center_x)*2 + 100
            if host_center_y > center_y:
                ysize = (np.max(aperture.bbox[:]) - center_y)*2 + 100
            else:
                ysize = (np.min(aperture.bbox[:]) - center_y)*2 + 100

        else:
            # we need to include the host position
            # +50 pixels(?)
            host_center_x,host_center_y = \
                wcs.wcs_world2pix(transient.host.ra_deg,transient.host.dec_deg,0)

            xsize = np.abs(host_center_x - center_x)*2 + 100
            ysize = np.abs(host_center_y - center_y)*2 + 100
    # minimum size is 100x100 pixels
    if xsize < size_pix_5kpc:  xsize = size_pix_5kpc
    if ysize < size_pix_5kpc:  ysize = size_pix_5kpc
    if xsize < 100: xsize = 100
    if ysize < 100: ysize = 100
    
    cutout = Cutout2D(
        hdu[0].data,
        (center_x,center_y),
        (xsize,ysize),
        wcs=wcs
    )
    hdu.data = cutout.data
    hdu.header.update(cutout.wcs.to_header())

    hdu.writeto(cutout.fits.name, overwrite=True)

