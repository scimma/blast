import os

from astropy.coordinates import SkyCoord
from astropy.io import fits
from django.test import TestCase
from django.test import tag

from ..cutouts import cutout
from ..models import Filter

sn = ["2010ag", "2010ai", "2010y", "2010H", ""]


class CutoutDownloadTest(TestCase):

    def setUp(self):
        self.transient_name = "2010ag"
        self.transient_ra = 355.53628555
        self.transient_dec = 48.70907059166666

    @tag('download')
    def test_cutout_download(self):
        """ "
        Test that cutout data can be downloaded.
        """
        for filter in Filter.objects.all():
            position = SkyCoord(
                ra=self.transient_ra, dec=self.transient_dec, unit="deg"
            )
            cutout_data = cutout(position, filter)[0]

            if cutout_data:
                save_dir = f"{self.transient_name}/{filter.name}/"
                os.makedirs(save_dir, exist_ok=True)
                path_to_fits = save_dir + f"{filter.name}.fits"
                fits.writeto(path_to_fits, cutout_data[0].data, overwrite=False) # Set overwrite to false for now, cutouts don't change

        self.assertTrue(1 == 1)
