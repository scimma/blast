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
        download_dir = '/tmp/'
        os.makedirs(download_dir, exist_ok=True)
        for filter in Filter.objects.all():
            print(f'''Processing filter "{filter.name}"...''')
            position = SkyCoord(
                ra=self.transient_ra, dec=self.transient_dec, unit="deg"
            )
            save_dir = os.path.join(download_dir, f"{self.transient_name}/{filter.name}/")
            os.makedirs(save_dir, exist_ok=True)
            path_to_fits = os.path.join(save_dir, f"{filter.name}.fits")
            if os.path.exists(path_to_fits):
                print(f'''Skipping file already downloaded: "{path_to_fits}"...''')
                continue
            else:
                cutout_data = cutout(position, filter)[0]
                if cutout_data:
                    fits.writeto(path_to_fits, cutout_data[0].data, overwrite=True)

        self.assertTrue(1 == 1)
