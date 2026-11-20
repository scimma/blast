import os

from astropy.coordinates import SkyCoord
from django.test import TestCase
from django.test import tag
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..cutouts import cutout
from ..models import Filter

from host.log import get_logger
logger = get_logger(__name__)


class CutoutDownloadTest(TestCase):

    def setUp(self):
        self.transient_name = "2026dgt"
        self.transient_ra = 132.3563
        self.transient_dec = 29.5105
        self.status_failed = "failed"
        self.status_succeeded = "processed"
        self.download_dir = '/tmp/'

    def process_filter(self, filter):
        print(f'''Processing filter "{filter.name}"...''')
        position = SkyCoord(ra=self.transient_ra, dec=self.transient_dec, unit="deg")
        save_dir = os.path.join(self.download_dir, f"{self.transient_name}/{filter.name}/")
        os.makedirs(save_dir, exist_ok=True)
        path_to_fits = os.path.join(save_dir, f"{filter.name}.fits")
        try:
            # Delete file if for some reason it already exists
            os.remove(path_to_fits)
        except FileNotFoundError:
            pass
        cutout_data, status, error = cutout(position, filter, download_max_tries=1)
        if cutout_data is not None:
            logger.info(f'[{filter}] Cutout data downloaded.')
            return self.status_succeeded
        elif status == 0 and not error:
            logger.info(f'[{filter}] No cutout data available, but no errors.')
            return self.status_succeeded
        else:
            logger.error(f'[{filter}] Error downloading cutout data: {error}')
            return self.status_failed

    @tag('download')
    def test_cutout_download_panstarrs(self):
        status = self.process_filter(Filter.objects.get(name__exact='PanSTARRS_r'))
        self.assertEqual(status, self.status_succeeded)

    @tag('download')
    def test_cutout_download_all(self):
        """ "
        Test that cutout data can be downloaded.
        """
        os.makedirs(self.download_dir, exist_ok=True)

        results = []
        filter_set = Filter.objects.all()
        with ThreadPoolExecutor(max_workers=len(filter_set)) as executor:
            future_to_filter = {executor.submit(self.process_filter, filter): filter for filter in filter_set}
            for future in as_completed(future_to_filter):
                filter = future_to_filter[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logger.error('%r generated an exception: %s' % (filter, exc))
                    result = self.status_failed
                logger.debug(f'''"{filter}" result: "{result}"''')
                results.append(result)

        # If any of the cutout downloads failed, the test fails.
        self.assertFalse(not results or [result for result in results if result != self.status_succeeded])
