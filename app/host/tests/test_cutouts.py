import os

from astropy.coordinates import SkyCoord
from astropy.io import fits
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

    @tag('download')
    def test_cutout_download(self):
        """ "
        Test that cutout data can be downloaded.
        """
        status_failed = "failed"
        status_succeeded = "processed"
        download_dir = '/tmp/'
        os.makedirs(download_dir, exist_ok=True)

        def process_filter(filter):
            print(f'''Processing filter "{filter.name}"...''')
            position = SkyCoord(ra=self.transient_ra, dec=self.transient_dec, unit="deg")
            save_dir = os.path.join(download_dir, f"{self.transient_name}/{filter.name}/")
            os.makedirs(save_dir, exist_ok=True)
            path_to_fits = os.path.join(save_dir, f"{filter.name}.fits")
            try:
                # Delete file if for some reason it already exists
                os.remove(path_to_fits)
            except FileNotFoundError:
                pass
            cutout_data, status, error = cutout(position, filter)
            if cutout_data:
                logger.info(f'[{filter}] Cutout data downloaded.')
                return status_succeeded
            elif status == 0 and not error:
                logger.info(f'[{filter}] No cutout data available, but no errors.')
                return status_succeeded
            else:
                logger.error(f'[{filter}] Error downloading cutout data: {error}')
                return status_failed

        results = []
        filter_set = Filter.objects.all()
        with ThreadPoolExecutor(max_workers=len(filter_set)) as executor:
            future_to_filter = {executor.submit(process_filter, filter): filter for filter in filter_set}
            for future in as_completed(future_to_filter):
                filter = future_to_filter[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logger.error('%r generated an exception: %s' % (filter, exc))
                    result = status_failed
                logger.debug(f'''"{filter}" result: "{result}"''')
                results.append(result)

        # If any of the cutout downloads failed, the test fails.
        self.assertFalse(not results or [result for result in results if result != status_succeeded])
