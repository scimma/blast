from astropy.coordinates import SkyCoord
from django.test import TestCase
from django.test import tag

from ..catalog_photometry import catalog_photometry
from ..catalog_photometry import filter_information
from ..cutouts import cutout
from ..host_utils import survey_list


class CutoutDownloadTest(TestCase):
    def setUp(self):
        self.surveys = survey_list("host/data/survey_metadata.yml")

    @tag('download')
    def needs_review_test_cutout_download(self):
        """ "
        Test that cutout data can be downloaded.
        """
        for survey in self.surveys:
            position = SkyCoord(
                ra=survey.test_ra_deg, dec=survey.test_dec_deg, unit="deg"
            )
            cutout_data = cutout(position, survey)
            with self.subTest(survey=survey.name):
                self.assertTrue(cutout_data is not None)


class CatalogDownloadTest(TestCase):
    def setUp(self):
        self.catalogs = survey_list("host/data/catalog_metadata.yml")

    @tag('download')
    def test_catalog_download(self):
        """
        Test that catalog data can be downloaded.
        """
        for catalog in self.catalogs:
            position = SkyCoord(
                ra=catalog.test_ra_deg, dec=catalog.test_dec_deg, unit="deg"
            )
            catagog_data = catalog_photometry(position, catalog)
            with self.subTest(catalog=catalog.name):
                self.assertTrue(catagog_data is not None)

    @tag('download')
    def test_filter_information_download(self):
        """
        Test that catalog filter data can be downloaded.
        """
        for catalog in self.catalogs:
            filter_data = filter_information(catalog)
            with self.subTest(catalog=catalog.name):
                self.assertTrue(filter_data is not None)
