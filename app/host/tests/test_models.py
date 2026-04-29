from astropy.coordinates import SkyCoord
from django.test import TestCase

from ..models import SkyObject


class SkyObjectTest(TestCase):
    def setUp(self):
        class test_obj(SkyObject):
            pass

        self.sky_obj = test_obj(ra_deg=12.0, dec_deg=13.0)
        self.sky_coord = SkyCoord(ra=12.0, dec=13.0, unit="deg")

    def test_sky_coord(self):
        sky_coord = self.sky_obj.sky_coord
        self.assertTrue(sky_coord == self.sky_coord)
        self.assertTrue(sky_coord.ra.deg == self.sky_coord.ra.deg)
        self.assertTrue(sky_coord.dec.deg == self.sky_coord.dec.deg)

    def test_ra_string(self):
        self.assertTrue(self.sky_obj.ra == "0h48m00.00s")

    def test_dec_string(self):
        self.assertTrue(self.sky_obj.dec == "13d00m00.00s")
