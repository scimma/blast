from astropy.coordinates import SkyCoord
from django.test import TestCase
from django.core.exceptions import ValidationError

from ..models import SkyObject
from ..models import Host


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


class ValidateHostNameTest(TestCase):
    valid_names = [
        'V4l.1_d-N4m+3',
    ]
    invalid_names = [
        'name with spaces',
        'name	with	tabs',
        'name_$_inv@lid_chars',
        '_leading_underscore',
        'trailing_underscore_',
        '-leading-hyphen',
        'trailing-hyphen-',
        'consecutive__under___scores',
        'consecutive--hyph----ens',
        'consecutive++plus++++signs',
        'consecutive..period....s',
        'name__collision__',
        'name____collision',
        'name_$$_collision',
        " name with spaces ",
    ]

    def test_host_name_validation(self):
        for name in self.valid_names + self.invalid_names:
            try:
                Host.validate_name(name)
                assert name in self.valid_names
            except ValidationError:
                assert name in self.invalid_names

