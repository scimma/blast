import numpy as np
from astropy.io import fits
from django.test import TestCase

from ..host_utils import build_source_catalog
from ..host_utils import estimate_background
from ..models import Transient, Filter
from ..transient_tasks import GlobalApertureConstruction
from ..cutouts import download_and_save_cutouts

class TestApertureConstruction(TestCase):

    def test_aperture_construction(self):
        transient = Transient.objects.get(name="2010H")
        download_and_save_cutouts(
            transient,
            filter_set=Filter.objects.filter(name='PanSTARRS_g')
        )
        gac_cls = GlobalApertureConstruction(transient_name=transient.name)

        status_message = gac_cls._run_process(transient)

        assert status_message == "processed"

    def test_aperture_failures(self):
        data = np.zeros((500, 5000), dtype=np.float64)
        hdu = fits.PrimaryHDU(data=data)
        hdulist = fits.HDUList(hdus=[hdu])

        background = estimate_background(hdulist)
        catalog = build_source_catalog(hdulist, background)

        assert catalog is None
