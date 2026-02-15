from django.test import TestCase
from sedpy.observate import Filter as SedpyFilter

from ..models import Filter


class FilterTest(TestCase):

    def test_filter_conversion(self):
        """
        Test Blast filter is converted to sedpy filter
        """

        for filter in Filter.objects.all():
            # raw_data = pd.read_csv(f'{settings.TRANSMISSION_CURVES_ROOT}/{filter.name}.txt',
            #                  header=None, delim_whitespace=True)
            # raw_wavelength, raw_transmission = raw_data[0].values, raw_data[1].values
            sedpy_filter = filter.transmission_curve()
            self.assertTrue(sedpy_filter.nick == filter.name)
            self.assertTrue(isinstance(sedpy_filter, SedpyFilter))
            # assert_array_equal(sedpy_filter.wavelength, raw_wavelength)
            # assert_array_equal(sedpy_filter.transmission, raw_transmission)


class PropsectorBuildObsTest(TestCase):

    def test_build_obs(self):
        pass


# class SEDFittingFullTest(TestCase):

#     def test_prospector_global(self):

#     transient = Transient.objects.get(name="2010H")

#     apphot_cls = GlobalAperturePhotometry()
#     status_message = apphot_cls._run_process(transient)

#     sed_cls = GlobalHostSEDFitting()
#     status_message = sed_cls._run_process(transient, mode="test")

#     pr = SEDFittingResult.objects.filter(
#        Q(transient=transient)
#        & Q(aperture__type="global")
#        & Q(posterior__contains="/tmp")
#     )
#     self.assertTrue(len(pr) == 1)
#     self.assertTrue(status_message == "processed")
#     self.assertTrue(pr[0].log_ssfr_50 != None)

#     def test_prospector_local(self):

#       transient = Transient.objects.get(name="2010H")

#       apphot_cls = LocalAperturePhotometry()
#       status_message = apphot_cls._run_process(transient)

#       valid_cls = ValidateLocalPhotometry()
#       status_message = valid_cls._run_process(transient)

#       sed_cls = LocalHostSEDFitting()
#       status_message = sed_cls._run_process(transient, mode="test")

#       pr = SEDFittingResult.objects.filter(
#           Q(transient=transient)
#           & Q(aperture__type="local")
#           & Q(posterior__contains="/tmp")
#       )
#       self.assertTrue(len(pr) == 1)
#       self.assertTrue(status_message == "processed")
#       self.assertTrue(pr[0].log_ssfr_50 != None)
