from unittest import mock

from django.test import TestCase
from sedpy.observate import Filter as SedpyFilter

from ..models import Filter
from ..transient_tasks import GlobalHostSEDFitting


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


class HostSEDFittingNoValidDataTest(TestCase):

    def test_no_valid_data_returns_not_enough_filters(self):
        """
        If the S/N cut drops every filter, prospect's fix_obs raises
        AssertionError("No valid data to fit..."); the task should report
        "not enough filters" instead of failing.
        """
        transient = mock.MagicMock(best_redshift=0.1)
        transient.name = "TEST_NO_VALID_DATA"
        no_valid = AssertionError("No valid data to fit: check the sign of the masks.")
        with mock.patch("host.transient_tasks.Aperture") as aperture_cls,                 mock.patch("host.transient_tasks.build_obs", side_effect=no_valid):
            aperture_cls.objects.filter.return_value = [mock.MagicMock()]
            status = GlobalHostSEDFitting()._run_process(transient, mode="test")
        self.assertEqual(status, "not enough filters")

    def test_other_assertion_errors_still_raise(self):
        transient = mock.MagicMock(best_redshift=0.1)
        transient.name = "TEST_OTHER_ASSERT"
        with mock.patch("host.transient_tasks.Aperture") as aperture_cls,                 mock.patch("host.transient_tasks.build_obs",
                           side_effect=AssertionError("something else")):
            aperture_cls.objects.filter.return_value = [mock.MagicMock()]
            with self.assertRaises(AssertionError):
                GlobalHostSEDFitting()._run_process(transient, mode="test")


# class SEDFittingFullTest(TestCase):

#     def test_prospector_global(self):

#     transient = Transient.objects.get(name="2026dix")

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

#       transient = Transient.objects.get(name="2026dix")

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
