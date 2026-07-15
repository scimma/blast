import itertools
from typing import List

from django.test import TestCase
from host import models

from .. import serializers
from ..components import aperture_component
from ..components import data_model_components
from ..components import host_component
from ..components import photometry_component
from ..components import sed_fit_component
from ..components import transient_component
from ..datamodel import DataModelComponent
from ..datamodel import serialize_blast_science_data
from ..datamodel import unpack_component_groups


class DatamodelConstructionTest(TestCase):
    fixtures = ["../fixtures/test/test_transient_data.yaml"]

    def test_datamodel_build_with_data(self):
        host = host_component("2022testone")
        transient = transient_component("2022testone")
        data = serialize_blast_science_data(host + transient)
        self.assertTrue(data["transient_name"] == "2022testone")
        self.assertTrue(data["host_name"] == "PSO J080624.103+010209.859")

    def test_datamodel_build_without_data(self):
        host = host_component("thisTransientDoesNotExist")
        transient = transient_component("thisTransientDoesNotExist")
        data = serialize_blast_science_data(host + transient)
        self.assertTrue(data["transient_name"] is None)
        self.assertTrue(data["host_name"] is None)

    def test_photometry_build(self):
        photometry = photometry_component("2022testone")
        data = serialize_blast_science_data(photometry)
        self.assertTrue(data["local_aperture_2MASS_H_flux"] == 2183.8)
        self.assertTrue(data["local_aperture_2MASS_H_flux_error"] == 224.97)
        self.assertTrue(data["local_aperture_2MASS_H_magnitude"] == 0.0)
        self.assertTrue(data["local_aperture_2MASS_H_magnitude_error"] == 0.0)

        self.assertTrue(data["local_aperture_2MASS_J_flux"] == 1091.48)
        self.assertTrue(data["local_aperture_2MASS_J_flux_error"] == 130.38)
        self.assertTrue(data["local_aperture_2MASS_J_magnitude"] == 0.0)
        self.assertTrue(data["local_aperture_2MASS_J_magnitude_error"] == 0.0)

        self.assertTrue(data["global_aperture_2MASS_J_flux"] == 99.0)
        self.assertTrue(data["global_aperture_2MASS_J_flux_error"] == 99.0)
        self.assertTrue(data["global_aperture_2MASS_J_magnitude"] == 0.0)
        self.assertTrue(data["global_aperture_2MASS_J_magnitude_error"] == 0.0)

        self.assertTrue(data["global_aperture_2MASS_H_flux"] == 1.0)
        self.assertTrue(data["global_aperture_2MASS_H_flux_error"] == 1.0)
        self.assertTrue(data["global_aperture_2MASS_H_magnitude"] == 10.0)
        self.assertTrue(data["global_aperture_2MASS_H_magnitude_error"] == 0.2)

    def test_aperture_build(self):
        aperture = aperture_component("2022testone")
        data = serialize_blast_science_data(aperture)

        self.assertTrue(data["local_aperture_ra_deg"] == 121.6015)
        self.assertTrue(data["local_aperture_dec_deg"] == 1.03586)
        self.assertTrue(data["local_aperture_semi_major_axis_arcsec"] == 1.0)
        self.assertTrue(data["local_aperture_semi_minor_axis_arcsec"] == 1.0)
        self.assertTrue(data["local_aperture_cutout"] is None)

        self.assertTrue(data["global_aperture_ra_deg"] == 11.6015)
        self.assertTrue(data["global_aperture_dec_deg"] == 10.03586)
        self.assertTrue(data["global_aperture_semi_major_axis_arcsec"] == 0.4)
        self.assertTrue(data["global_aperture_semi_minor_axis_arcsec"] == 0.5)
        self.assertTrue(data["global_aperture_cutout"]["name"] == "2022testone_2MASS_J")

    def test_sed_fitting_result_build(self):
        sed_fit = sed_fit_component("2022testone")
        data = serialize_blast_science_data(sed_fit)
        self.assertTrue(data["local_aperture_host_log_mass_16"] == 10.0)
        self.assertTrue(data["local_aperture_host_log_mass_50"] == 20.0)
        self.assertTrue(data["local_aperture_host_log_mass_84"] == 30.0)
        self.assertTrue(data["local_aperture_host_log_sfr_16"] == 123.4546)
        self.assertTrue(data["local_aperture_host_log_sfr_50"] == 123.4566)
        self.assertTrue(data["local_aperture_host_log_sfr_84"] == 56.564565)
        self.assertTrue(data["local_aperture_host_log_ssfr_16"] == 15.676)
        self.assertTrue(data["local_aperture_host_log_ssfr_50"] == 12.34343)
        self.assertTrue(data["local_aperture_host_log_ssfr_84"] == 12)
        self.assertTrue(data["local_aperture_host_age_16"] == 1.0)
        self.assertTrue(data["local_aperture_host_age_50"] == 0.1)
        self.assertTrue(data["local_aperture_host_age_84"] == 5.0)

        self.assertTrue(data["global_aperture_host_log_mass_16"] == 1.0)
        self.assertTrue(data["global_aperture_host_log_mass_50"] == 2.0)
        self.assertTrue(data["global_aperture_host_log_mass_84"] == 3.0)
        self.assertTrue(data["global_aperture_host_log_sfr_16"] == 123.4546)
        self.assertTrue(data["global_aperture_host_log_sfr_50"] == 123.4566)
        self.assertTrue(data["global_aperture_host_log_sfr_84"] == 56.564565)
        self.assertTrue(data["global_aperture_host_log_ssfr_16"] == 15.676)
        self.assertTrue(data["global_aperture_host_log_ssfr_50"] == 12.34343)
        self.assertTrue(data["global_aperture_host_log_ssfr_84"] == 12)
        self.assertTrue(data["global_aperture_host_age_16"] == 1.0)
        self.assertTrue(data["global_aperture_host_age_50"] == 0.1)
        self.assertTrue(data["global_aperture_host_age_84"] == 5.0)


class DataModelComponentTests(TestCase):
    def test_all_datamodel_components_output_type(self):
        for component in data_model_components:
            output = component("thisTransientDoesNotExist")
            self.assertIsInstance(output, List)
            for model_component in output:
                self.assertIsInstance(model_component, DataModelComponent)


class GroupComponentUnpackTest(TestCase):
    def test_component_group_unpacking(self):
        nested = [[1], [2], [3, 4]]
        nested_two = [[1], [2], [3], [4]]
        flat = [1, 2, 3, 4]
        unpack_nested = unpack_component_groups(nested)
        self.assertTrue(flat == unpack_nested)
        unpack_flat = unpack_component_groups(nested_two)
        self.assertTrue(flat == unpack_flat)
