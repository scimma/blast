import json
import os
from pathlib import Path
from django.contrib.auth.models import User, Permission
from django.test import Client
from django.test import TestCase
from rest_framework.test import APIClient
from rest_framework import status
from django.contrib.contenttypes.models import ContentType
from host.models import Alias


class APITest(TestCase):
    fixtures = ["../fixtures/test/test_transient_data.yaml"]

    def setUp(self):
        self.client = Client()

    def test_transient_get(self):
        client = APIClient()
        request = client.get("/api/transient/get/2022testone?format=json")
        data = json.loads(request.content)

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

        self.assertTrue(data["transient_name"] == "2022testone")
        self.assertTrue(data["host_name"] == "PSO J080624.103+010209.859")

        self.assertTrue(data["local_aperture_host_log_mass_16"] == 10.0)
        self.assertTrue(data["local_aperture_host_log_mass_50"] == 20.0)
        self.assertTrue(data["local_aperture_host_log_mass_84"] == 30.0)
        self.assertTrue(data["local_aperture_host_log_sfr_16"] == 123.4546)
        self.assertTrue(data["local_aperture_host_log_sfr_50"] == 123.4566)
        self.assertTrue(data["local_aperture_host_log_sfr_84"] == 56.564565)
        self.assertTrue(data["local_aperture_host_log_ssfr_16"] == 15.676)
        self.assertTrue(data["local_aperture_host_log_ssfr_50"] == 12.34343)
        self.assertTrue(data["local_aperture_host_log_ssfr_84"] == 12)
        self.assertTrue(data["local_aperture_host_log_age_16"] == 1.0)
        self.assertTrue(data["local_aperture_host_log_age_50"] == 0.1)
        self.assertTrue(data["local_aperture_host_log_age_84"] == 5.0)

        self.assertTrue(data["global_aperture_host_log_mass_16"] == 1.0)
        self.assertTrue(data["global_aperture_host_log_mass_50"] == 2.0)
        self.assertTrue(data["global_aperture_host_log_mass_84"] == 3.0)
        self.assertTrue(data["global_aperture_host_log_sfr_16"] == 123.4546)
        self.assertTrue(data["global_aperture_host_log_sfr_50"] == 123.4566)
        self.assertTrue(data["global_aperture_host_log_sfr_84"] == 56.564565)
        self.assertTrue(data["global_aperture_host_log_ssfr_16"] == 15.676)
        self.assertTrue(data["global_aperture_host_log_ssfr_50"] == 12.34343)
        self.assertTrue(data["global_aperture_host_log_ssfr_84"] == 12)
        self.assertTrue(data["global_aperture_host_log_age_16"] == 1.0)
        self.assertTrue(data["global_aperture_host_log_age_50"] == 0.1)
        self.assertTrue(data["global_aperture_host_log_age_84"] == 5.0)

        self.assertTrue(request.status_code == status.HTTP_200_OK)

    def test_transient_post(self):
        client = APIClient()
        request = client.post("/api/transient/post/name=2022testnew&ra=-1.0&dec=-5.0")
        data = json.loads(request.content)
        self.assertTrue(request.status_code == status.HTTP_201_CREATED)
        self.assertTrue(
            data["message"]
            == "transient successfully posted: 2022testnew: ra = -1.0, dec= -5.0"
        )

    def test_transient_bad_post(self):
        client = APIClient()
        request = client.post("/api/transient/post/name=2022new&ra=-*1.0&dec=-5.0")
        data = json.loads(request.content)
        self.assertTrue(request.status_code == status.HTTP_400_BAD_REQUEST)
        self.assertTrue(data["message"] == "bad ra and dec: ra=-*1.0, dec=-5.0")

        request = client.post(
            "/api/transient/post/name=2022new&ra=-999999&dec=-78895.0"
        )
        data = json.loads(request.content)
        self.assertTrue(request.status_code == status.HTTP_400_BAD_REQUEST)

    def test_transient_already_in_database(self):
        client = APIClient()
        request = client.post("/api/transient/post/name=2022testone&ra=-1.0&dec=-5.0")
        data = json.loads(request.content)
        self.assertTrue(request.status_code == status.HTTP_409_CONFLICT)
        self.assertTrue(data["message"] == "2022testone already in database")

    def test_get_no_transient(self):
        client = APIClient()
        request = client.get("/api/transient/get/2022NotInDatabase?format=json")
        data = json.loads(request.content)
        self.assertTrue(request.status_code == status.HTTP_404_NOT_FOUND)
        self.assertTrue(data["message"] == "2022NotInDatabase not in database")

    def test_alias(self):
        # Create a temporary user and authenticate them
        user = User.objects.create_user(username="testola", password='password')
        self.client.force_login(user)
        object_type = 'transient'
        name = '2022testone'
        alias = '2022testone-alias-test!'
        # Attempt to create an alias without permission
        response = self.client.post(f'/api/alias/{alias}/{object_type}/{name}/')
        # print(f'[{response.status_code}] {response.content}')
        data = json.loads(response.content)
        self.assertTrue(response.status_code == status.HTTP_403_FORBIDDEN)
        # Grant the user permission
        add_permission = Permission.objects.get(
            codename="add_alias",
            content_type=ContentType.objects.get_for_model(Alias),
        )
        user.user_permissions.add(add_permission)
        assert user.has_perm('host.add_alias')
        response = self.client.post(f'/api/alias/{alias}/{object_type}/{name}/')
        self.assertTrue(response.status_code == status.HTTP_201_CREATED)
        data = json.loads(response.content)
        self.assertTrue(data["message"].startswith("Alias successfully created:"))
        # Fetch information about the alias anonymously
        self.client.logout()
        response = self.client.get(f'/api/alias/{alias}/')
        self.assertTrue(response.status_code == status.HTTP_200_OK)
        # Fail when attempting to create another alias with the same name
        self.client.force_login(user)
        object_type = 'host'
        response = self.client.post(f'/api/alias/{alias}/{object_type}/{name}/')
        self.assertTrue(response.status_code == status.HTTP_409_CONFLICT)
        # Attempt to delete an alias without permission
        response = self.client.delete(f'/api/alias/{alias}/')
        # print(f'[{response.status_code}] {response.content}')
        self.assertTrue(response.status_code == status.HTTP_403_FORBIDDEN)
        # Grant the user delete permission
        delete_permission = Permission.objects.get(
            codename="delete_alias",
            content_type=ContentType.objects.get_for_model(Alias),
        )
        user.user_permissions.add(delete_permission)
        # Delete the alias
        response = self.client.delete(f'/api/alias/{alias}/')
        # print(f'[{response.status_code}] {response.content}')
        self.assertTrue(response.status_code == status.HTTP_200_OK)
        # Attempt to delete a non-existent alias
        alias = 'foo'
        response = self.client.delete(f'/api/alias/{alias}/')
        self.assertTrue(response.status_code == status.HTTP_404_NOT_FOUND)
        # print(f'[{response.status_code}] {response.content}')

    def test_transient_export_no_files(self):
        client = APIClient()
        # Load expected data
        with open(os.path.join(Path(__file__).resolve().parent, '2022testone_export_test.json')) as fp:
            expected_data = json.load(fp)
        expected_data.pop('metadata')
        # Fetch data from API
        request = client.get("/api/transient/export/2022testone/")
        data = json.loads(request.content)
        # Remove the metadata content that contains the generation timestamp
        data.pop('metadata')
        self.assertTrue(data == expected_data)
