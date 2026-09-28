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
from host.models import Transient
from host.host_utils import import_transient_info
from host.host_utils import equal_dicts
from host.object_store import ObjectStore
from django.conf import settings


class APITestDataset(TestCase):
    """Unit tests for transient dataset API functions.
       THESE UNIT TESTS MUST BE RUN IN INDIVIDUAL DJANGO TEST RUNNER INSTANCES because Django’s test runner is designed
       to create one test database per configured database alias for the entire test run, not one database per app,
       test module, or TestCase subclass."""
    def setUp(self):
        self.client = Client()
        with open('''/data/transient_datasets/2026dix.tar.gz''', 'rb') as dataset_fileobj:
            import_transient_info(dataset_fileobj)

    def test_dataset_get(self):
        client = APIClient()
        # Load expected data
        with open(os.path.join(Path(__file__).resolve().parent, 'dataset_get_test_expected_data.json')) as fp:
            expected_data = json.load(fp)
        # Ignore metadata that includes timestamp of export
        expected_data.pop('metadata')
        # Fetch data from API
        request = client.get("/api/dataset/2026dix/")
        data = json.loads(request.content)
        # Remove the metadata content that contains the generation timestamp
        data.pop('metadata')
        self.assertTrue(equal_dicts(data, expected_data))
        self.assertTrue(request.status_code == status.HTTP_200_OK)

    def test_dataset_delete(self):
        # Create a temporary user and authenticate them
        user = User.objects.create_user(username="testola", password='password')
        self.client.force_login(user)
        transient_name = '2026dix'
        # Attempt to delete the dataset without permission
        response = self.client.delete(f'/api/dataset/{transient_name}/')
        self.assertTrue(response.status_code == status.HTTP_403_FORBIDDEN)
        # Grant the user permission
        add_permission = Permission.objects.get(
            codename="delete_transient",
            content_type=ContentType.objects.get_for_model(Transient),
        )
        user.user_permissions.add(add_permission)
        assert user.has_perm('host.delete_transient')
        # Try again with permission
        response = self.client.delete(f'/api/dataset/{transient_name}/')
        self.assertTrue(response.status_code == status.HTTP_204_NO_CONTENT)

    def test_dataset_delete_with_files(self):
        # Create a temporary user and authenticate them
        user = User.objects.create_user(username="testola", password='password')
        self.client.force_login(user)
        transient_name = '2026dix'
        # Attempt to delete the dataset without permission
        response = self.client.delete(f'/api/dataset/{transient_name}/?files=true')
        self.assertTrue(response.status_code == status.HTTP_403_FORBIDDEN)
        # Grant the user permission
        add_permission = Permission.objects.get(
            codename="delete_transient",
            content_type=ContentType.objects.get_for_model(Transient),
        )
        user.user_permissions.add(add_permission)
        assert user.has_perm('host.delete_transient')
        # Try again with permission
        response = self.client.delete(f'/api/dataset/{transient_name}/?files=true')
        self.assertTrue(response.status_code == status.HTTP_204_NO_CONTENT)
        s3 = ObjectStore()
        self.assertFalse(s3.object_exists(os.path.join(settings.S3_BASE_PATH, settings.CUTOUT_ROOT.strip('/'),
                                                       '2026dix/PanSTARRS/PanSTARRS_g.jpg')))

    def test_dataset_get_missing(self):
        client = APIClient()
        request = client.get("/api/dataset/NotInDatabase/")
        self.assertTrue(request.status_code == status.HTTP_404_NOT_FOUND)
        data = json.loads(request.content)
        self.assertTrue(data["message"] == "NotInDatabase not in database")


class APITestAlias(TestCase):
    def setUp(self):
        self.client = Client()
        with open('''/data/transient_datasets/2026dix.tar.gz''', 'rb') as dataset_fileobj:
            import_transient_info(dataset_fileobj)

    def test_alias(self):
        # Create a temporary user and authenticate them
        user = User.objects.create_user(username="testola", password='password')
        self.client.force_login(user)
        object_type = 'transient'
        name = '2026dix'
        alias = '2026dix-alias-test!'
        # Attempt to create an alias without permission
        response = self.client.post('/api/alias/', json={
            'alias': alias,
            object_type: name,
        })
        self.assertTrue(response.status_code == status.HTTP_403_FORBIDDEN)
        # Grant the user permission
        add_permission = Permission.objects.get(
            codename="add_alias",
            content_type=ContentType.objects.get_for_model(Alias),
        )
        user.user_permissions.add(add_permission)
        assert user.has_perm('host.add_alias')
        response = self.client.post('/api/alias/', data={
            'alias': alias,
            object_type: name,
        })
        self.assertTrue(response.status_code == status.HTTP_201_CREATED)
        # Fetch information about the alias anonymously
        self.client.logout()
        response = self.client.get(f'/api/alias/{alias}/')
        self.assertTrue(response.status_code == status.HTTP_200_OK)
        # Fail when attempting to create another alias with the same name
        self.client.force_login(user)
        object_type = 'host'
        response = self.client.post('/api/alias/', data={
            'alias': alias,
            object_type: name,
        })
        self.assertTrue(response.status_code == status.HTTP_400_BAD_REQUEST)
        # Attempt to delete an alias without permission
        response = self.client.delete(f'/api/alias/{alias}/')
        self.assertTrue(response.status_code == status.HTTP_403_FORBIDDEN)
        # Grant the user delete permission
        delete_permission = Permission.objects.get(
            codename="delete_alias",
            content_type=ContentType.objects.get_for_model(Alias),
        )
        user.user_permissions.add(delete_permission)
        # Delete the alias
        response = self.client.delete(f'/api/alias/{alias}/')
        self.assertTrue(response.status_code == status.HTTP_204_NO_CONTENT)
        # Attempt to delete a non-existent alias
        alias = 'foo'
        response = self.client.delete(f'/api/alias/{alias}/')
        self.assertTrue(response.status_code == status.HTTP_404_NOT_FOUND)
