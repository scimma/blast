from django.contrib.auth.models import User
from django.test import TestCase


class LogInTest(TestCase):
    def setUp(self):
        self.credentials = {"username": "testuser", "password": "secret"}
        User.objects.create_user(**self.credentials, is_superuser=True)

    def test_login(self):
        # send login data
        response = self.client.post("/accounts/login/", self.credentials, follow=True)
        # should be logged in now
        self.assertTrue(response.context["user"].is_active)
