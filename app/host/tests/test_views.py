from django.test import TestCase
from textwrap import dedent
from django.contrib.auth.models import User


class ViewTest(TestCase):
    fixtures = ["../fixtures/test/setup_test_transient.yaml"]

    def test_transient_list_page(self):
        response = self.client.get("/transients/")
        self.assertEqual(response.status_code, 200)

    def test_transient_page(self):
        response = self.client.get("/transients/2022testone/")
        self.assertEqual(response.status_code, 200)

        response = self.client.get("/transients/2022testtwo/")
        self.assertEqual(response.status_code, 200)


class SEDPlotTest(TestCase):

    def test_tansient_page(self):
        response = self.client.get("/transients/2010H/")
        self.assertEqual(response.status_code, 200)


class AddTransientTest(TestCase):
    def setUp(self):
        import base64
        username = 'testuser'
        b64username = base64.urlsafe_b64encode(username.encode('utf-8')).decode('utf-8').strip('=')
        self.credentials = {"username": b64username, "password": "secret"}
        User.objects.create_user(**self.credentials, is_superuser=True)
        self.client.login(**self.credentials)

    def test_add_tansients_by_definition(self):
        # TODO: This test is fragile due to the explicit HTML string search.
        response = self.client.post("/add/", data={
            'full_info': dedent('''
                2010ag,255.97346,31.50172,None,None
                2010H,255.97346,31.50172,None,None
                SN_2010ag,255.98554,31.511860000000002,None,None
                AT_2010ag,255.98554,31.511860000000002,None,None
                2010ag_foo,255.97138,31.50172,None,None
                2010ag_bar,255.97346,31.50186,None,None
                new_hi,255.98554,31.511860000000002,None,None
                new_lo,255.96138000000002,31.49158,None,None
                65_character_long_transient_name_00000000000000000000000000000000,256.00,31.7,None,None
                64_character_long_transient_name_0000000000000000000000000000000,255.99,31.6,None,None
            ''')})
        self.assertEqual(response.status_code, 200)
        # Check that three transients were added
        self.assertContains(response, text='<p>The following transients were successfully added to the Blast database:</p>\n  <ul>\n    \n    <li><a href="/transients/new_hi">new_hi</a></li>\n    \n    <li><a href="/transients/new_lo">new_lo</a></li>\n    \n    <li><a href="/transients/64_character_long_transient_name_0000000000000000000000000000000">64_character_long_transient_name_0000000000000000000000000000000</a></li>\n    \n  </ul>')  # noqa
        # Check that cone search discarded two transients
        self.assertContains(response, text='<li>Transient &quot;2010ag_foo&quot; is within 1 arcsec of existing transient(s) 2010ag. Discarding.</li>')  # noqa
        self.assertContains(response, text='<li>Transient &quot;2010ag_bar&quot; is within 1 arcsec of existing transient(s) 2010ag. Discarding.</li>')  # noqa
        # Check that naming conventions are enforced
        self.assertContains(response, text='<li>Error creating transient: SN_2010ag starts with an illegal prefix (SN or AT)</li>')  # noqa
        self.assertContains(response, text='<li>Error creating transient: AT_2010ag starts with an illegal prefix (SN or AT)</li>')  # noqa
        self.assertContains(response, text='<li>Error creating transient: 65_character_long_transient_name_00000000000000000000000000000000 is longer than the max length of 64 characters.</li>')  # noqa
        # print(f'''Response: [{response.status_code}]\n{response.content}''')
        # print(f'''Response: [{response.status_code}]\n{response.content.decode('utf-8')}''')
