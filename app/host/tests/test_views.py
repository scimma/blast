from django.test import TestCase
from textwrap import dedent
from django.contrib.auth.models import User
from django.db.models import Q
from ..host_utils import import_transient_info
from host.models import Transient
from host.models import Aperture
from host.models import TaskRegister


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


class ModifyTransientTest(TestCase):
    def setUp(self):
        import base64
        username = 'testuser'
        b64username = base64.urlsafe_b64encode(username.encode('utf-8')).decode('utf-8').strip('=')
        self.credentials = {"username": b64username, "password": "secret"}
        User.objects.create_user(**self.credentials, is_superuser=True)
        self.client.login(**self.credentials)
        with open('''/data/transient_datasets/2026dix.tar.gz''', 'rb') as dataset_fileobj:
            import_transient_info(dataset_fileobj)
        with open('''/data/transient_datasets/2026dgt.tar.gz''', 'rb') as dataset_fileobj:
            import_transient_info(dataset_fileobj)

    def test_update_tansient(self):
        # TODO: This test is fragile due to the explicit HTML string search.
        response = self.client.post("/add/", data={
            'update_info': dedent("""
                2026dix,None,None,0.05,SN Ia,None,None,None,None,None,None,None,z and class test
                2026dgt,None,None,None,None,None,132.3561625,29.5105694,None,5,5,0,host pos test
            """)})
        self.assertEqual(response.status_code, 200)
        # Check that transients were updated
        self.assertContains(
            response,
            text=(""" <p>The following transients were updated and re-triggered:</p>\n  <ul>"""
                  """\n    \n    <li><a href="/transients/2026dix">2026dix</a></li>\n    \n    <li>"""
                  """<a href="/transients/2026dgt">2026dgt</a></li>"""))

        # Check that 2026dix z and class were updated
        transient_26dix = Transient.objects.get(name='2026dix')
        self.assertEqual(transient_26dix.redshift, 0.05)
        self.assertEqual(transient_26dix.spectroscopic_class, 'SN Ia')

        # Check the 2026dgt host coord and aperture were updated
        transient_26dgt = Transient.objects.get(name='2026dgt')
        self.assertEqual(transient_26dgt.host.ra_deg, 132.3561625)
        self.assertEqual(transient_26dgt.host.dec_deg, 29.5105694)

        aperture = Aperture.objects.get(transient__name='2026dgt', cutout__filter__name='PanSTARRS_g')
        self.assertEqual(aperture.semi_major_axis_arcsec, 5)
        self.assertEqual(aperture.semi_minor_axis_arcsec, 5)
        self.assertEqual(aperture.orientation_deg, 0)

        # check that 2026dix tasks set to unprocessed
        tasks_not_processed = TaskRegister.objects.filter(transient__name='2026dix', task__name__in=[
            'Global host SED inference', 'Local aperture photometry',
            'Validate local photometry', 'Local host SED inference'
        ]
        )
        for t in tasks_not_processed:
            self.assertEqual(t.status.message, 'not processed')

        tasks_processed = TaskRegister.objects.filter(transient__name='2026dix').filter(
            ~Q(
                task__name__in=[
                    'Global host SED inference', 'Local aperture photometry',
                    'Validate local photometry', 'Local host SED inference'
                ]
            )
        )
        for t in tasks_processed:
            self.assertEqual(t.status.message, 'processed')

        # check that 2026dgt tasks set to unprocessed
        tasks_not_processed = TaskRegister.objects.filter(transient__name='2026dgt', task__name__in=[
            'Global aperture photometry',
            'Validate global photometry',
            'Global host SED inference'
        ]
        )
        for t in tasks_not_processed:
            self.assertEqual(t.status.message, 'not processed')

        tasks_processed = TaskRegister.objects.filter(transient__name='2026dgt').filter(
            ~Q(
                task__name__in=[
                    'Global aperture photometry',
                    'Validate global photometry',
                    'Global host SED inference'
                ]
            )
        )
        for t in tasks_processed:
            self.assertEqual(t.status.message, 'processed')


class AddTransientTest(TestCase):
    def setUp(self):
        import base64
        username = 'testuser'
        b64username = base64.urlsafe_b64encode(username.encode('utf-8')).decode('utf-8').strip('=')
        self.credentials = {"username": b64username, "password": "secret"}
        User.objects.create_user(**self.credentials, is_superuser=True)
        self.client.login(**self.credentials)
        with open('''/data/transient_datasets/2026dix.tar.gz''', 'rb') as dataset_fileobj:
            import_transient_info(dataset_fileobj)

    def test_add_tansients_by_definition(self):
        # TODO: This test is fragile due to the explicit HTML string search.
        response = self.client.post("/add/", data={
            'full_info': dedent('''
                new_hi,255.98554,31.511860000000002,None,None,new_hi
                new_lo,255.96138000000002,31.49158,None,None,new_lo
                64_character_long_transient_name_0000000000000000000000000000000,255.99,31.6,None,None,64_character_long_transient_name_0000000000000000000000000000000
                abcdefg1234567,254.97138,32.50172,None,None,abcdefg1234567
                2026dix,177.65625,55.353639,None,None,2026dix should skip because it exists
                SN_2026dix,178.6562,55.353639,None,None,SN_2026dix invalid prefix
                AT_2026dix,177.6562,54.353639,None,None,AT_2026dix invalid prefix
                2026dix_foo,177.65425,55.353639,None,None,2026dix_foo too close
                2026dix_bar,177.65625,55.353739,None,None,2026dix_bar too close
                65_character_long_transient_name_00000000000000000000000000000000,256.00,31.7,None,None,65_character_long_transient_name_00000000000000000000000000000000
                spaced name,254.97138,32.50172,None,None,spaced name
                -abcdefg1234567,254.97138,32.50172,None,None,-abcdefg1234567
                abcdefg1234567_,254.97138,32.50172,None,None,abcdefg1234567_
                abcdefg1234__7,254.97138,32.50172,None,None,abcdefg1234__7
                abcde--1234567,254.97138,32.50172,None,None,abcde--1234567
            ''')})
        self.assertEqual(response.status_code, 200)
        # Check that three transients were added
        self.assertContains(
            response,
            text=('<p>The following transients were successfully added to the Blast database:</p>\n  <ul>\n    \n    '''
                  '''<li><a href="/transients/new_hi">new_hi</a></li>\n    \n    '''
                  '''<li><a href="/transients/new_lo">new_lo</a></li>\n    \n    '''
                  '''<li><a href="/transients/64_character_long_transient_name_0000000000000000000000000000000">'''
                  '''64_character_long_transient_name_0000000000000000000000000000000</a></li>\n    \n    '''
                  '''<li><a href="/transients/abcdefg1234567">abcdefg1234567</a></li>'''))
        # Check that cone search discarded two transients
        for name in ['2026dix_foo', '2026dix_bar']:
            self.assertContains(
                response,
                text=f'Transient &quot;{name}&quot; is within 1 arcsec of existing transient(s) 2026dix. Discarding.')
        # Check that naming conventions are enforced
        for name in ['SN_2026dix', 'AT_2026dix']:
            self.assertContains(response,
                                text=f'&quot;{name}&quot; may not start with &quot;SN&quot; or &quot;AT&quot;')
        self.assertContains(
            response,
            text=('''&quot;65_character_long_transient_name_00000000000000000000000000000000&quot; '''
                  '''is longer than the max length of 64 characters.'''))
        for name in ['spaced name', '-abcdefg1234567', 'abcdefg1234567_']:
            self.assertContains(
                response,
                text=(f'&quot;{name}&quot; must begin and end with alphanumeric characters,'''
                      ''' and may include underscores and hyphens. Spaces are not allowed.'''))
        for name, character in [('abcdefg1234__7', '_'), ('abcde--1234567', '-')]:
            self.assertContains(
                response,
                text=f'&quot;{name}&quot; may not contain consecutive &quot;{character}&quot; characters')

        # print(f'''Response: [{response.status_code}]\n{response.content}''')
        # print(f'''Response: [{response.status_code}]\n{response.content.decode('utf-8')}''')
