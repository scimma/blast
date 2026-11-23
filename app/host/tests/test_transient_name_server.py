from django.test import TestCase

from ..transient_name_server import build_tns_header
from ..transient_name_server import build_tns_url
from ..transient_name_server import tns_to_blast_transient


class BuildTNSHeaderTest(TestCase):
    def setUp(self):
        self.tns_bot_id = 12345
        self.tns_bot_name = "test_bot"
        self.header = build_tns_header(self.tns_bot_id, self.tns_bot_name)

    def test_marker(self):
        marker = self.header["User-Agent"]
        true_marker = (
            f'tns_marker{{"tns_id": "{self.tns_bot_id}",'
            f'"type": "bot", "name": "{self.tns_bot_name}"}}'
        )
        self.assertEqual(marker, true_marker)

    def test_return_type_is_dict(self):
        self.assertIsInstance(self.header, dict)

    def test_return_contains_correct_key(self):
        user_agent = self.header.get("User-Agent")
        self.assertTrue(user_agent is not None)

    def test_return_dictionary_size(self):
        self.assertTrue(len(self.header) == 1)


class BuildTNSUrlTest(TestCase):
    def test_search_mode(self):
        url = build_tns_url("test", mode="search")
        self.assertEqual(url, "test/search")

    def test_get_mode(self):
        url = build_tns_url("test", mode="get")
        self.assertEqual(url, "test/object")

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            build_tns_url("test", mode="something_else")
        with self.assertRaises(ValueError):
            build_tns_url("test")


class ConvertTNSToBLASTTest(TestCase):
    def test_tns_to_blast(self):
        tns_transient = {}

        tns_transient["objname"] = "2021evd"
        tns_transient["objid"] = 2345
        tns_transient["radeg"] = 12.0
        tns_transient["decdeg"] = 13.0
        tns_transient["name_prefix"] = "SN"
        tns_transient["discoverydate"] = "2022-02-04 07:29:02.112+00:00"
        tns_transient["object_type"] = {"name": "SN 1a"}
        tns_transient["redshift"] = 0.0

        blast_transient = tns_to_blast_transient(tns_transient)

        self.assertEqual(tns_transient["objname"], blast_transient.name)
        self.assertEqual(tns_transient["objid"], blast_transient.tns_id)
        self.assertEqual(tns_transient["radeg"], blast_transient.ra_deg)
        self.assertEqual(tns_transient["decdeg"], blast_transient.dec_deg)
        self.assertEqual(tns_transient["name_prefix"], blast_transient.tns_prefix)
        self.assertEqual(
            tns_transient["discoverydate"], blast_transient.public_timestamp
        )
        self.assertEqual(
            tns_transient["object_type"]["name"], blast_transient.spectroscopic_class
        )
