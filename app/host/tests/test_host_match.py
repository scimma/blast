from django.test import TestCase

from ..models import Transient
from ..transient_tasks import HostMatch
from ..host_utils import import_transient_info


class TestHostMatch(TestCase):
    def setUp(self):
        with open('''/data/transient_datasets/2026dix.tar.gz''', 'rb') as dataset_fileobj:
            import_transient_info(dataset_fileobj)

    def test_aperture_construction(self):
        transient = Transient.objects.get(name="2026dix")

        host_cls = HostMatch(transient_name=transient.name)

        status_message = host_cls._run_process(transient)

        assert status_message == "processed"
