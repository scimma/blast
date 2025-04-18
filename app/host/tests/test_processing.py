from django.test import TestCase

from ..base_tasks import initialise_all_tasks_status
from ..base_tasks import TransientTaskRunner
from ..base_tasks import update_status
from ..models import Cutout
from ..models import Filter
from ..models import Status
from ..models import Task
from ..models import TaskRegister
from ..models import Transient
from ..tasks import periodic_tasks
from ..transient_tasks import HostMatch
from ..transient_tasks import ImageDownload


class TaskRunnerTest(TestCase):
    fixtures = [
        "../fixtures/test/setup_test_transient.yaml",
        "../fixtures/test/setup_test_task_register.yaml",
        "../fixtures/test/test_cutout.yaml",
    ]

    def setUp(self):
        class TestRunnerProcessed(TransientTaskRunner):
            def _run_process(self, transient):
                return "processed"

            def _prerequisites(self):
                return {"Cutout download": "not processed"}

            @property
            def task_name(self):
                return "Cutout download"

            def _failed_status_message(self):
                return "failed"

        self.processed_runner = TestRunnerProcessed(transient_name="dummy")

        class TestRunnerFailed(TransientTaskRunner):
            def _run_process(self, transient):
                raise ValueError

            def _prerequisites(self):
                return {"Cutout download": "not processed"}

            @property
            def task_name(self):
                return "Cutout download"

            def _failed_status_message(self):
                return "failed"

        self.failed_runner = TestRunnerFailed(transient_name="dummy")

        class TestRunnerNotProcessed(TransientTaskRunner):
            def _run_process(self, transient):
                return "not processed"

            def _prerequisites(self):
                return {"Cutout download": "not processed"}

            @property
            def task_name(self):
                return "Cutout download"

            def _failed_status_message(self):
                return "failed"

        self.not_processed_runner = TestRunnerNotProcessed(transient_name="dummy")

        class TestRunnerTwoPrereqs(TransientTaskRunner):
            def _run_process(self, transient):
                return "not processed"

            def _prerequisites(self):
                return {"Cutout download": "not processed", "Host match": "processed"}

            @property
            def task_name(self):
                return "Cutout download"

            def _failed_status_message(self):
                return "failed"

        self.two_prereqs_runner = TestRunnerTwoPrereqs(transient_name="dummy")

        class TestRunnerTwoPrereqsSuc(TransientTaskRunner):
            def _run_process(self, transient):
                return "not processed"

            def _prerequisites(self):
                return {
                    "Cutout download": "not processed",
                    "Host match": "not processed",
                }

            @property
            def task_name(self):
                return "Cutout download"

            def _failed_status_message(self):
                return "failed"

        self.two_prereqs_suc_runner = TestRunnerTwoPrereqsSuc(transient_name="dummy")

    def needs_review_test_run_process(self):
        self.processed_runner.run_process()

        # 2022testone is the oldest transient so should be selected and
        # processed. 2022testtwo should not be selected or processed.
        transient = Transient.objects.get(name__exact="2022testtwo")
        task = Task.objects.get(name__exact="Cutout download")
        task_register = TaskRegister.objects.get(transient=transient, task=task)
        self.assertTrue(task_register.status.message == "not processed")

        transient = Transient.objects.get(name__exact="2022testone")
        task = Task.objects.get(name__exact="Cutout download")
        task_register = TaskRegister.objects.get(transient=transient, task=task)
        self.assertTrue(task_register.status.message == "processed")

    def needs_review_test_run_failed(self):
        try:
            self.failed_runner.run_process()
        except ValueError:
            pass

        # 2022testone is the oldest transient so should be selected and
        # processed. 2022testtwo should not be selected or processed.
        transient = Transient.objects.get(name__exact="2022testtwo")
        task = Task.objects.get(name__exact="Cutout download")
        task_register = TaskRegister.objects.get(transient=transient, task=task)
        self.assertTrue(task_register.status.message == "not processed")

        transient = Transient.objects.get(name__exact="2022testone")
        task = Task.objects.get(name__exact="Cutout download")
        task_register = TaskRegister.objects.get(transient=transient, task=task)
        self.assertTrue(task_register.status.message == "failed")

    def test_run_not_processed(self):
        self.not_processed_runner.run_process()

        # 2022testone is the oldest transient so should be selected and
        # processed. 2022testtwo should not be selected or processed.
        transient = Transient.objects.get(name__exact="2022testtwo")
        task = Task.objects.get(name__exact="Cutout download")
        task_register = TaskRegister.objects.get(transient=transient, task=task)
        self.assertTrue(task_register.status.message == "not processed")

        transient = Transient.objects.get(name__exact="2022testone")
        task = Task.objects.get(name__exact="Cutout download")
        task_register = TaskRegister.objects.get(transient=transient, task=task)
        self.assertTrue(task_register.status.message == "not processed")

    def needs_review_test_multiple_transients_processed(self):
        self.processed_runner.run_process()
        self.processed_runner.run_process()

        # both should be processed
        transient = Transient.objects.get(name__exact="2022testtwo")
        task = Task.objects.get(name__exact="Cutout download")
        task_register = TaskRegister.objects.get(transient=transient, task=task)
        self.assertTrue(task_register.status.message == "processed")

        transient = Transient.objects.get(name__exact="2022testone")
        task = Task.objects.get(name__exact="Cutout download")
        task_register = TaskRegister.objects.get(transient=transient, task=task)
        self.assertTrue(task_register.status.message == "processed")

    def test_update_status_status_change(self):
        register_item = TaskRegister.objects.get(
            transient__name__exact="2022testtwo", task__name__exact="Host match"
        )
        failed_status = Status.objects.get(message__exact="failed")
        self.assertTrue(register_item.status.message != "failed")
        update_status(register_item, failed_status)
        self.assertTrue(register_item.status.message == "failed")

    def test_update_status_last_modified(self):
        register_item = TaskRegister.objects.get(
            transient__name__exact="2022testtwo", task__name__exact="Host match"
        )
        failed_status = Status.objects.get(message__exact="failed")
        last_modified_time = register_item.last_modified
        update_status(register_item, failed_status)
        self.assertTrue(register_item.last_modified != last_modified_time)

    def needs_review_test_find_register_items_meeting_prerequisites(self):
        # there should be two tasks that meet the prereqs
        items = self.processed_runner.find_register_items_meeting_prerequisites()
        self.assertTrue(len(items) == 2)

        # there should be no tasks that meet the prereqs
        items = self.two_prereqs_runner.find_register_items_meeting_prerequisites()
        self.assertTrue(len(items) == 0)

        # there should be two tasks that meet the prereqs
        items = self.two_prereqs_suc_runner.find_register_items_meeting_prerequisites()
        self.assertTrue(len(items) == 2)

    def obsolete_test_select_highest_priority(self):
        # tests that the oldest register item is selected
        register = TaskRegister.objects.all()
        oldest = self.processed_runner._select_highest_priority(register)
        self.assertTrue(oldest.transient.name == "2022testone")

    def test_no_register_items(self):
        # test that the case of now register items case works
        self.assertTrue(self.two_prereqs_runner.select_register_item() is None)

        # both should be not processed
        transient = Transient.objects.get(name__exact="2022testtwo")
        task = Task.objects.get(name__exact="Cutout download")
        task_register = TaskRegister.objects.get(transient=transient, task=task)
        self.assertTrue(task_register.status.message == "not processed")

        transient = Transient.objects.get(name__exact="2022testone")
        task = Task.objects.get(name__exact="Cutout download")
        task_register = TaskRegister.objects.get(transient=transient, task=task)
        self.assertTrue(task_register.status.message == "not processed")

        self.two_prereqs_runner.run_process()

        # both should be still not processed
        transient = Transient.objects.get(name__exact="2022testtwo")
        task = Task.objects.get(name__exact="Cutout download")
        task_register = TaskRegister.objects.get(transient=transient, task=task)
        self.assertTrue(task_register.status.message == "not processed")

        transient = Transient.objects.get(name__exact="2022testone")
        task = Task.objects.get(name__exact="Cutout download")
        task_register = TaskRegister.objects.get(transient=transient, task=task)
        self.assertTrue(task_register.status.message == "not processed")

    def test_overwrite_object(self):
        transient = Transient.objects.get(name__exact="2022testone")
        wise_filter = Filter.objects.get(name__exact="WISE_W4")

        query = {"transient": transient, "filter": wise_filter}
        data = {
            "transient": transient,
            "filter": wise_filter,
            "fits": "test",
            "name": "test_name",
        }
        self.processed_runner._overwrite_or_create_object(Cutout, query, data)

        cutout_changed = Cutout.objects.get(name__exact="test_name")
        self.assertTrue(cutout_changed.fits.name == "test")
        self.assertTrue(cutout_changed.name == "test_name")

    def test_create_object(self):
        transient = Transient.objects.get(name__exact="2022testone")
        wise_filter = Filter.objects.get(name__exact="WISE_W1")

        query = {"transient": transient, "filter": wise_filter}
        data = {
            "transient": transient,
            "filter": wise_filter,
            "fits": "test",
            "name": "test_name",
        }
        self.processed_runner._overwrite_or_create_object(Cutout, query, data)

        cutout_changed = Cutout.objects.get(name__exact="test_name")
        self.assertTrue(cutout_changed.fits.name == "test")
        self.assertTrue(cutout_changed.name == "test_name")
        self.assertTrue(cutout_changed.filter.name == "WISE_W1")


class GHOSTRunnerTest(TestCase):

    def setUp(self):
        self.ghost_runner = HostMatch(transient_name="dummy")

    def test_prereqs(self):
        self.assertTrue(
            self.ghost_runner._prerequisites()
            == {
                "Host match": "not processed",
                "Cutout download": "processed",
                "Transient MWEBV": "processed",
            }
        )

    def test_failed_status(self):
        self.assertTrue(self.ghost_runner._failed_status_message() == "no host match")


class InitializeTaskRegisterTest(TestCase):
    fixtures = [
        "../fixtures/test/setup_test_transient.yaml",
    ]

    def test_task_register_init(self):
        transient_name = "2022testone"
        transient = Transient.objects.get(name__exact=transient_name)
        # should be no tasks register first
        registered_tasks = TaskRegister.objects.filter(transient__name__exact=transient_name)
        self.assertTrue(not registered_tasks)
        initialise_all_tasks_status(transient)
        registered_tasks = TaskRegister.objects.filter(transient__name__exact=transient_name)
        self.assertTrue(registered_tasks)


class ImageDownloadTest(TestCase):
    fixtures = [
        "../fixtures/test/setup_test_transient.yaml",
        "../fixtures/test/setup_test_task_register.yaml",
    ]

    def setUp(self):
        class DummyImageDownloadRunner(ImageDownload):
            def _run_process(self, transient):
                return "processed"

        self.image_runner = DummyImageDownloadRunner(transient_name="2022testone")

    def test_prereqs(self):
        self.assertTrue(
            self.image_runner._prerequisites() == {"Cutout download": "not processed"}
        )

    def test_failed_status(self):
        self.assertTrue(self.image_runner._failed_status_message() == "failed")

    def test_run_process(self):
        self.image_runner.run_process()
        transient = Transient.objects.get(name__exact="2022testone")
        task = Task.objects.get(name__exact="Cutout download")
        task_register = TaskRegister.objects.get(transient=transient, task=task)
        self.assertTrue(task_register.status.message == "processed")


class TestAllRegisteredTaskRunners(TestCase):

    def test_task_type(self):
        for task_runner in periodic_tasks:
            transient_type = task_runner.task_type == "transient"
            system_type = task_runner.task_type == "system"
            self.assertTrue(system_type or transient_type)
            self.assertTrue(isinstance(task_runner.task_type, str))

    def test_transient_task_prerequisites(self):
        for task_runner in periodic_tasks:
            if task_runner.task_type == "transient":
                prereq = task_runner.prerequisites
                self.assertTrue(isinstance(prereq, dict))

                for name, status in prereq.items():
                    db_status = Status.objects.get(message__exact=status)
                    self.assertTrue(
                        db_status.message == status,
                        f"{task_runner.task_name}: {db_status.message} == {status}",
                    )
                    db_task = Task.objects.get(name__exact=name)
                    self.assertTrue(
                        db_task.name == name,
                        f"{task_runner.task_name}: {db_task.name} == {name}",
                    )

    def test_transient_task_name(self):
        for task_runner in periodic_tasks:
            if task_runner.task_type == "transient":
                self.assertTrue(isinstance(task_runner.task_name, str))
                db_task = Task.objects.get(name__exact=task_runner.task_name)
                self.assertTrue(db_task.name == task_runner.task_name)

    def test_transient_task_failed_status(self):
        for task_runner in periodic_tasks:
            if task_runner.task_type == "transient":
                failed_message = task_runner._failed_status_message()
                self.assertTrue(isinstance(failed_message, str))
                db_status = Status.objects.get(message__exact=failed_message)
                self.assertTrue(db_status.message == failed_message)
