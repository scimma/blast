from django.test import TestCase

from ..models import Cutout
from ..models import Filter
from ..models import Status
from ..models import Task
from ..models import TaskRegister
from ..models import Transient
from ..processing import GhostRunner
from ..processing import ImageDownloadRunner
from ..processing import initialise_all_tasks_status
from ..processing import TaskRunner
from ..processing import update_status


class TaskRunnerTest(TestCase):
    fixtures = [
        "../fixtures/test/setup_test_transient.yaml",
        "../fixtures/initial/setup_tasks.yaml",
        "../fixtures/initial/setup_status.yaml",
        "../fixtures/test/setup_test_task_register.yaml",
        "../fixtures/initial/setup_survey_data.yaml",
        "../fixtures/initial/setup_filter_data.yaml",
        "../fixtures/test/test_cutout.yaml",
    ]

    def setUp(self):
        class TestRunnerProcessed(TaskRunner):
            def _run_process(self, transient):
                return "processed"

            def _prerequisites(self):
                return {"Cutout download": "not processed"}

            def _task_name(self):
                return "Cutout download"

            def _failed_status_message(self):
                return "failed"

        self.processed_runner = TestRunnerProcessed()

        class TestRunnerFailed(TaskRunner):
            def _run_process(self, transient):
                raise ValueError

            def _prerequisites(self):
                return {"Cutout download": "not processed"}

            def _task_name(self):
                return "Cutout download"

            def _failed_status_message(self):
                return "failed"

        self.failed_runner = TestRunnerFailed()

        class TestRunnerNotProcessed(TaskRunner):
            def _run_process(self, transient):
                return "not processed"

            def _prerequisites(self):
                return {"Cutout download": "not processed"}

            def _task_name(self):
                return "Cutout download"

            def _failed_status_message(self):
                return "failed"

        self.not_processed_runner = TestRunnerNotProcessed()

        class TestRunnerTwoPrereqs(TaskRunner):
            def _run_process(self, transient):
                return "not processed"

            def _prerequisites(self):
                return {"Cutout download": "not processed", "Host match": "processed"}

            def _task_name(self):
                return "Cutout download"

            def _failed_status_message(self):
                return "failed"

        self.two_prereqs_runner = TestRunnerTwoPrereqs()

        class TestRunnerTwoPrereqsSuc(TaskRunner):
            def _run_process(self, transient):
                return "not processed"

            def _prerequisites(self):
                return {
                    "Cutout download": "not processed",
                    "Host match": "not processed",
                }

            def _task_name(self):
                return "Cutout download"

            def _failed_status_message(self):
                return "failed"

        self.two_prereqs_suc_runner = TestRunnerTwoPrereqsSuc()

    def test_run_process(self):

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

    def test_run_failed(self):

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

    def test_multiple_transients_processed(self):
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

    def test_find_register_items_meeting_prerequisites(self):
        # there should be two tasks that meet the prereqs
        items = self.processed_runner.find_register_items_meeting_prerequisites()
        self.assertTrue(len(items) == 2)

        # there should be no tasks that meet the prereqs
        items = self.two_prereqs_runner.find_register_items_meeting_prerequisites()
        self.assertTrue(len(items) == 0)

        # there should be two tasks that meet the prereqs
        items = self.two_prereqs_suc_runner.find_register_items_meeting_prerequisites()
        self.assertTrue(len(items) == 2)

    def test_select_highest_priority(self):
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
    fixtures = [
        "../fixtures/initial/setup_tasks.yaml",
        "../fixtures/initial/setup_status.yaml",
    ]

    def setUp(self):
        self.ghost_runner = GhostRunner()

    def test_prereqs(self):
        self.assertTrue(
            self.ghost_runner._prerequisites() == {"Host Match": "not processed"}
        )

    def test_failed_status(self):
        self.assertTrue(self.ghost_runner._failed_status_message() == "no GHOST match")


class InitializeTaskRegisterTest(TestCase):
    fixtures = [
        "../fixtures/test/setup_test_transient.yaml",
        "../fixtures/initial/setup_status.yaml",
        "../fixtures/initial/setup_tasks.yaml",
    ]

    def test_task_register_init(self):
        transient = Transient.objects.get(name__exact="2022testone")
        # should be no tasks register first
        self.assertTrue(TaskRegister.objects.all().exists() is False)
        initialise_all_tasks_status(transient)
        self.assertTrue(TaskRegister.objects.all().exists() is True)


class ImageDownloadTest(TestCase):
    fixtures = [
        "../fixtures/initial/setup_tasks.yaml",
        "../fixtures/initial/setup_status.yaml",
        "../fixtures/test/setup_test_transient.yaml",
        "../fixtures/test/setup_test_task_register.yaml",
    ]

    def setUp(self):
        class DummyImageDownloadRunner(ImageDownloadRunner):
            def _run_process(self, transient):
                return "processed"

        self.image_runner = DummyImageDownloadRunner()

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
