import os
from abc import ABC
from abc import abstractmethod
from time import process_time

from billiard.exceptions import SoftTimeLimitExceeded
from django.db.models import Q
from django.utils import timezone

from .models import Status
from .models import Task
from .models import TaskRegister
from .models import Transient
from host.log import get_logger
logger = get_logger(__name__)

task_time_limit = int(os.environ.get("TASK_TIME_LIMIT", "3800"))
task_soft_time_limit = int(os.environ.get("TASK_SOFT_TIME_LIMIT", "3600"))

"""This module contains the base classes for TaskRunner in blast."""


def get_progress(transient_name):
    tasks = TaskRegister.objects.filter(Q(transient__name__exact=transient_name) &
                                        ~Q(task__name="Log transient processing status"))
    failed_tasks = tasks.filter(status__type='error')
    completed_tasks = tasks.filter(status__type='success')
    incomplete_tasks = tasks.filter(status__type='blank')
    
    total_tasks = len(tasks)
    completed_tasks = len(
        [task for task in tasks if task.status.message == "processed"]
    )
    if not len(failed_tasks):
        progress = 100 * (1 - len(incomplete_tasks) / total_tasks) if total_tasks > 0 else 0
    else:
        remaining_tasks = len(incomplete_tasks)
        for task_name in [
                'Cutout download','Transient information','MWEBV transient',
                'Host match','Host information']:
            if failed_tasks.filter(task__name=task_name).exists():
                remaining_tasks = 0
                break
        if remaining_tasks == 0:
            progress = 100
        else:
            # local chain
            if failed_tasks.filter(task__name='Local aperture photometry'): remaining_tasks -= 2
            elif failed_tasks.filter(task__name='Validate local photometry'): remaining_tasks -= 1

            # global chain
            if failed_tasks.filter(task__name='MWEBV host').exists() or \
               failed_tasks.filter(task__name='Validate global photometry').exists(): remaining_tasks -= 1
            elif failed_tasks.filter(task__name='Global aperture photometry'): remaining_tasks -= 2
            elif failed_tasks.filter(task__name='Global aperture construction'): remaining_tasks -= 3

            progress = 100 * (1 - remaining_tasks / total_tasks)
        
    return int(round(progress, 0))


def get_processing_status(transient):
    tasks = TaskRegister.objects.filter(
        Q(transient__name__exact=transient.name)
        & ~Q(task__name="Log transient processing status")
    )
    processing_task_qs = TaskRegister.objects.filter(
        transient__name__exact=transient.name,
        task__name="Log transient processing status",
    )

    total_tasks = len(tasks)
    completed_tasks = len(
        [task for task in tasks if task.status.message == "processed"]
    )
    blocked = len([task for task in tasks if task.status.type == "error"])

    progress = "processing"

    if total_tasks == 0:
        progress = "processing"
    elif total_tasks == completed_tasks:
        progress = "completed"
    elif total_tasks < completed_tasks:
        progress = "processing"
    elif blocked > 0:
        progress = "blocked"

    # save task
    if len(processing_task_qs) == 1:
        processing_task = processing_task_qs[0]
        processing_task.status = Status.objects.get(
            message=progress if progress != "completed" else "processed"
        )
        processing_task.save()

    # save transient progress
    return progress


def get_image_trim_status(transient):
    # for image trimming to be ready, we need either
    # 1) image trimming is already finished
    # 2) local AND global photometry validation are not 'not processed'
    # 3) progress is complete

    if transient.image_trim_status == "processed":
        return "processed"
    elif transient.processing_status == "completed" or transient.processing_status == "blocked":
        return "ready"
    else:
        global_valid_status = TaskRegister.objects.filter(task__name="Validate local photometry",transient=transient)
        local_valid_status = TaskRegister.objects.filter(task__name="Validate global photometry",transient=transient)
        if not len(global_valid_status) or not len(local_valid_status):
            return "not ready"
        
        global_valid_status = TaskRegister.objects.filter(task__name="Validate local photometry",transient=transient)[0].status.message
        local_valid_status = TaskRegister.objects.filter(task__name="Validate global photometry",transient=transient)[0].status.message
        if global_valid_status != "not processed" and local_valid_status != "not processed":
            return "ready"
        else:
            return "not ready"


class TaskRunner(ABC):
    """
    Abstract base class for a TaskRunner.

    Attributes:
        task_frequency_seconds (int): Positive integer defining the frequency
            the task in run at. Defaults to 60 seconds.
        task_initially_enabled (bool): True means the task is initially enabled,
            False means the task is initially disabled. Default is enabled
            (True).
        task_name (str): Name of the task the TaskRunner works on.
        task_type (str): Type of task the TaskRunner works on.
        task_function_name(str): Name of the function used to register the task
            in celery.

    """

    def __init__(self):
        """
        Initialized method which sets up the task runner.
        """
        pass

    def _overwrite_or_create_object(self, model, unique_object_query, object_data):
        """
        Overwrites or creates new objects in the blast database.

        Parameters
            model (dango.model): blast model of the object that needs to be updated
            unique_object_query (dict): query to be passed to model.objects.get that will
                uniquely identify the object of interest
            object_data (dict): data to be saved or overwritten for the object.
        """

        try:
            object = model.objects.get(**unique_object_query)
            object.delete()
            model.objects.create(**object_data)
        except model.DoesNotExist:
            model.objects.create(**object_data)

    @property
    def task_frequency_seconds(self) -> int:
        """
        Defines the frequency in seconds the task should be run at.
        """
        return 60

    @property
    def task_initially_enabled(self):
        """
        Defines if the task should be run on blast startup.
        """
        return True

    @property
    def task_function_name(self) -> str:
        """
        TaskRunner function name to be registered by celery.
        """
        return "host.tasks." + self.task_name.replace(" ", "_").lower()

    @abstractmethod
    def run_process(self):
        """
        Runs a task runner process. Needs to be implemented.
        """
        pass

    @property
    @abstractmethod
    def task_name(self) -> str:
        """
        Name of the task the TaskRunner works on.
        """
        pass

    @property
    @abstractmethod
    def task_type(self) -> str:
        """
        Type of task the TaskRunner works on.
        """
        pass


class TransientTaskRunner(TaskRunner):
    """
    Abstract base class for a TransientTaskRunner.

    Attributes:
        task_frequency_seconds (int): Positive integer defining the frequency
            the task in run at. Defaults to 60 seconds.
        task_initially_enabled (bool): True means the task is initially enabled,
            False means the task is initially disabled. Default is enabled
            (True).
        task_name (str): Name of the task the TaskRunner works on.
        task_type (str): Type of task the TaskRunner works on.
        task_function_name(str): Name of the function used to register the task
            in celery.
        processing_status (models.Status): Status of the task while runner is
            running a task.
        failed_status (model.Status): Status of the task is if the runner fails.
        prerequisites (dict): Prerequisite tasks and statuses required for the
            runner to process.
    """

    def __init__(self, transient_name=None):
        """
        Initialized method which sets up the task runner.
        """

        self.prerequisites = self._prerequisites()
        assert transient_name
        self.transient_name = transient_name

    def find_register_items_meeting_prerequisites(self):
        """
        Finds the register items meeting the prerequisites.

        Returns:
            (QuerySet): Task register items meeting prerequisites.
        """
        task = Task.objects.get(name__exact=self.task_name)
        task_register = TaskRegister.objects.all()
        if self.transient_name:
            current_transients = Transient.objects.filter(
                name__exact=self.transient_name
            )
        else:
            current_transients = Transient.objects.all()

        for task_name, status_message in self.prerequisites.items():
            task_prereq = Task.objects.get(name__exact=task_name)
            status = Status.objects.get(message__exact=status_message)

            current_transients = current_transients & Transient.objects.filter(
                taskregister__task=task_prereq, taskregister__status=status
            )

        return task_register.filter(transient__in=list(current_transients), task=task)

    def _select_highest_priority(self, register):
        """
        Select highest priority task by finding the one with the oldest
        transient timestamp.

        Args:
            register (QuerySet): register of tasks to select from.
        Returns:
            register item (model.TaskRegister): highest priority register item.
        """
        return register.order_by("transient__public_timestamp")[0]

    def _update_status(self, task_status, updated_status):
        """
        Update the processing status of a task.

        Parameters:
            task_status (models.TaskProcessingStatus): task processing status to be
                updated.
            updated_status (models.Status): new status to update the task with.
        Returns:
            None: Saves the new updates to the backend.
        """
        task_status.status = updated_status
        task_status.last_modified = timezone.now()
        task_status.save()

    def select_register_item(self):
        """
        Selects register item to be processed by task runner.

        Returns:
            register item (models.TaskRegister): returns item if one exists,
                returns None otherwise.
        """
        register = self.find_register_items_meeting_prerequisites()
        return self._select_highest_priority(register) if register.exists() else None

    def run_process(self, task_register_item=None):
        """
        Runs task runner process.
        """
        # self.task = Task.objects.get(name__exact=self.task_name)
        if task_register_item is None:
            task_register_item = self.select_register_item()
        processing_status = Status.objects.get(message__exact="processing")

        if task_register_item is not None:
            logger.debug(f'''task_register_item: {task_register_item}''')
            self._update_status(task_register_item, processing_status)
            transient = task_register_item.transient

            start_time = process_time()
            try:
                status_message = self._run_process(transient)
            except SoftTimeLimitExceeded:
                status_message = "time limit exceeded"
                raise
            except Exception:
                status_message = self._failed_status_message()
                raise
            finally:
                end_time = process_time()
                logger.debug(f'''TransientTaskRunner status message: "{status_message}"''')
                status = Status.objects.get(message__exact=status_message)
                self._update_status(task_register_item, status)
                processing_time = round(end_time - start_time, 2)
                task_register_item.last_processing_time_seconds = processing_time
                task_register_item.save()
                transient.progress = get_progress(transient.name)
                transient.processing_status = get_processing_status(transient)
                transient.image_trim_status = get_image_trim_status(transient)
                transient.save()
            return transient.name

    @abstractmethod
    def _run_process(self, transient):
        """
        Run process function to be implemented by child classes.

        Args:
            transient (models.Transient): transient for the task runner to
                process
        Returns:
            runner status (models.Status): status of the task after the task
                runner has completed.
        """
        pass

    @abstractmethod
    def _prerequisites(self):
        """
        Task prerequisites to be implemented by child classes.

        Returns:
            prerequisites (dict): key is the name of the task, value is the task
                status.
        """
        pass

    @abstractmethod
    def _failed_status_message(self):
        """
        Message of the failed status.

        Returns:
            failed message (str): Name of the message of the failed status.
        """
        pass

    @property
    def task_type(self):
        return "transient"

    @property
    def task_function_name(self) -> str:
        """
        TaskRunner function name to be registered by celery.
        """
        return "host.transient_tasks." + self.task_name.replace(" ", "_").lower()


class SystemTaskRunner(TaskRunner):
    """
    Abstract base class for a SystemTaskRunner.

    Attributes:
        task_frequency_seconds (int): Positive integer defining the frequency
            the task in run at. Defaults to 60 seconds.
        task_initially_enabled (bool): True means the task is initially enabled,
            False means the task is initially disabled. Default is enabled
            (True).
        task_name (str): Name of the task the TaskRunner works on.
        task_type (str): Type of task the TaskRunner works on.
        task_function_name(str): Name of the function used to register the task
            in celery.
    """

    @property
    def task_type(self):
        return "system"

    @property
    def task_function_name(self) -> str:
        """
        TaskRunner function name to be registered by celery.
        """
        return "host.system_tasks." + self.task_name.replace(" ", "_").lower()


def update_status(task_status, updated_status):
    """
    Update the processing status of a task.

    Parameters:
        task_status (models.TaskProcessingStatus): task processing status to be
            updated.
        updated_status (models.Status): new status to update the task with.
    Returns:
        None: Saves the new updates to the backend.
    """
    task_status.status = updated_status
    task_status.last_modified = timezone.now()
    task_status.save()


def initialise_all_tasks_status(transient):
    """
    Set all available tasks for a transient to not processed.

    Parameters:
        transient (models.Transient): Transient to have all of its task status
            initialized.
    Returns:
        None: Saves the new updates to the backend.
    """
    tasks = Task.objects.all()
    not_processed = Status.objects.get(message__exact="not processed")

    for task in tasks:
        task_status = TaskRegister.objects.filter(task=task, transient=transient)
        if not len(task_status):
            task_status = TaskRegister(task=task, transient=transient)
            # if the task already exists, let's not change it
            # because bad things seem to happen....
            update_status(task_status, not_processed)
        else:
            task_status = task_status[0]
