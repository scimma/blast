import os
from abc import ABC
from abc import abstractmethod

from django.utils import timezone

from .models import Status
from .models import Task
from .models import TaskRegister
from host.log import get_logger
logger = get_logger(__name__)

task_time_limit = int(os.environ.get("TASK_TIME_LIMIT", "3800"))
task_soft_time_limit = int(os.environ.get("TASK_SOFT_TIME_LIMIT", "3600"))

"""This module contains the base classes for TaskRunner in Blast."""


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

    # Reset all tasks to status "not processed"
    for task in tasks:
        # If tasks have been added to the transient workflow since the transient was last
        # processed, ensure they are created.
        task_status, created = TaskRegister.objects.get_or_create(task=task, transient=transient,
                                                                  defaults={"status": not_processed})
        if created:
            logger.info(f'Created task "{task.name}" for transient "{transient.name}".')
        update_status(task_status, not_processed)
