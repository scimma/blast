import glob
import shutil

from celery import shared_task
from dateutil import parser
from django.conf import settings
from django.db.models import Q
from datetime import datetime, timedelta, timezone
from host.base_tasks import initialise_all_tasks_status
from host.base_tasks import SystemTaskRunner
from host.base_tasks import task_soft_time_limit
from host.base_tasks import task_time_limit
from host.workflow import reprocess_transient
from host.workflow import transient_workflow
from host.trim_images import trim_images
from app.celery import app
from celery.contrib.abortable import AbortableTask, AbortableAsyncResult

from .models import Status
from .models import TaskRegister
from .models import TaskRegisterSnapshot
from .models import Transient
from .transient_name_server import get_daily_tns_staging_csv
from .transient_name_server import get_tns_credentials
from .transient_name_server import get_transients_from_tns
from .transient_name_server import tns_staging_blast_transient
from .transient_name_server import tns_staging_file_date_name
from .transient_name_server import update_blast_transient
# Configure logging
import os
import logging
logging.basicConfig(format='%(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv('LOG_LEVEL', logging.INFO))


class TNSDataIngestion(SystemTaskRunner):
    def run_process(self, interval_minutes=200):
        print("TNS STARTED")
        # # When testing periodic task management and behavior, use this short-circuit to avoid contacting TNS.
        # # START TESTING SHORT CIRCUIT
        # import random
        # sleep_time = random.choice([8, 12, 43, 43])
        # print(f'Sleeping {sleep_time} seconds...')
        # sleep(sleep_time)
        # print("TNS COMPLETED")
        # return
        # # END TESTING SHORT CIRCUIT
        now = datetime.now(timezone.utc)
        time_delta = timedelta(minutes=interval_minutes)
        tns_credentials = get_tns_credentials()
        transients_from_tns = get_transients_from_tns(
            now - time_delta, tns_credentials=tns_credentials
        )
        print("TNS DONE")
        saved_transients = Transient.objects.all()
        count = 0
        for transient_from_tns in transients_from_tns:
            print(transient_from_tns.name)

            # If the transient has not already been ingested, save the TNS
            # data and proceed to the next transient
            saved_transient = saved_transients.filter(name__exact=transient_from_tns.name)
            if not saved_transient:
                transient_from_tns.save()
                count += 1
                continue
            # If the transient was previously ingested, compare to the incoming TNS data.
            saved_transient = saved_transient[0]

            # Determine if the redshift value changed or was added.
            redshift_updated = False
            if transient_from_tns.redshift:
                redshift_updated = \
                    not saved_transient.redshift or \
                    saved_transient.redshift != transient_from_tns.redshift

            # Determine if the timestamps are different.
            saved_timestamp = saved_transient.public_timestamp.replace(tzinfo=None)
            tns_timestamp = parser.parse(transient_from_tns.public_timestamp)
            identical_timestamps = saved_timestamp - tns_timestamp == timedelta(0)
            # If the timestamps are identical and there was no redshift update, skip to the next TNS transient.
            if identical_timestamps and not redshift_updated:
                continue

            # Update the saved transient data with the latest TNS data
            saved_transient.tns_id = transient_from_tns.tns_id
            saved_transient.ra_deg = transient_from_tns.ra_deg
            saved_transient.dec_deg = transient_from_tns.dec_deg
            saved_transient.tns_prefix = transient_from_tns.tns_prefix
            saved_transient.public_timestamp = transient_from_tns.public_timestamp
            saved_transient.spectroscopic_class = transient_from_tns.spectroscopic_class
            saved_transient.redshift = transient_from_tns.redshift
            # Reinitialize the transient state so that its processing workflow will run again if necessary.
            saved_transient.tasks_initialized = False
            saved_transient.save()

            # If the redshift value was updated, reprocess the entire workflow.
            if redshift_updated:
                reprocess_transient(slug=saved_transient.name)

        print(f"Added {count} new transients")
        print("TNS UPLOADED")

    @property
    def task_name(self):
        return "TNS data ingestion"

    @property
    def task_frequency_seconds(self):
        return 240

    @property
    def task_initially_enabled(self):
        return True


class InitializeTransientTasks(SystemTaskRunner):
    def run_process(self):
        """
        Initializes all task in the database to not processed for new transients.
        """

        uninitialized_transients = Transient.objects.filter(
            tasks_initialized__exact="False"
        )
        for transient in uninitialized_transients:
            initialise_all_tasks_status(transient)
            transient.tasks_initialized = "True"
            transient.save()
            transient_workflow.delay(transient.name)

    @property
    def task_name(self):
        return "Initialize transient task"


class IngestMissedTNSTransients(SystemTaskRunner):
    def run_process(self):
        """
        Gets missed transients from tns and update them using the daily staging csv
        """
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        date_string = tns_staging_file_date_name(yesterday)
        data = get_daily_tns_staging_csv(
            date_string,
            tns_credentials=get_tns_credentials(),
            save_dir=settings.TNS_STAGING_ROOT,
        )
        saved_transients = Transient.objects.all()

        for _, transient in data.iterrows():
            # if transient exists update it
            try:
                blast_transient = saved_transients.get(
                    name__exact=transient["name"])
                update_blast_transient(blast_transient, transient)
            # if transient does not exist add it
            except Transient.DoesNotExist:
                blast_transient = tns_staging_blast_transient(transient)
                blast_transient.save()
                transient_workflow.delay(transient.name)

    @property
    def task_name(self):
        return "Ingest missed TNS transients"

    @property
    def task_initially_enabled(self):
        return False


class DeleteGHOSTFiles(SystemTaskRunner):
    def run_process(self):
        """
        Removes GHOST files
        """
        dir_list = glob.glob("transients_*/")

        for dir in dir_list:
            try:
                shutil.rmtree(dir)
            except OSError as e:
                print("Error: %s : %s" % (dir, e.strerror))

        dir_list = glob.glob("quiverMaps/")

        for dir in dir_list:
            try:
                shutil.rmtree(dir)
            except OSError as e:
                print("Error: %s : %s" % (dir, e.strerror))

    @property
    def task_name(self):
        return "Delete GHOST files"


class SnapshotTaskRegister(SystemTaskRunner):
    def run_process(self, interval_minutes=100):
        """
        Takes snapshot of task register for diagnostic purposes.
        """
        transients = Transient.objects.all()
        total, completed, waiting, not_completed = 0, 0, 0, 0

        for transient in transients:
            total += 1
            if transient.progress == 100:
                completed += 1
            if transient.progress == 0:
                waiting += 1
            if transient.progress < 100 and transient.progress > 0:
                not_completed += 1

        now = datetime.now(timezone.utc)

        for aggregate, label in zip(
            [not_completed, total, completed, waiting],
            ["not completed", "total", "completed", "waiting"],
        ):
            TaskRegisterSnapshot.objects.create(
                time=now, number_of_transients=aggregate, aggregate_type=label
            )

    @property
    def task_name(self):
        return "Snapshot task register"


class LogTransientProgress(SystemTaskRunner):
    def run_process(self):
        """
        Updates the processing status for all transients.
        """
        transients = Transient.objects.all()

        for transient in transients:
            tasks = TaskRegister.objects.filter(
                Q(transient__name__exact=transient.name) & ~Q(
                    task__name=self.task_name)
            )
            processing_task_qs = TaskRegister.objects.filter(
                transient__name__exact=transient.name, task__name=self.task_name
            )

            total_tasks = len(tasks)
            completed_tasks = len(
                [task for task in tasks if task.status.message == "processed"]
            )
            blocked = len(
                [task for task in tasks if task.status.type == "error"])

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
            transient.processing_status = progress
            transient.save()

    @property
    def task_name(self):
        return "Log transient processing status"

    @property
    def task_initially_enabled(self):
        return False


class TrimTransientImages(SystemTaskRunner):
    def run_process(self):
        """
        Updates the processing status for all transients.
        """
        transients = Transient.objects.filter(image_trim_status="ready")

        for transient in transients:
            trim_images(transient)

    @property
    def task_name(self):
        return "Trim transient images"

    @property
    def task_frequency_seconds(self):
        return 3600

    @property
    def task_initially_enabled(self):
        return True


# Periodic tasks
@shared_task(
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def trim_transient_images():
    TrimTransientImages().run_process()


@shared_task(
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
    bind=True,
    base=AbortableTask,
)
def tns_data_ingestion(self):
    def get_all_active_tasks():
        inspect = app.control.inspect()
        active_task_info = inspect.active(safe=True)
        all_active_tasks = []
        for worker, worker_active_tasks in active_task_info.items():
            logger.debug(f'''{worker}: {worker_active_tasks}''')
            all_active_tasks.extend(worker_active_tasks)
        logger.debug(f'''active tasks: {len(all_active_tasks)}''')
        return all_active_tasks
    # Ensure that there are no concurrent executions of the TNS ingestion to avoid exceeding the TNS API rate limits.
    # Use the Celery app control system to list all active tns_data_ingestion tasks, and only contact TNS if there are
    # no running instances. Running or stalled instances that are older than the TNS_INGEST_TIMEOUT value should be 
    # aborted and terminated so they do not permanently block the ingest; however, the methods available to terminate
    # these processes in Celery are unreliable, so additional monitoring may be necessary.
    task_name = 'host.system_tasks.tns_data_ingestion'
    task_id = self.request.id
    all_active_tasks = get_all_active_tasks()
    active_tasks = [task for task in all_active_tasks if task['name'] == task_name and task['id'] != task_id]
    time_threshold = datetime.now(timezone.utc) - timedelta(seconds=settings.TNS_INGEST_TIMEOUT)
    # Attempt to abort expired tasks
    for task in active_tasks:
        time_start = datetime.fromtimestamp(task['time_start'], tz=timezone.utc)
        if time_start < time_threshold:
            logger.debug(f'''Active task {task['id']} has expired: {time_start.strftime('%Y/%m/%d %H:%M:%S')}''')
            logger.info(f'''Aborting expired task {task['id']}...''')
            abortable_task = AbortableAsyncResult(task['id'])
            abortable_task.abort()
            logger.info(f'''Revoking expired task {task['id']}...''')
            app.control.terminate(task['id'], signal='SIGKILL')
    # If there is still an active task other than this current task, abort.
    if [task for task in get_all_active_tasks() if task['name'] == task_name and task['id'] != task_id]:
    # # When testing TNS query mutex locking mechanism, use the following line instead to allow some concurrency:
    # if len([task for task in get_all_active_tasks() if task['name'] == task_name and task['id'] != task_id]) > 2:
        logger.info(f'''There is already an active "{task_name}" task. Aborting.''')
        return
    # Run the TNS ingestion
    TNSDataIngestion().run_process()


@shared_task(
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def initialize_transient_task():
    InitializeTransientTasks().run_process()


@shared_task(
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def snapshot_task_register():
    SnapshotTaskRegister().run_process()


@shared_task(
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def log_transient_processing_status():
    LogTransientProgress().run_process()


@shared_task(
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def delete_ghost_files():
    DeleteGHOSTFiles().run_process()


@shared_task(
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def ingest_missed_tns_transients():
    IngestMissedTNSTransients().run_process()
