import datetime
import glob
import shutil

from celery import shared_task
from dateutil import parser
from django.conf import settings
from django.db.models import Q
from django.utils import timezone
from host.base_tasks import initialise_all_tasks_status
from host.base_tasks import SystemTaskRunner
from host.base_tasks import task_soft_time_limit
from host.base_tasks import task_time_limit
from host.workflow import reprocess_transient
from host.workflow import transient_workflow
from host.trim_images import trim_images

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


class TNSDataIngestion(SystemTaskRunner):
    def run_process(self, interval_minutes=200):
        print("TNS STARTED")
        now = timezone.now()
        time_delta = datetime.timedelta(minutes=interval_minutes)
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
            identical_timestamps = saved_timestamp - tns_timestamp == datetime.timedelta(0)
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
        yesterday = timezone.now() - datetime.timedelta(days=1)
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

        now = timezone.now()

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
)
def tns_data_ingestion():
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
