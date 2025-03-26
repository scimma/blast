from __future__ import absolute_import
from __future__ import unicode_literals

from celery import shared_task
from host.base_tasks import task_soft_time_limit
from host.base_tasks import task_time_limit
from host.workflow import transient_workflow
from .models import Transient, Status, TaskRegister
from host.system_tasks import DeleteGHOSTFiles
from host.system_tasks import IngestMissedTNSTransients
from host.system_tasks import InitializeTransientTasks
from host.system_tasks import LogTransientProgress
from host.system_tasks import SnapshotTaskRegister
from host.system_tasks import TNSDataIngestion
from host.system_tasks import TrimTransientImages


periodic_tasks = [
    TNSDataIngestion(),
    InitializeTransientTasks(),
    SnapshotTaskRegister(),
    LogTransientProgress(),
    DeleteGHOSTFiles(),
    IngestMissedTNSTransients(),
    TrimTransientImages()
]


@shared_task(
    name="Import transients from TNS",
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def import_transient_list(transient_names, retrigger=False):
    def process_transient(transient_name):
        transient_workflow.delay(transient_name)
    existing_transients = []
    new_transient_names = []
    for transient_name in transient_names:
        transient = Transient.objects.filter(name__exact=transient_name)
        print(f'Querying transient "{transient_name}"...')
        if transient:
            print(f'Transient already saved: "{transient_name}"')
            existing_transients.append(transient[0])
        else:
            print(f'New transient detected: "{transient_name}"')
            new_transient_names.append(transient_name)
    # Re-trigger workflows for existing transients
    for transient in existing_transients:
        if retrigger:
            print(f'Retriggering workflow for transient "{transient.name}"')
            # Assume that any task in a "processing" state is not actually running,
            # resetting the status to "not processed" so that it will run again.
            # TODO: Search Celery queue and terminate any actively running tasks first.
            processing_status = Status.objects.get(message__exact="processing")
            not_processed_status = Status.objects.get(message__exact="not processed")
            tasks = TaskRegister.objects.filter(transient=transient)
            for task in [task for task in tasks if task.status == processing_status]:
                task.status = not_processed_status
                task.save()
            process_transient(transient.name)
        else:
            print(f'Skipping existing transient "{transient.name}"')
    # Process new transients
    uploaded_transient_names = []
    if new_transient_names:
        for transient_name in new_transient_names:
            print(f'Triggering workflow for new transient "{transient_name}"...')
            process_transient(transient_name)
            uploaded_transient_names += [transient_name]
    return uploaded_transient_names
