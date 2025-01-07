from __future__ import absolute_import
from __future__ import unicode_literals

from celery import shared_task
from host.base_tasks import task_soft_time_limit
from host.base_tasks import task_time_limit
from host.workflow import transient_workflow
from .models import Transient
from host.system_tasks import DeleteGHOSTFiles
from host.system_tasks import IngestMissedTNSTransients
from host.system_tasks import InitializeTransientTasks
from host.system_tasks import LogTransientProgress
from host.system_tasks import SnapshotTaskRegister
from host.system_tasks import TNSDataIngestion
from host.system_tasks import TrimTransientImages
from .transient_name_server import get_transients_from_tns_by_name


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
