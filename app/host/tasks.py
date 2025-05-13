from __future__ import absolute_import
from __future__ import unicode_literals

from celery import shared_task
from host.base_tasks import task_soft_time_limit
from host.base_tasks import task_time_limit
from host.workflow import transient_workflow
from host.models import Transient
from host.system_tasks import IngestMissedTNSTransients
from host.system_tasks import InitializeTransientTasks
from host.system_tasks import SnapshotTaskRegister
from host.system_tasks import TNSDataIngestion
from host.system_tasks import TrimTransientImages
from host.system_tasks import RetriggerIncompleteWorkflows
from host.transient_tasks import get_processing_status_and_progress
from django.urls import reverse_lazy
from django.http import HttpResponseRedirect

from host.log import get_logger
logger = get_logger(__name__)

periodic_tasks = [
    TNSDataIngestion(),
    InitializeTransientTasks(),
    SnapshotTaskRegister(),
    IngestMissedTNSTransients(),
    TrimTransientImages(),
    RetriggerIncompleteWorkflows(),
]


def retrigger_transient(request=None, slug=''):
    transient_name = slug
    assert transient_name
    result = None
    try:
        transient = Transient.objects.get(name__exact=transient_name)
        logger.debug(f'Retrigger requested for transient "{transient.name}"')
        progress, processing_status = get_processing_status_and_progress(transient)
        if progress < 100:
            logger.debug(f'''"{transient.name}": "{processing_status}"''')
            # If the transient workflow is already in progress, do nothing; otherwise, retrigger the workflow.
            riw = RetriggerIncompleteWorkflows()
            # Filter out the current task executing this function, or the transient will never be retriggered!
            all_tasks = [task for task in riw.inspect_worker_tasks()
                         if task['name'] != 'Import transients from TNS']
            if riw.reset_workflow_if_not_processing(transient, all_tasks):
                logger.info(f'Retriggering workflow for transient "{transient.name}"')
                result = transient_workflow.delay(transient_name)
            else:
                logger.warning(f'Workflow for transient "{transient.name}" was not retriggered because '
                               'it is queued or actively running.')
                logger.debug(f'''tasks: {all_tasks}''')
        else:
            logger.info(f'Workflow is already complete for transient "{transient.name}".')
    except Transient.DoesNotExist:
        result = None
    if request:
        return HttpResponseRedirect(reverse_lazy("results", kwargs={"slug": transient_name}))
    else:
        return result


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
        logger.info(f'Querying transient "{transient_name}"...')
        if transient:
            logger.info(f'Transient already saved: "{transient_name}"')
            existing_transients.append(transient[0])
        else:
            logger.info(f'New transient detected: "{transient_name}"')
            new_transient_names.append(transient_name)
    # Re-trigger workflows for existing transients
    for transient in existing_transients:
        if retrigger:
            retrigger_transient(slug=transient.name)
        else:
            logger.info(f'Skipping existing transient "{transient.name}"')
    # Process new transients
    uploaded_transient_names = []
    if new_transient_names:
        for transient_name in new_transient_names:
            logger.info(f'Triggering workflow for new transient "{transient_name}"...')
            process_transient(transient_name)
            uploaded_transient_names += [transient_name]
    return uploaded_transient_names
