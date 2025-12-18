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
from host.system_tasks import RetriggerIncompleteWorkflows
from host.system_tasks import UsageLogRoller
from host.system_tasks import GarbageCollector
from host.transient_tasks import get_processing_status_and_progress
from django.urls import reverse_lazy
from django.http import HttpResponseRedirect
from django.contrib.auth.decorators import login_required, permission_required
from host.decorators import log_usage_metric
from host.host_utils import inspect_worker_tasks
from host.host_utils import reset_workflow_if_not_processing
from host.log import get_logger
logger = get_logger(__name__)

periodic_tasks = [
    TNSDataIngestion(),
    InitializeTransientTasks(),
    SnapshotTaskRegister(),
    IngestMissedTNSTransients(),
    RetriggerIncompleteWorkflows(),
    GarbageCollector(),
    UsageLogRoller(),
]


@login_required
@permission_required("host.retrigger_transient", raise_exception=True)
@log_usage_metric()
def retrigger_transient_view(request=None, slug=''):
    return retrigger_transient(request, slug)


def retrigger_transient(request=None, slug=''):
    transient_name = slug
    assert transient_name
    result = None
    try:
        transient = Transient.objects.get(name__exact=transient_name)
        logger.debug(f'Retrigger requested for transient "{transient.name}"')
        progress, processing_status = get_processing_status_and_progress(transient)
        # When manually retriggering a workflow, attempt to rerun failed tasks,
        # because these may have failed for operational instead of intrinsic reasons.
        if processing_status in ['processing', 'blocked']:
            logger.debug(f'''"{transient.name}": "{processing_status}"''')
            # If the transient workflow is already in progress, do nothing; otherwise, retrigger the workflow.
            # Filter out the current task executing this function, or the transient will never be retriggered!
            all_tasks = [task for task in inspect_worker_tasks()
                         if task['name'] != 'Import transients from TNS']
            if reset_workflow_if_not_processing(transient, all_tasks, reset_failed=True):
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
def import_transient_list(transient_names):
    '''This function assumes that the input transient_names are not in the database.'''
    def process_transient(transient_name):
        transient_workflow.delay(transient_name)
    uploaded_transient_names = []
    for transient_name in transient_names:
        logger.info(f'Triggering workflow for new transient "{transient_name}"...')
        try:
            process_transient(transient_name)
            uploaded_transient_names.append(transient_name)
        except Exception as err:
            logger.error(f'''Error processing new transient: {err}''')
    return uploaded_transient_names
