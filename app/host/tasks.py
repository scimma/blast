from __future__ import absolute_import
from __future__ import unicode_literals
from shutil import rmtree
import os

from celery import shared_task
from host.base_tasks import task_soft_time_limit
from host.base_tasks import task_time_limit
from host.workflow import transient_workflow
from host.models import Transient
from host.system_tasks import IngestMissedTNSTransients
from host.system_tasks import TNSDataIngestion
from host.system_tasks import RetriggerIncompleteWorkflows
from host.system_tasks import UsageLogRoller
from host.system_tasks import GarbageCollector
from host.host_utils import get_processing_status_and_progress
from django.urls import reverse_lazy
from django.http import HttpResponseRedirect
from django.contrib.auth.decorators import login_required, permission_required
from django.conf import settings
from host.decorators import log_usage_metric
from host.host_utils import inspect_worker_tasks
from host.host_utils import reset_workflow_if_not_processing
from host.host_utils import export_dataset
from host.host_utils import equal_dicts
from host.object_store import ObjectStore
from host.models import DatasetRevision
from host.log import get_logger
logger = get_logger(__name__)

periodic_tasks = [
    TNSDataIngestion(),
    IngestMissedTNSTransients(),
    RetriggerIncompleteWorkflows(),
    GarbageCollector(),
    UsageLogRoller(),
]


@login_required
@permission_required("host.retrigger_transient", raise_exception=True)
@log_usage_metric()
def retrigger_transient_view(request=None, transient_name=''):
    return retrigger_transient(request, transient_name)


def retrigger_transient(request=None, transient_name=''):
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
        return HttpResponseRedirect(reverse_lazy("results", kwargs={"transient_name": transient_name}))
    else:
        return result


@shared_task(
    name="Import transients from TNS",
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def import_transient_list(transient_names):
    '''This function assumes that the input transient_names are not in the database.'''
    uploaded_transient_names = []
    for transient_name in transient_names:
        logger.info(f'Triggering workflow for new transient "{transient_name}"...')
        try:
            transient_workflow.delay(transient_name)
            uploaded_transient_names.append(transient_name)
        except Exception as err:
            logger.error(f'''Error processing new transient: {err}''')
    return uploaded_transient_names


@shared_task(
    name="Get Final Progress",
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def final_progress(transient_name):
    transient = Transient.objects.get(name=transient_name)
    transient.progress, transient.processing_status = get_processing_status_and_progress(transient)
    logger.debug(f'''Final progress: {(transient.progress, transient.processing_status)}''')
    transient.save()
    # Clean up scratch directories
    for base_path in [settings.CUTOUT_ROOT, settings.SED_OUTPUT_ROOT]:
        try:
            rmtree(os.path.join(base_path, transient.name))
        except FileNotFoundError:
            pass


@shared_task(
    name="Dataset version control",
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def dataset_revision(transient_name):
    """Create a new dataset revision if needed."""

    def get_checksum(s3_instance, canonical_path):
        object_key = os.path.join(settings.S3_BASE_PATH, canonical_path.strip('/'))
        logger.debug(f'Getting checksum for "{object_key}"')
        file_obj = s3_instance.object_info(object_key)
        etag = file_obj.etag
        return etag

    transient = Transient.objects.get(name=transient_name)
    # Export tabular data
    dataset = export_dataset(transient.name)

    # Generate table of file object checksums
    canonical_paths = []
    # SED fitting results
    for aperture in dataset['apertures']:
        for sedfittingresult in aperture['sedfittingresults']:
            canonical_paths.extend([
                sedfittingresult['fields']['posterior'],
                sedfittingresult['fields']['chains_file'],
                sedfittingresult['fields']['percentiles_file'],
                sedfittingresult['fields']['model_file'],
            ])
    # Cutout images
    for cutout_path in [cutout['fields']['fits'] for cutout in dataset['cutouts'] if cutout['fields']['fits']]:
        canonical_paths.append(cutout_path)
    # Host spectra files
    for host_spectrum in dataset['host_spectra']:
        canonical_paths.append(host_spectrum['fields']['spectrum_file'])
    s3 = ObjectStore()
    checksums = {canonical_path: get_checksum(s3, canonical_path) for canonical_path in canonical_paths}

    # Create a candidate DR for comparison
    candidate_dr = DatasetRevision(transient=transient, data={'export': dataset, 'files': checksums})
    # Fetch latest DR
    previous_dr = DatasetRevision.objects.filter(transient=transient).order_by("-revision", "pk").first()
    if previous_dr is None:
        # Save candidate DR to the database as the first revision
        assert candidate_dr.revision == 0
        candidate_dr.save()
        logger.debug(f'New dataset revision created: {candidate_dr}')
        return
    # If there is an existing DR, compare and create a new DR if any differences are detected.
    try:
        # Must have the same number of files
        assert len(candidate_dr.data['files']) == len(previous_dr.data['files'])
        # If any changes to output files are detected, make a new revision
        for canonical_path, checksum in previous_dr.data['files'].items():
            assert candidate_dr.data['files'][canonical_path] == checksum
        # If any changes to the immutable tabular data fields are detected, make a new revision
        # Check app version in metadata
        # logger.debug(candidate_dr.data['export'])
        # logger.debug(previous_dr.data['export'])
        candidate_app_version = candidate_dr.data['export']['metadata']['app_version']
        previous_app_version = previous_dr.data['export']['metadata']['app_version']
        assert candidate_app_version == previous_app_version
        # Compare other top-level objects
        previous_dr_data = previous_dr.data['export']
        candidate_dr_data = candidate_dr.data['export']
        previous_dr_data.pop('metadata')
        candidate_dr_data.pop('metadata')
        previous_dr_data.pop('workflow_tasks')
        candidate_dr_data.pop('workflow_tasks')
        assert equal_dicts(previous_dr_data, candidate_dr_data)
        logger.info(f'Dataset "{transient.name}" unchanged. No dataset revision created.')
    except (AssertionError, IndexError) as err:
        logger.debug(err)
        # Increment the revision index
        candidate_dr.revision = previous_dr.revision + 1
        candidate_dr.save()
        logger.debug(f'New dataset revision created: {candidate_dr}')
