from celery import chain
from celery import chord
from celery import group
from celery import shared_task
from host.transient_tasks import crop_transient_images
from host.transient_tasks import generate_thumbnail
from host.transient_tasks import generate_thumbnail_final
from host.transient_tasks import global_aperture_construction
from host.transient_tasks import global_aperture_photometry
from host.transient_tasks import global_host_sed_fitting
from host.transient_tasks import host_information
from host.transient_tasks import host_match
from host.transient_tasks import image_download
from host.transient_tasks import local_aperture_photometry
from host.transient_tasks import local_host_sed_fitting
from host.transient_tasks import mwebv_host
from host.transient_tasks import mwebv_transient
from host.transient_tasks import final_progress
from host.base_tasks import task_soft_time_limit
from host.base_tasks import task_time_limit
from host.transient_tasks import validate_global_photometry
from host.transient_tasks import validate_local_photometry
from host.transient_tasks import generate_thumbnail_sed_local
from host.transient_tasks import generate_thumbnail_sed_global
from django.urls import reverse_lazy
from django.http import HttpResponseRedirect

from .base_tasks import initialize_all_tasks_status
from .models import Transient
from .transient_name_server import get_transients_from_tns_by_name
from django.contrib.auth.decorators import login_required, permission_required
from host.decorators import log_usage_metric
from .host_utils import wait_for_free_space

from host.log import get_logger
logger = get_logger(__name__)


@login_required
@permission_required("host.reprocess_transient", raise_exception=True)
@log_usage_metric()
def reprocess_transient_view(request=None, transient_name=''):
    return reprocess_transient(request, transient_name)


@shared_task(
    name="Workflow init",
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def workflow_init(transient_name=None):
    wait_for_free_space()


def reprocess_transient(request=None, transient_name=''):
    assert transient_name
    try:
        transient = Transient.objects.get(name__exact=transient_name)
        # TODO: This could be smarter. We don't *always* need to re-process every stage.
        initialize_all_tasks_status(transient)
        result = transient_workflow.delay(transient_name)
    except Transient.DoesNotExist:
        result = None
    if request:
        return HttpResponseRedirect(reverse_lazy("results", kwargs={"transient_name": transient_name}))
    else:
        return result


@shared_task(
    name="Transient Workflow",
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def transient_workflow(transient_name=None):
    assert transient_name
    try:
        Transient.objects.get(name__exact=transient_name)
        print(f'Transient already exists: "{transient_name}"...')
    except Transient.DoesNotExist:
        print(f'Downloading transient info from TNS: "{transient_name}"...')
        blast_transients = get_transients_from_tns_by_name([transient_name])
        for transient in blast_transients:
            # TO DO: User object is not JSON-serializable, and this task is also launched
            #        by a periodic system task, so we could consider replacing the
            #        added_by value with a simple string of the username.
            # transient.added_by = request.User
            transient.save()
            print(f'New transient added from TNS: "{transient_name}"...')
    # Initialize the tasks
    uninitialized_transients = Transient.objects.filter(
        tasks_initialized__exact="False",
        name__exact=transient_name,
    )
    for transient in uninitialized_transients:
        if transient.name == transient_name:
            initialize_all_tasks_status(transient)
            transient.tasks_initialized = "True"
            transient.save()
    # Execute the workflow
    workflow = chain(
        workflow_init.si(transient_name),
        image_download.si(transient_name),
        group(
            generate_thumbnail.si(transient_name),
            chain(
                mwebv_transient.si(transient_name),
                host_match.si(transient_name),
                host_information.si(transient_name),
                group(
                    mwebv_host.si(transient_name),
                    chain(
                        global_aperture_construction.si(transient_name),
                        global_aperture_photometry.si(transient_name),
                        validate_global_photometry.si(transient_name),
                    ),
                    chain(
                        local_aperture_photometry.si(transient_name),
                        validate_local_photometry.si(transient_name),
                    ),
                ),
                crop_transient_images.si(transient_name),
            ),
        ),
        group(
            generate_thumbnail_final.si(transient_name),
            chain(
                global_host_sed_fitting.si(transient_name),
                generate_thumbnail_sed_global.si(transient_name),
            ),
            chain(
                local_host_sed_fitting.si(transient_name),
                generate_thumbnail_sed_local.si(transient_name),
            ),
        ),
        final_progress.si(transient_name)
    )
    workflow.delay()

    return transient_name
