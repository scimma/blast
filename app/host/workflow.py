from celery import chain
from celery import chord
from celery import group
from celery import shared_task
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
from host.transient_tasks import transient_information
from host.transient_tasks import validate_global_photometry
from host.transient_tasks import validate_local_photometry
from django.urls import reverse_lazy
from django.http import HttpResponseRedirect
from host.models import TaskRegister
from host.models import Status

from .base_tasks import initialise_all_tasks_status
from .models import Transient
from .transient_name_server import get_transients_from_tns_by_name


def reprocess_transient(request=None, slug=''):
    transient_name = slug
    assert transient_name
    tasks = TaskRegister.objects.filter(transient__name=transient_name)
    if tasks:
        for task in tasks:
            # TODO: This could be smarter. We don't *always* need to re-process every stage.
            task.status = Status.objects.get(message="not processed")
            task.save()
        result = transient_workflow.delay(transient_name)
        if request:
            return HttpResponseRedirect(reverse_lazy("results", kwargs={"slug": transient_name}))
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
            initialise_all_tasks_status(transient)
            transient.tasks_initialized = "True"
            transient.save()
    # Execute the workflow
    workflow = chain(
        image_download.si(transient_name),
        transient_information.si(transient_name),
        mwebv_transient.si(transient_name),
        host_match.si(transient_name),
        host_information.si(transient_name),
        group(
            chain(
                local_aperture_photometry.si(transient_name),
                validate_local_photometry.si(transient_name),
                local_host_sed_fitting.si(transient_name),
            ),
            chord(
                (
                    mwebv_host.si(transient_name),
                    chain(
                        global_aperture_construction.si(transient_name),
                        global_aperture_photometry.si(transient_name),
                        validate_global_photometry.si(transient_name),
                    ),
                ),
                global_host_sed_fitting.si(transient_name),
            ),
        ),
        final_progress.si(transient_name)
    )
    workflow.delay()

    return transient_name
