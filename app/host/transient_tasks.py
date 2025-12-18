import math
import os

import numpy as np
from astropy.io import fits
from celery import shared_task
from abc import abstractmethod
from django.db.models import Q
from host.base_tasks import task_soft_time_limit
from host.base_tasks import task_time_limit
from time import process_time
from billiard.exceptions import SoftTimeLimitExceeded
from django.utils import timezone

from host.cutouts import download_and_save_cutouts
from host.prost import run_prost
from host.host_utils import check_global_contamination
from host.host_utils import check_local_radius
from host.host_utils import construct_aperture
from host.host_utils import do_aperture_photometry
from host.host_utils import get_dust_maps
from host.host_utils import get_local_aperture_size
from host.host_utils import query_ned
from host.host_utils import query_sdss
from host.host_utils import select_cutout_aperture
from host.models import Aperture
from host.models import AperturePhotometry
from host.models import Cutout
from host.models import SEDFittingResult
from host.models import StarFormationHistoryResult
from host.models import Transient
from host.prospector import build_model
from host.prospector import build_obs
from host.prospector import fit_model
from host.prospector import prospector_result_to_blast
from host.object_store import ObjectStore
from host.base_tasks import TaskRunner
from host.crop_images import crop_images
from django.conf import settings

"""This module contains all of the TransientTaskRunners in blast."""

from host.models import TaskRegister
from host.models import Status
from host.models import Task

from host.log import get_logger
logger = get_logger(__name__)


def get_all_task_prerequisites(transient_name):
    return {
        ImageDownload(transient_name).task_name: ImageDownload(transient_name)._prerequisites(),
        MWEBV_Transient(transient_name).task_name: MWEBV_Transient(transient_name)._prerequisites(),
        HostMatch(transient_name).task_name: HostMatch(transient_name)._prerequisites(),
        HostInformation(transient_name).task_name: HostInformation(transient_name)._prerequisites(),
        LocalAperturePhotometry(transient_name).task_name: LocalAperturePhotometry(transient_name)._prerequisites(),
        ValidateLocalPhotometry(transient_name).task_name: ValidateLocalPhotometry(transient_name)._prerequisites(),
        LocalHostSEDFitting(transient_name).task_name: LocalHostSEDFitting(transient_name)._prerequisites(),
        MWEBV_Host(transient_name).task_name: MWEBV_Host(transient_name)._prerequisites(),
        GlobalApertureConstruction(transient_name).task_name: GlobalApertureConstruction(transient_name)._prerequisites(),  # noqa
        GlobalAperturePhotometry(transient_name).task_name: GlobalAperturePhotometry(transient_name)._prerequisites(),
        ValidateGlobalPhotometry(transient_name).task_name: ValidateGlobalPhotometry(transient_name)._prerequisites(),
        GlobalHostSEDFitting(transient_name).task_name: GlobalHostSEDFitting(transient_name)._prerequisites(),
        CropTransientImages(transient_name).task_name: CropTransientImages(transient_name)._prerequisites(),
        GenerateThumbnails(transient_name).task_name: GenerateThumbnails(transient_name)._prerequisites(),
        GenerateThumbnailsFinal(transient_name).task_name: GenerateThumbnailsFinal(transient_name)._prerequisites(),
    }


def get_processing_status_and_progress(transient):
    # Collect all workflow tasks.
    tasks = TaskRegister.objects.filter(transient__name__exact=transient.name)
    num_total_tasks = len(tasks)
    # If there are no tasks associated with the transient, the progress is zero.
    if num_total_tasks == 0:
        return 0, "processing"

    # Collect incomplete tasks, which include those currently processing and those not yet processed.
    incomplete_tasks = tasks.filter(status__message__in=['not processed', 'processing'])
    logger.debug(f'''"{transient.name}" incomplete tasks: {[task.task.name for task in incomplete_tasks]}''')

    # For each unprocessed task in the workflow, determine whether it could possibly run based on its prerequisites.
    unprocessed_tasks = incomplete_tasks.filter(status__message__exact='not processed')
    prerequisites = get_all_task_prerequisites(transient.name)
    blocked_tasks = []
    for unprocessed_task in unprocessed_tasks:
        prereq_tasks = prerequisites[unprocessed_task.task.name]
        prereqs_errored = []
        # Iterate over the task's prerequisites and check for any that have failed
        for task_name, status_message in prereq_tasks.items():
            prereq_task = Task.objects.get(name__exact=task_name)
            prereqs_errored.extend(tasks.filter(
                task__exact=prereq_task,
                status__type__exact='error',
            ))
            # If any of the unprocessed task's prerequisites failed, then it will never execute; it is "blocked".
            if prereqs_errored:
                logger.debug(f'''"{transient.name}" prereqs_errored: {[task.task.name for task in prereqs_errored]}''')
                blocked_tasks.append(unprocessed_task)
                break
    unblocked_tasks = [task for task in unprocessed_tasks if task not in blocked_tasks]
    logger.debug(f'''"{transient.name}" unblocked tasks: {[task.task.name for task in unblocked_tasks]}''')

    # The number of remaining tasks is the sum of the unblocked tasks and those currently processing
    processing_tasks = incomplete_tasks.filter(status__message__exact='processing')
    num_remaining_tasks = len(unblocked_tasks) + len(processing_tasks)
    progress_raw = 100 * (1 - num_remaining_tasks / num_total_tasks)
    progress = int(round(progress_raw, 0))
    logger.debug(f'''"{transient.name}" progress (raw): {progress_raw}''')

    # If the progress is not 100%, then the workflow is still processing
    if progress < 100:
        processing_status = 'processing'
    # If the progress is 100%, then the workflow is finished.
    elif progress == 100:
        # It is blocked if there are errors; otherwise it is complete.
        if tasks.filter(status__type='error'):
            processing_status = 'blocked'
        else:
            processing_status = 'completed'

    return progress, processing_status


class TransientTaskRunner(TaskRunner):
    """
    Abstract base class for a TransientTaskRunner.

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
        processing_status (models.Status): Status of the task while runner is
            running a task.
        failed_status (model.Status): Status of the task is if the runner fails.
        prerequisites (dict): Prerequisite tasks and statuses required for the
            runner to process.
    """

    def __init__(self, transient_name=None):
        """
        Initialized method which sets up the task runner.
        """

        self.prerequisites = self._prerequisites()
        assert transient_name
        self.transient_name = transient_name

    def find_register_items_meeting_prerequisites(self):
        """
        Finds the register items meeting the prerequisites.

        Returns:
            (QuerySet): Task register items meeting prerequisites.
        """
        task = Task.objects.get(name__exact=self.task_name)
        task_register = TaskRegister.objects.all()
        if self.transient_name:
            current_transients = Transient.objects.filter(
                name__exact=self.transient_name
            )
        else:
            current_transients = Transient.objects.all()

        for task_name, status_message in self.prerequisites.items():
            task_prereq = Task.objects.get(name__exact=task_name)
            status = Status.objects.get(message__exact=status_message)

            current_transients = current_transients & Transient.objects.filter(
                taskregister__task=task_prereq, taskregister__status=status
            )

        return task_register.filter(transient__in=list(current_transients), task=task)

    def _select_highest_priority(self, register):
        """
        Select highest priority task by finding the one with the oldest
        transient timestamp.

        Args:
            register (QuerySet): register of tasks to select from.
        Returns:
            register item (model.TaskRegister): highest priority register item.
        """
        return register.order_by("transient__public_timestamp")[0]

    def _update_status(self, task_status, updated_status):
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

    def select_register_item(self):
        """
        Selects register item to be processed by task runner.

        Returns:
            register item (models.TaskRegister): returns item if one exists,
                returns None otherwise.
        """
        register = self.find_register_items_meeting_prerequisites()
        return self._select_highest_priority(register) if register.exists() else None

    def run_process(self, task_register_item=None):
        """
        Runs task runner process.

        This logic of this function is a bit confusing due to the change in Blast's workflow execution method (see
        commit 23d34a1ca8e1d6086672b4b0a60fe9f74fb3bdfa 2024/05/06).
        Originally, this function was executed periodically by calling it from the subclasses corresponding
        to the various stages of the transient workflow. This execution method was independent of individual
        transients: The algorithm of the "select_register_item()" function was designed to identify all the
        transients whose workflows had reached the task associated with the calling subclass
        (e.g. "ValidateLocalPhotometry.run_process()") by analyzing the "prerequisites" for each registered task to
        determine if they had been met, and then select which transient's workflow task to run using
        a priority system based on the transient discovery time.
        The current workflow execution method is quite different: Now the DAG of tasks defining a workflow are
        executed by the Celery Canvas system. The workflow definition in this Canvas system is therefore redundant
        with the original Blast mechanism of "prerequisites". Because the "task_register_item" in this function is
        still selected using the same logic, however, these prerequisites must remain consistent with the Canvas
        workflow definition.
        """
        if task_register_item is None:
            task_register_item = self.select_register_item()

        if task_register_item is not None:
            logger.debug(f'''task_register_item: {task_register_item}''')
            self._update_status(task_register_item, Status.objects.get(message__exact="processing"))
            transient = task_register_item.transient

            start_time = process_time()
            status_message = ''
            try:
                status_message = self._run_process(transient)
            except SoftTimeLimitExceeded:
                status_message = "time limit exceeded"
                raise
            except Exception:
                status_message = self._failed_status_message()
                logger.error(f'''"{self.task_name}" error message: "{status_message}"''')
                raise
            finally:
                end_time = process_time()
                logger.debug(f'''"{self.task_name}" status message: "{status_message}"''')
                status = Status.objects.get(message__exact=status_message)
                self._update_status(task_register_item, status)
                processing_time = round(end_time - start_time, 2)
                task_register_item.last_processing_time_seconds = processing_time
                task_register_item.save()
                # The processing status should be calculated
                transient.progress, transient.processing_status = get_processing_status_and_progress(transient)
                transient.save()
            return transient.name

    @abstractmethod
    def _run_process(self, transient):
        """
        Run process function to be implemented by child classes.

        Args:
            transient (models.Transient): transient for the task runner to
                process
        Returns:
            runner status (models.Status): status of the task after the task
                runner has completed.
        """
        pass

    @abstractmethod
    def _prerequisites(self):
        """
        Task prerequisites to be implemented by child classes.

        Returns:
            prerequisites (dict): key is the name of the task, value is the task
                status.
        """
        pass

    @abstractmethod
    def _failed_status_message(self):
        """
        Message of the failed status.

        Returns:
            failed message (str): Name of the message of the failed status.
        """
        pass

    @property
    def task_type(self):
        return "transient"

    @property
    def task_function_name(self) -> str:
        """
        TaskRunner function name to be registered by celery.
        """
        return "host.transient_tasks." + self.task_name.replace(" ", "_").lower()


class HostMatch(TransientTaskRunner):
    """
    TaskRunner to run the host matching algorithm.
    """

    def _prerequisites(self):
        """
        Only prerequisite is that the host match task is not processed.
        """
        return {
            "Cutout download": "processed",
            "Transient MWEBV": "processed",
            "Host match": "not processed",
        }

    @property
    def task_name(self):
        """
        Task status to be altered is host match.
        """
        return "Host match"

    def _failed_status_message(self):
        """
        Emit status message for failure consistent with the available Status objects
        """
        return "no host match"

    def _run_process(self, transient):
        """
        Run the host matching algorithm.
        """
        host = run_prost(transient)

        if host is not None:
            host.save()
            transient.host = host
            transient.save()

            # having a weird error here
            # possible issues communicating with the database
            transient_check = Transient.objects.get(name=transient.name)
            if transient_check.host is None:
                # let's try twice just in case
                transient.host = host
                transient.save()

                transient_check = Transient.objects.get(name=transient.name)
                if transient_check.host is None:
                    raise RuntimeError("problem saving transient to the database!")

            status_message = "processed"
        else:
            status_message = "no host match"

        return status_message


class MWEBV_Transient(TransientTaskRunner):
    """
    TaskRunner to run get Milky Way E(B-V) values at the transient location.
    """

    def _prerequisites(self):
        """
        Only prerequisite is that the transient MWEBV task is not processed.
        """
        return {
            "Cutout download": "processed",
            "Transient MWEBV": "not processed",
        }

    @property
    def task_name(self):
        """
        Task status to be altered is transient MWEBV.
        """
        return "Transient MWEBV"

    def _failed_status_message(self):
        """
        Failed status - not sure why this would ever fail so keeping it vague.
        """
        return "failed"

    def _run_process(self, transient):
        """
        Run the E(B-V) script.
        """

        try:
            mwebv = get_dust_maps(transient.sky_coord)
        except Exception:
            mwebv = None

        if mwebv is not None:
            transient.milkyway_dust_reddening = mwebv
            transient.save()
            status_message = "processed"
        else:
            status_message = "no transient MWEBV"

        return status_message


class MWEBV_Host(TransientTaskRunner):
    """
    TaskRunner to run get Milky Way E(B-V) values at the host location.
    """

    def _prerequisites(self):
        """
        Only prerequisite is that the host MWEBV task is not processed.
        """
        return {
            "Cutout download": "processed",
            "Transient MWEBV": "processed",
            "Host match": "processed",
            "Host information": "processed",
            "Host MWEBV": "not processed"
        }

    @property
    def task_name(self):
        """
        Task status to be altered is host MWEBV.
        """
        return "Host MWEBV"

    def _failed_status_message(self):
        """
        Failed status - not sure why this would ever fail so keeping it vague.
        """
        return "failed"

    def _run_process(self, transient):
        """
        Run the E(B-V) script.
        """
        if transient.host is not None:
            try:
                mwebv = get_dust_maps(transient.host.sky_coord)
            except Exception:
                mwebv = None

            if mwebv is not None:
                transient.host.milkyway_dust_reddening = mwebv
                transient.host.save()
                status_message = "processed"
            else:
                status_message = "no host MWEBV"
        else:
            status_message = "no host MWEBV"

        return status_message


class ImageDownload(TransientTaskRunner):
    """Task runner to download cutout images"""

    def _prerequisites(self):
        """
        No prerequisites
        """
        return {
            "Cutout download": "not processed"
        }

    @property
    def task_name(self):
        """
        Task status to be altered is host match.
        """
        return "Cutout download"

    def _failed_status_message(self):
        """
        Emit status message for failure consistent with the available Status objects
        """
        return "failed"

    def _run_process(self, transient):
        """
        Download cutout images
        """

        # If the downloaded images have already been cropped
        task = Task.objects.get(name__exact="Crop transient images")
        task_register = TaskRegister.objects.get(transient=transient, task=task)
        if task_register.status.message == "processed":
            overwrite = "True"
        else:
            overwrite = "False"

        message = download_and_save_cutouts(
            transient,
            overwrite=overwrite
        )

        return message


class GlobalApertureConstruction(TransientTaskRunner):
    """Task runner to construct apertures from the cutout download"""

    def _prerequisites(self):
        """
        Need both the Cutout and Host match to be processed
        """
        return {
            "Cutout download": "processed",
            "Transient MWEBV": "processed",
            "Host match": "processed",
            "Host information": "processed",
            "Global aperture construction": "not processed",
        }

    @property
    def task_name(self):
        """
        Task status to be altered is host match.
        """
        return "Global aperture construction"

    def _failed_status_message(self):
        """
        Failed status if not aperture is found
        """
        return "failed"

    def _run_process(self, transient):
        """Code goes here"""

        if not transient.host:
            print(f"""No host associated with "{transient.name}".""")
            return self. _failed_status_message()
        if not transient.host.sky_coord:
            print(f"""No sky_coord associated with "{transient.name}" host.""")
            return self. _failed_status_message()
        cutouts = Cutout.objects.filter(transient=transient).filter(~Q(fits=""))
        choice = 0
        aperture = None
        while aperture is None and choice <= 8:
            aperture_cutout = select_cutout_aperture(cutouts, choice=choice)
            # Download FITS file local file cache
            fits_basepath = aperture_cutout[0].fits.name
            local_fits_path = f'''{fits_basepath}.GlobalApertureConstruction'''
            if not os.path.isfile(local_fits_path):
                s3 = ObjectStore()
                object_key = os.path.join(settings.S3_BASE_PATH, fits_basepath.strip('/'))
                s3.download_object(path=object_key, file_path=local_fits_path)
            assert os.path.isfile(local_fits_path)
            # Construct aperture
            image = fits.open(local_fits_path)
            try:
                aperture = construct_aperture(image, transient.host.sky_coord)
            finally:
                try:
                    # Delete FITS file from local file cache
                    os.remove(local_fits_path)
                except FileNotFoundError:
                    pass
            choice += 1
        if aperture is None:
            return self. _failed_status_message()

        query = {"name": f"{aperture_cutout[0].name}_global"}
        data = {
            "name": f"{aperture_cutout[0].name}_global",
            "cutout": aperture_cutout[0],
            "orientation_deg": (180 / np.pi) * aperture.theta.value,
            "ra_deg": aperture.positions.ra.degree,
            "dec_deg": aperture.positions.dec.degree,
            "semi_major_axis_arcsec": aperture.a.value,
            "semi_minor_axis_arcsec": aperture.b.value,
            "transient": transient,
            "type": "global",
        }

        self._overwrite_or_create_object(Aperture, query, data)
        return "processed"


class LocalAperturePhotometry(TransientTaskRunner):
    """Task Runner to perform local aperture photometry around host"""

    def _prerequisites(self):
        """
        Need both the Cutout and Host match to be processed
        """
        return {
            "Cutout download": "processed",
            "Transient MWEBV": "processed",
            "Host match": "processed",
            "Host information": "processed",
            "Local aperture photometry": "not processed",
        }

    @property
    def task_name(self):
        """
        Task status to be altered is Local Aperture photometry
        """
        return "Local aperture photometry"

    def _failed_status_message(self):
        """
        Failed status if not aperture is found
        """
        return "failed"

    def _run_process(self, transient):
        """Code goes here"""

        if transient.best_redshift is None or transient.best_redshift < 0:
            return "failed"

        aperture_size = get_local_aperture_size(transient.best_redshift)

        query = {"name__exact": f"{transient.name}_local"}
        data = {
            "name": f"{transient.name}_local",
            "orientation_deg": 0.0,
            "ra_deg": transient.sky_coord.ra.degree,
            "dec_deg": transient.sky_coord.dec.degree,
            "semi_major_axis_arcsec": aperture_size,
            "semi_minor_axis_arcsec": aperture_size,
            "transient": transient,
            "type": "local",
        }

        self._overwrite_or_create_object(Aperture, query, data)
        aperture = Aperture.objects.get(**query)
        print(aperture)
        cutouts = Cutout.objects.filter(transient=transient).filter(~Q(fits=""))

        for cutout in cutouts:
            fits_basepath = cutout.fits.name
            local_fits_path = f'''{fits_basepath}.LocalAperturePhotometry'''
            if not os.path.isfile(local_fits_path):
                # Download FITS file local file cache
                s3 = ObjectStore()
                object_key = os.path.join(settings.S3_BASE_PATH, fits_basepath.strip('/'))
                s3.download_object(path=object_key, file_path=local_fits_path)
            assert os.path.isfile(local_fits_path)
            image = fits.open(local_fits_path)
            try:
                photometry = do_aperture_photometry(
                    image, aperture.sky_aperture, cutout.filter
                )

                query = {
                    "aperture": aperture,
                    "transient": transient,
                    "filter": cutout.filter,
                }

                data = {
                    "aperture": aperture,
                    "transient": transient,
                    "filter": cutout.filter,
                    "flux": photometry["flux"],
                    "flux_error": photometry["flux_error"],
                }

                if photometry["flux"] is not None and photometry["flux"] > 0:
                    data["magnitude"] = photometry["magnitude"]
                    data["magnitude_error"] = photometry["magnitude_error"]

                self._overwrite_or_create_object(AperturePhotometry, query, data)
            finally:
                try:
                    # Delete FITS file from local file cache
                    os.remove(local_fits_path)
                except FileNotFoundError:
                    pass
        return "processed"


class GlobalAperturePhotometry(TransientTaskRunner):
    """Task Runner to perform local aperture photometry around host"""

    def _prerequisites(self):
        """
        Need both the Cutout and Host match to be processed
        """
        return {
            "Cutout download": "processed",
            "Transient MWEBV": "processed",
            "Host match": "processed",
            "Host information": "processed",
            "Global aperture construction": "processed",
            "Global aperture photometry": "not processed",
        }

    @property
    def task_name(self):
        """
        Task status to be altered is Local Aperture photometry
        """
        return "Global aperture photometry"

    def _failed_status_message(self):
        """
        Failed status if not aperture is found
        """
        return "failed"

    def _run_process(self, transient):
        """Code goes here"""

        cutouts = Cutout.objects.filter(transient=transient).filter(~Q(fits=""))
        choice = 0
        aperture = None
        for choice in range(9):
            cutout_for_aperture = select_cutout_aperture(cutouts, choice=choice)[0]
            aperture = Aperture.objects.filter(
                cutout__name=cutout_for_aperture.name, type="global"
            )
            if aperture.exists():
                aperture = aperture[0]
                break
        query = {"name": f"{cutout_for_aperture.name}_global"}
        for cutout in cutouts:
            fits_basepath = cutout.fits.name
            local_fits_path = f'''{fits_basepath}.GlobalAperturePhotometry'''
            if not os.path.isfile(local_fits_path):
                # Download FITS file local file cache
                s3 = ObjectStore()
                object_key = os.path.join(settings.S3_BASE_PATH, fits_basepath.strip('/'))
                s3.download_object(path=object_key, file_path=local_fits_path)
            assert os.path.isfile(local_fits_path)
            image = fits.open(local_fits_path)
            # make new aperture
            # adjust semi-major/minor axes for size
            if f"{cutout.name}_global" != aperture.name:

                if not len(
                    Aperture.objects.filter(cutout__name=f"{cutout.name}_global")
                ):
                    # quadrature differences in resolution
                    semi_major_axis = (
                        np.sqrt(
                            aperture.semi_major_axis_arcsec**2.
                            - aperture.cutout.filter.image_fwhm_arcsec**2.  # / 2.354  # noqa: W503
                            + cutout.filter.image_fwhm_arcsec**2.  # / 2.354  # noqa: W503
                        )
                    )
                    semi_minor_axis = (
                        np.sqrt(
                            aperture.semi_minor_axis_arcsec**2.
                            - aperture.cutout.filter.image_fwhm_arcsec**2.  # / 2.354  # noqa: W503
                            + cutout.filter.image_fwhm_arcsec**2.  # / 2.354  # noqa: W503
                        )
                    )

                    query = {"name": f"{cutout.name}_global"}
                    data = {
                        "name": f"{cutout.name}_global",
                        "cutout": cutout,
                        "orientation_deg": aperture.orientation_deg,
                        "ra_deg": aperture.ra_deg,
                        "dec_deg": aperture.dec_deg,
                        "semi_major_axis_arcsec": semi_major_axis,
                        "semi_minor_axis_arcsec": semi_minor_axis,
                        "transient": transient,
                        "type": "global",
                    }

                    self._overwrite_or_create_object(Aperture, query, data)
                    aperture = Aperture.objects.get(
                        transient=transient, name=f"{cutout.name}_global"
                    )

            try:
                photometry = do_aperture_photometry(
                    image, aperture.sky_aperture, cutout.filter
                )
                if photometry["flux"] is None:
                    continue

                query = {
                    "aperture": aperture,
                    "transient": transient,
                    "filter": cutout.filter,
                }

                data = {
                    "aperture": aperture,
                    "transient": transient,
                    "filter": cutout.filter,
                    "flux": photometry["flux"],
                    "flux_error": photometry["flux_error"],
                }
                if photometry["flux"] > 0:
                    data["magnitude"] = photometry["magnitude"]
                    data["magnitude_error"] = photometry["magnitude_error"]

                self._overwrite_or_create_object(AperturePhotometry, query, data)
            finally:
                try:
                    # Delete FITS file from local file cache
                    os.remove(local_fits_path)
                except FileNotFoundError:
                    pass

        return "processed"


class ValidateLocalPhotometry(TransientTaskRunner):
    """
    TaskRunner to validate the local photometry.
    We need to make sure image seeing is ~smaller than the aperture size
    """

    def _prerequisites(self):
        """
        Prerequisites are that the validate local photometry task is
        not processed and the local photometry task is processed.
        """
        return {
            "Cutout download": "processed",
            "Transient MWEBV": "processed",
            "Host match": "processed",
            "Host information": "processed",
            "Local aperture photometry": "processed",
            "Validate local photometry": "not processed",
        }

    @property
    def task_name(self):
        """
        Task status to be altered is validate local photometry.
        """
        return "Validate local photometry"

    def _failed_status_message(self):
        """
        Emit status message for failure consistent with the available Status objects
        """
        return "phot valid failed"

    def _run_process(self, transient):
        """
        Run the local photometry validation
        """

        local_aperture_photometry = AperturePhotometry.objects.filter(
            transient=transient, aperture__type="local"
        )
        redshift = transient.best_redshift

        # we can't measure the local aperture if we don't know the redshift
        if redshift is None:
            for local_aperture_phot in local_aperture_photometry:
                local_aperture_phot.is_validated = "false"
                local_aperture_phot.save()

        if not len(local_aperture_photometry):
            return self._failed_status_message()

        for local_aperture_phot in local_aperture_photometry:
            is_validated = check_local_radius(
                redshift,
                local_aperture_phot.filter.image_fwhm_arcsec,
            )
            local_aperture_phot.is_validated = is_validated
            local_aperture_phot.save()

        validated_local_aperture_photometry = AperturePhotometry.objects.filter(
            transient=transient, aperture__type="local", is_validated="true"
        )
        if len(validated_local_aperture_photometry):
            return "processed"
        else:
            return "no valid phot"


class ValidateGlobalPhotometry(TransientTaskRunner):
    """
    TaskRunner to validate the global photometry.
    We need to check for contaminating objects in the aperture
    """

    def _prerequisites(self):
        """
        Prerequisites are that the validate global photometry task
        is not processed and the global photometry task is processed.
        """
        return {
            "Cutout download": "processed",
            "Transient MWEBV": "processed",
            "Host match": "processed",
            "Host information": "processed",
            "Global aperture construction": "processed",
            "Global aperture photometry": "processed",
            "Validate global photometry": "not processed",
        }

    @property
    def task_name(self):
        """
        Task status to be altered is validate global photometry.
        """
        return "Validate global photometry"

    def _failed_status_message(self):
        """
        Emit status message for failure consistent with the available Status objects
        """
        return "phot valid failed"

    def _run_process(self, transient):
        """
        Run the global photometry validation
        """

        cutouts = Cutout.objects.filter(transient=transient).filter(~Q(fits=""))
        cutout_for_aperture = select_cutout_aperture(cutouts)[0]
        aperture_primary = Aperture.objects.get(
            cutout__name=cutout_for_aperture.name, type="global"
        )

        global_aperture_photometry = AperturePhotometry.objects.filter(
            transient=transient, aperture__type="global"
        )

        if not len(global_aperture_photometry):
            return self._failed_status_message()

        is_contam_list = []
        # issue_warning = True
        # no_contam_count = 0
        for global_aperture_phot in global_aperture_photometry:
            # check if there are contaminating objects in the
            # cutout image used for aperture construction at
            # the PSF-adjusted radius
            # AND
            # if there are contaminating objects detected in
            # the cutout image used for the photometry
            is_contam = check_global_contamination(
                global_aperture_phot, aperture_primary
            )

            # if all of our photometry is contaminated, the best move is just to
            # go ahead and compute things, then warn the user
            # otherwise, nearby galaxies are gonna be a huge pain
            is_contam_list += [is_contam]
        # issue warning and proceed if max 2 un-contaminated photometry points
        issue_warning = (
            True if len(np.where(~np.array(is_contam_list))[0]) <= 2 else False
        )

        for global_aperture_phot, is_contam in zip(
            global_aperture_photometry, is_contam_list
        ):
            if issue_warning:
                contam_message = "contamination warning" if is_contam else "true"
            else:
                contam_message = "false" if is_contam else "true"
            global_aperture_phot.is_validated = contam_message
            global_aperture_phot.save()

        return "processed"


class HostInformation(TransientTaskRunner):
    """Task Runner to gather host information from NED"""

    def _prerequisites(self):
        """
        Need both the Cutout and Host match to be processed
        """
        return {
            "Cutout download": "processed",
            "Transient MWEBV": "processed",
            "Host match": "processed",
            "Host information": "not processed"
        }

    @property
    def task_name(self):
        return "Host information"

    def _failed_status_message(self):
        """
        Failed status if not aperture is found
        """
        return "failed"

    def _run_process(self, transient):
        '''Obtain a redshift value and return the task status:
           "no host", "processed", or "no host redshift"
        '''

        if transient.host is None:
            return "no host"

        # First try obtaining redshift from SDSS.
        redshift = None
        try:
            galaxy_sdss_data = query_sdss(transient.host.sky_coord)
            redshift = galaxy_sdss_data["redshift"]
        except Exception as err:
            logger.warning(f''''Error querying SDSS: {err}''')
        # If SDSS query fails to return a valid value, query NED.
        if not redshift:
            try:
                galaxy_ned_data = query_ned(transient.host.sky_coord)
                assert not math.isnan(galaxy_ned_data["redshift"])
                redshift = galaxy_ned_data["redshift"]
            except Exception as err:
                logger.warning(f''''Error querying NED: {err}''')

        # If one of the queries yielded a redshift value, assign it.
        if redshift and not math.isnan(redshift):
            transient.host.redshift = redshift
            transient.save()
        if (
            transient.host.redshift
            or transient.host.photometric_redshift is not None
            or transient.redshift is not None
        ):
            status_message = "processed"
        else:
            status_message = "no host redshift"

        return status_message


class HostSEDFitting(TransientTaskRunner):
    """Task Runner to run host galaxy inference with prospector"""

    def _run_process(
        self, transient, aperture_type="global", mode="fast", sbipp=True, save=True
    ):
        """Run the SED-fitting task"""

        query = {
            "transient__name__exact": f"{transient.name}",
            "type__exact": aperture_type,
        }

        if transient.best_redshift is None or transient.best_redshift > 0.2:
            # training sample doesn't work here
            return "redshift too high"

        aperture = Aperture.objects.filter(**query)
        if len(aperture) == 0:
            raise RuntimeError(f"no apertures found for transient {transient.name}")

        observations = build_obs(transient, aperture_type)
        model_components = build_model(observations)

        if mode == "test" and not sbipp:
            # garbage results but the test runs
            print("running in test mode")
            fitting_settings = dict(
                nlive_init=1,
                nested_method="rwalk",
                nested_target_n_effective=1,
                nested_maxcall_init=1,
                nested_maxiter_init=1,
                nested_maxcall=1,
                nested_maxiter=1,
                verbose=True,
            )
        elif mode == "fast" and not sbipp:
            # 3000 - "reasonable but approximate posteriors"
            print("running in fast mode")
            fitting_settings = dict(
                nlive_init=400,
                nested_method="rwalk",
                nested_target_n_effective=3000,
            )
        else:
            # 10000 - "high-quality posteriors"
            fitting_settings = dict(
                nlive_init=400,
                nested_method="rwalk",
                nested_target_n_effective=10000,
            )

        print("starting model fit")
        posterior, errflag = fit_model(
            observations,
            model_components,
            fitting_settings,
            sbipp=sbipp,
            fit_type=aperture_type,
        )
        if errflag:
            return "not enough filters"

        if mode == "test":
            prosp_results, sfh_results = prospector_result_to_blast(
                transient,
                aperture[0],
                posterior,
                model_components,
                observations,
                sed_output_root="/tmp",
            )
        else:
            prosp_results, sfh_results = prospector_result_to_blast(
                transient,
                aperture[0],
                posterior,
                model_components,
                observations,
                sbipp=sbipp,
            )
        if save:
            pr = SEDFittingResult.objects.filter(
                transient=transient, aperture__type=aperture_type
            )
            if len(pr):
                pr.update(**prosp_results)
                pr = pr[0]
            else:
                pr = SEDFittingResult.objects.create(**prosp_results)
            for sfh_r in sfh_results:
                ps = pr.logsfh.filter(
                    logsfr_tmin=sfh_r['logsfr_tmin']
                )
                sfh_r['transient'] = transient
                sfh_r['aperture'] = aperture[0]

                if len(ps):
                    ps.update(**sfh_r)
                else:
                    ps = StarFormationHistoryResult.objects.create(**sfh_r)
                    pr.logsfh.add(ps)
        else:
            print("printing results")
            print(prosp_results)
        return "processed"


class LocalHostSEDFitting(HostSEDFitting):
    """Task Runner to run local host galaxy inference with prospector"""

    def _prerequisites(self):
        """
        Need both the Cutout and Host match to be processed
        """
        return {
            "Cutout download": "processed",
            "Transient MWEBV": "processed",
            "Host match": "processed",
            "Host information": "processed",
            "Local aperture photometry": "processed",
            "Validate local photometry": "processed",
            "Local host SED inference": "not processed",
        }

    @property
    def task_name(self):
        """
        Task status to be altered is Local Aperture photometry
        """
        return "Local host SED inference"

    def _failed_status_message(self):
        """
        Failed status if not aperture is found
        """
        return "failed"

    def _run_process(self, transient, mode="fast"):
        """Run the SED-fitting task"""

        status_message = super()._run_process(
            transient, aperture_type="local", mode=mode
        )

        return status_message


class GlobalHostSEDFitting(HostSEDFitting):
    """Task Runner to run global host galaxy inference"""

    def _prerequisites(self):
        """
        Need both the Cutout and Host match to be processed
        """
        return {
            "Cutout download": "processed",
            "Transient MWEBV": "processed",
            "Host match": "processed",
            "Host information": "processed",
            "Global aperture construction": "processed",
            "Global aperture photometry": "processed",
            "Validate global photometry": "processed",
            "Host MWEBV": "processed",
            "Global host SED inference": "not processed",
        }

    @property
    def task_name(self):
        """
        Task status to be altered is Local Aperture photometry
        """
        return "Global host SED inference"

    def _failed_status_message(self):
        """
        Failed status if not aperture is found
        """
        return "failed"

    def _run_process(self, transient, mode="fast", save=True):
        """Run the SED-fitting task"""

        status_message = super()._run_process(
            transient, aperture_type="global", mode=mode, save=save
        )

        return status_message


class GenerateThumbnails(TransientTaskRunner):
    """
    TaskRunner to generate a static compressed thumbnails of the interactive data widgets.
    """

    def _prerequisites(self):
        return {
            "Cutout download": "processed",
            "Generate thumbnails": "not processed",
        }

    @property
    def task_name(self):
        return "Generate thumbnails final"

    def _failed_status_message(self):
        return "failed"

    def _run_process(self, transient):
        status_message = "processed"
        # TODO : DO STUFF
        return status_message


class GenerateThumbnailsFinal(GenerateThumbnails):
    """
    Supports a second invocation of the thumbnail generation task
    """

    def _prerequisites(self):
        return {
            "Cutout download": "processed",
            "Transient MWEBV": "processed",
            "Host match": "processed",
            "Host information": "processed",
            "Global aperture construction": "processed",
            "Global aperture photometry": "processed",
            "Validate global photometry": "processed",
            "Local aperture photometry": "processed",
            "Validate local photometry": "processed",
            "Generate thumbnails": "processed",
            "Crop transient images": "processed",
            "Generate thumbnails final": "not processed",
        }

    @property
    def task_name(self):
        return "Generate thumbnails final"


class CropTransientImages(TransientTaskRunner):
    """
    TaskRunner to crop cutout images to save disk space.
    """

    def _prerequisites(self):
        return {
            "Cutout download": "processed",
            "Transient MWEBV": "processed",
            "Host match": "processed",
            "Host information": "processed",
            "Global aperture construction": "processed",
            "Global aperture photometry": "processed",
            "Validate global photometry": "processed",
            "Local aperture photometry": "processed",
            "Validate local photometry": "processed",
            "Crop transient images": "not processed",
        }

    @property
    def task_name(self):
        return "Crop transient images"

    def _failed_status_message(self):
        return "failed"

    def _run_process(self, transient):
        status_message = "processed"
        crop_images(transient)
        return status_message


# Transient workflow tasks

@shared_task(
    name="Host Match", time_limit=task_time_limit, soft_time_limit=task_soft_time_limit
)
def host_match(transient_name):
    HostMatch(transient_name).run_process()


@shared_task(
    name="Generate thumbnails",
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def generate_thumbnails(transient_name):
    GenerateThumbnails(transient_name).run_process()


@shared_task(
    name="Generate thumbnails final",
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def generate_thumbnails_final(transient_name):
    GenerateThumbnailsFinal(transient_name).run_process()


@shared_task(
    name="Crop transient images",
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def crop_transient_images(transient_name):
    CropTransientImages(transient_name).run_process()


@shared_task(
    name="Global Aperture Construction",
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def global_aperture_construction(transient_name):
    GlobalApertureConstruction(transient_name).run_process()


@shared_task(
    name="Global Aperture Photometry",
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def global_aperture_photometry(transient_name):
    GlobalAperturePhotometry(transient_name).run_process()


@shared_task(
    name="Global Host SED Fitting",
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def global_host_sed_fitting(transient_name):
    GlobalHostSEDFitting(transient_name).run_process()


@shared_task(
    name="Host Information",
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def host_information(transient_name):
    HostInformation(transient_name).run_process()


@shared_task(
    name="Image Download",
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def image_download(transient_name):
    ImageDownload(transient_name).run_process()


@shared_task(
    name="Local Aperture Photometry",
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def local_aperture_photometry(transient_name):
    LocalAperturePhotometry(transient_name).run_process()


@shared_task(
    name="Local Host SED Fitting",
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def local_host_sed_fitting(transient_name):
    LocalHostSEDFitting(transient_name).run_process()


@shared_task(
    name="MWEBV Host", time_limit=task_time_limit, soft_time_limit=task_soft_time_limit
)
def mwebv_host(transient_name):
    MWEBV_Host(transient_name).run_process()


@shared_task(
    name="MWEBV Transient",
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def mwebv_transient(transient_name):
    MWEBV_Transient(transient_name).run_process()


@shared_task(
    name="Validate Global Photometry",
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def validate_global_photometry(transient_name):
    ValidateGlobalPhotometry(transient_name).run_process()


@shared_task(
    name="Validate Local Photometry",
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def validate_local_photometry(transient_name):
    ValidateLocalPhotometry(transient_name).run_process()


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
