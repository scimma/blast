import math

import numpy as np
from astropy.io import fits
from celery import shared_task
from django.db.models import Q
from host.base_tasks import task_soft_time_limit
from host.base_tasks import task_time_limit

from .base_tasks import TransientTaskRunner
from .cutouts import download_and_save_cutouts
from .ghost import run_ghost
from .host_utils import check_global_contamination
from .host_utils import check_local_radius
from .host_utils import construct_aperture
from .host_utils import do_aperture_photometry
from .host_utils import get_dust_maps
from .host_utils import get_local_aperture_size
from .host_utils import query_ned
from .host_utils import query_sdss
from .host_utils import select_cutout_aperture
from .models import Aperture
from .models import AperturePhotometry
from .models import Cutout
from .models import SEDFittingResult
from .models import StarFormationHistoryResult
from .models import Transient
from .prospector import build_model
from .prospector import build_obs
from .prospector import fit_model
from .prospector import prospector_result_to_blast

"""This module contains all of the TransientTaskRunners in blast."""


class Ghost(TransientTaskRunner):
    """
    TaskRunner to run the GHOST matching algorithm.
    """

    def _prerequisites(self):
        """
        Only prerequisite is that the host match task is not processed.
        """
        return {
            "Host match": "not processed",
            "Cutout download": "processed",
            "Transient MWEBV": "processed",
        }

    @property
    def task_name(self):
        """
        Task status to be altered is host match.
        """
        return "Host match"

    def _failed_status_message(self):
        """
        Failed status is no GHOST match status.
        """
        return "no GHOST match"

    def _run_process(self, transient):
        """
        Run the GHOST matching algorithm.
        """
        host = run_ghost(transient)

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
            status_message = "no ghost match"

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
            "Transient MWEBV": "not processed",
            "Transient information": "processed",
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
        return {"Host match": "processed", "Host MWEBV": "not processed"}

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
        return {"Cutout download": "not processed"}

    @property
    def task_name(self):
        """
        Task status to be altered is host match.
        """
        return "Cutout download"

    def _failed_status_message(self):
        """
        Failed status is no GHOST match status.
        """
        return "failed"

    def _run_process(self, transient):
        """
        Download cutout images
        """
        message = download_and_save_cutouts(transient)

        return message


class GlobalApertureConstruction(TransientTaskRunner):
    """Task runner to construct apertures from the cutout download"""

    def _prerequisites(self):
        """
        Need both the Cutout and Host match to be processed
        """
        return {
            "Cutout download": "processed",
            "Host match": "processed",
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
            return "failed"
        if not transient.host.sky_coord:
            print(f"""No sky_coord associated with "{transient.name}" host.""")
            return "failed"
        cutouts = Cutout.objects.filter(transient=transient).filter(~Q(fits=""))
        choice = 0
        aperture = None
        while aperture is None and choice <= 8:
            aperture_cutout = select_cutout_aperture(cutouts, choice=choice)
            image = fits.open(aperture_cutout[0].fits.name)
            aperture = construct_aperture(image, transient.host.sky_coord)
            choice += 1
        if aperture is None:
            return "failed"

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
            image = fits.open(cutout.fits.name)

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
            except Exception:
                raise
        return "processed"


class GlobalAperturePhotometry(TransientTaskRunner):
    """Task Runner to perform local aperture photometry around host"""

    def _prerequisites(self):
        """
        Need both the Cutout and Host match to be processed
        """
        return {
            "Cutout download": "processed",
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
            image = fits.open(cutout.fits.name)

            # make new aperture
            # adjust semi-major/minor axes for size
            if f"{cutout.name}_global" != aperture.name:
                if not len(
                    Aperture.objects.filter(cutout__name=f"{cutout.name}_global")
                ):
                    semi_major_axis = (
                        aperture.semi_major_axis_arcsec
                        - aperture.cutout.filter.image_fwhm_arcsec  # / 2.354
                        + cutout.filter.image_fwhm_arcsec  # / 2.354
                    )
                    semi_minor_axis = (
                        aperture.semi_minor_axis_arcsec
                        - aperture.cutout.filter.image_fwhm_arcsec  # / 2.354
                        + cutout.filter.image_fwhm_arcsec  # / 2.354
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
            except Exception:
                raise

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
        Failed status is local photometry validation failed.
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
            return "phot valid failed"

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
        Prerequisites are that the validate local photometry task
        is not processed and the local photometry task is processed.
        """
        return {
            "Global aperture photometry": "processed",
            "Validate global photometry": "not processed",
        }

    @property
    def task_name(self):
        """
        Task status to be altered is validate local photometry.
        """
        return "Validate global photometry"

    def _failed_status_message(self):
        """
        Failed status is local photometry validation failed.
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
            return "global photometry validation failed"

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


class TransientInformation(TransientTaskRunner):
    """Task Runner to gather information about the Transient"""

    def _prerequisites(self):
        return {
            "Transient information": "not processed",
            "Cutout download": "processed",
        }

    @property
    def task_name(self):
        return "Transient information"

    def _failed_status_message(self):
        """
        Failed status if not aperture is found
        """
        return "failed"

    def _run_process(self, transient):
        """Code goes here"""

        # get_dust_maps(10)
        return "processed"


class HostInformation(TransientTaskRunner):
    """Task Runner to gather host information from NED"""

    def _prerequisites(self):
        """
        Need both the Cutout and Host match to be processed
        """
        return {"Host match": "processed", "Host information": "not processed"}

    @property
    def task_name(self):
        return "Host information"

    def _failed_status_message(self):
        """
        Failed status if not aperture is found
        """
        return "failed"

    def _run_process(self, transient):
        """Code goes here"""

        host = transient.host
        if host is None:
            return "no host"

        galaxy_ned_data = query_ned(host.sky_coord)
        # too many SDSS errors
        try:
            galaxy_sdss_data = query_sdss(host.sky_coord)
        except Exception:
            galaxy_sdss_data = None

        status_message = "processed"

        if (
            galaxy_sdss_data is not None
            and galaxy_sdss_data["redshift"] is not None
            and not math.isnan(galaxy_sdss_data["redshift"])
        ):
            host.redshift = galaxy_sdss_data["redshift"]
        elif galaxy_ned_data["redshift"] is not None and not math.isnan(
            galaxy_ned_data["redshift"]
        ):
            host.redshift = galaxy_ned_data["redshift"]
        elif host.photometric_redshift is not None:
            pass
        elif transient.redshift is not None:
            pass
        else:
            status_message = "no host redshift"

        host.save()

        # shouldn't be necessary but seeing weird behavior on prod
        transient.host = host
        transient.save()

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
            prosp_results,sfh_results = prospector_result_to_blast(
                transient,
                aperture[0],
                posterior,
                model_components,
                observations,
                sed_output_root="/tmp",
            )
        else:
            prosp_results,sfh_results = prospector_result_to_blast(
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
            "Host match": "processed",
            "Host information": "processed",
            "Local aperture photometry": "processed",
            "Validate local photometry": "processed",
            "Local host SED inference": "not processed",
            "Transient MWEBV": "processed",
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
            "Host match": "processed",
            "Host information": "processed",
            "Global aperture photometry": "processed",
            "Validate global photometry": "processed",
            "Global host SED inference": "not processed",
            "Host MWEBV": "processed",
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

# Transient workflow tasks


@shared_task(
    name="Transient Information",
    time_limit=task_time_limit,
    soft_time_limit=task_soft_time_limit,
)
def transient_information(transient_name):
    TransientInformation(transient_name).run_process()


@shared_task(
    name="Host Match", time_limit=task_time_limit, soft_time_limit=task_soft_time_limit
)
def host_match(transient_name):
    Ghost(transient_name).run_process()


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
    transient.progress = 100
    transient.save()
