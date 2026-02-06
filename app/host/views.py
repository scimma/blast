import django_filters
from django.contrib.auth.decorators import login_required, permission_required
from django.contrib.auth.mixins import UserPassesTestMixin
from django.db.models import Q
from django.http import HttpResponse, JsonResponse
from django.http import HttpResponseRedirect
from django.http import StreamingHttpResponse
from django.shortcuts import get_object_or_404
from django.shortcuts import render
from django.urls import re_path
from django.urls import reverse_lazy
from django.core.exceptions import ValidationError
from django_tables2 import RequestConfig
from host.forms import ImageGetForm
from host.forms import TransientUploadForm
from host.host_utils import select_aperture
from host.host_utils import select_best_cutout
from host.models import Acknowledgement
from host.models import Aperture
from host.models import AperturePhotometry
from host.models import Cutout
from host.models import Filter
from host.models import SEDFittingResult
from host.models import TaskRegister
from host.models import Task
from host.models import Status
from host.models import TaskRegisterSnapshot
from host.models import Transient
from host.plotting_utils import plot_bar_chart
from host.plotting_utils import plot_cutout_image
from host.plotting_utils import render_sed_plot
from host.plotting_utils import plot_sed
from host.plotting_utils import plot_timeseries
from host.tables import TransientTable
from host.tasks import import_transient_list
from host.object_store import ObjectStore
from silk.profiling.profiler import silk_profile
from django.template.loader import render_to_string
import os
from django.conf import settings
from celery import shared_task
from host.decorators import log_usage_metric
import csv
import io
import base64
from host.host_utils import ARCSEC_DEC_IN_DEG
from host.host_utils import ARCSEC_RA_IN_DEG
from host.log import get_logger
logger = get_logger(__name__)


def filter_transient_categories(qs, value, task_register=None):
    if task_register is None:
        task_register = TaskRegister.objects.all()
    if value == "Transients with Basic Information":
        qs = qs.filter(
            pk__in=task_register.filter(
                task__name="Transient MWEBV", status__message="processed"
            ).values("transient")
        )
    elif value == "Transients with Matched Hosts":
        qs = qs.filter(
            pk__in=task_register.filter(
                task__name="Host match", status__message="processed"
            ).values("transient")
        )
    elif value == "Transients with Photometry":
        qs = qs.filter(
            Q(
                pk__in=task_register.filter(
                    task__name="Local aperture photometry",
                    status__message="processed",
                ).values("transient")
            ) | Q(
                pk__in=task_register.filter(
                    task__name="Global aperture photometry",
                    status__message="processed",
                ).values("transient")
            )
        )
    elif value == "Transients with SED Fitting":
        qs = qs.filter(
            Q(
                pk__in=task_register.filter(
                    task__name="Local host SED inference",
                    status__message="processed",
                ).values("transient")
            ) | Q(
                pk__in=task_register.filter(
                    task__name="Global host SED inference",
                    status__message="processed",
                ).values("transient")
            )
        )
    elif value == "Finished Transients":
        qs = qs.filter(
            ~Q(
                pk__in=task_register.filter(~Q(status__message="processed")).values(
                    "transient"
                )
            )
        )

    return qs


class TransientFilter(django_filters.FilterSet):
    hostmatch = django_filters.ChoiceFilter(
        choices=[
            ("All Transients", "All Transients"),
            ("Transients with Matched Hosts", "Transients with Matched Hosts"),
            ("Transients with Photometry", "Transients with Photometry"),
            ("Transients with SED Fitting", "Transients with SED Fitting"),
            ("Finished Transients", "Finished Transients"),
        ],
        method="filter_transients",
        label="Search",
        empty_label=None,
        null_label=None,
    )
    ex = django_filters.CharFilter(
        field_name="name", lookup_expr="contains", label="Name"
    )

    class Meta:
        model = Transient
        fields = ["hostmatch", "ex"]

    def filter_transients(self, qs, name, value):
        qs = filter_transient_categories(qs, value)

        return qs


@silk_profile(name="List transients")
def transient_list(request):
    transients = Transient.objects.order_by("-public_timestamp")
    transientfilter = TransientFilter(request.GET, queryset=transients)

    table = TransientTable(transientfilter.qs)
    RequestConfig(request, paginate={"per_page": 50}).configure(table)

    context = {"transients": transients, "table": table, "filter": transientfilter}
    return render(request, "transient_list.html", context)


@login_required
@permission_required("host.upload_transient", raise_exception=True)
@log_usage_metric()
def add_transient(request):
    def identify_existing_transients(transients=[]):
        '''Input "transients" is a list of dicts of the form:
            [{
                'name': str (required),
                'ra_deg': float (optional),
                'dec_deg': float (optional),
            }]
        '''
        errors = []
        transient_names = [tr['name'] for tr in transients]
        existing_transients = list(Transient.objects.filter(name__in=transient_names))
        # Filter the input transients list for members with names not found in the existing_transients object list
        nonexisting_transients = [new_tr for new_tr in transients
                                  if new_tr['name'] not in [tr.name for tr in existing_transients]]
        # Discard any transients whose coordinates are too close to an existing transient
        new_transients = []
        for transient in nonexisting_transients:
            if ('ra_deg' in transient
                    and 'dec_deg' in transient
                    and isinstance(transient['ra_deg'], float)
                    and isinstance(transient['dec_deg'], float)):
                proximate_transients = list(Transient.objects.filter(
                    Q(ra_deg__gte=transient['ra_deg'] - ARCSEC_RA_IN_DEG)
                    & Q(ra_deg__lte=transient['ra_deg'] + ARCSEC_RA_IN_DEG)
                    & Q(dec_deg__gte=transient['dec_deg'] - ARCSEC_DEC_IN_DEG)
                    & Q(dec_deg__lte=transient['dec_deg'] + ARCSEC_DEC_IN_DEG)))
            else:
                proximate_transients = []
            if proximate_transients:
                # If there are transients too close, discard the input transient
                prox_trans_names = ', '.join([tr.name for tr in proximate_transients])
                err_msg = (f'''Transient "{transient['name']}" is within 1 arcsec of existing transient(s) '''
                           f'''{prox_trans_names}. Discarding.''')
                logger.info(err_msg)
                errors.append(err_msg)
                # Append proximate transients to the list of existing transients
                for proximate_transient in proximate_transients:
                    if proximate_transient.name not in [tr.name for tr in existing_transients]:
                        existing_transients.append(proximate_transient)
            else:
                # if there are no nearby transients, consider the input transient new
                new_transients.append(transient)
        existing_transient_names = [tr.name for tr in existing_transients]
        new_transient_names = [tr['name'] for tr in new_transients]
        logger.info(f'''Existing transients detected: {','.join(existing_transient_names)}''')
        return existing_transient_names, new_transient_names, errors

    errors = []
    defined_transient_names = []
    imported_transient_names = []
    existing_transient_names = []

    # add transients -- either from TNS or from RA/Dec/redshift
    if request.method == "POST":
        form = TransientUploadForm(request.POST)

        if form.is_valid():
            info = form.cleaned_data["tns_names"]
            if info:
                transient_names = [transient_name.strip() for transient_name in info.splitlines()]
                existing_transient_names, imported_transient_names, identify_errors = identify_existing_transients([{
                    'name': name,
                    'ra_deg': None,
                    'dec_deg': None,
                } for name in transient_names])
                errors.extend(identify_errors)
                # Trigger import and processing of new transients
                import_transient_list.delay(imported_transient_names)

            info = form.cleaned_data["full_info"]
            if info:
                trans_info_set = []
                reader = csv.DictReader(io.StringIO(info), fieldnames=[
                    'name',
                    'ra',
                    'dec',
                    'redshift',
                    'specclass',
                    'display_name',
                ])
                for transient in reader:
                    try:
                        if transient['display_name'].lower().strip() == "none":
                            display_name = None
                        else:
                            display_name = transient['display_name'].strip()
                        if transient['specclass'].lower().strip() == "none":
                            spectroscopic_class = None
                        else:
                            spectroscopic_class = transient['specclass'].strip()
                        if transient['redshift'].lower().strip() == "none":
                            redshift = None
                        else:
                            redshift = float(transient['redshift'].strip())
                        trans_info = {
                            "name": transient['name'].strip(),
                            "ra_deg": float(transient['ra'].strip()),
                            "dec_deg": float(transient['dec'].strip()),
                            "redshift": redshift,
                            "spectroscopic_class": spectroscopic_class,
                            "tns_id": 0,
                            "tns_prefix": "",
                            "added_by": request.user,
                            "display_name": display_name,
                        }
                    except Exception as err:
                        err_msg = f'''Error parsing line "{transient}": {err}'''
                        logger.error(err_msg)
                        errors.append(err_msg)
                        continue
                    trans_info_set.append(trans_info)
                transient_names = [trans_info['name'] for trans_info in trans_info_set]
                existing_transient_names, new_transient_names, identify_errors = identify_existing_transients([{
                    'name': trans_info['name'],
                    'ra_deg': trans_info['ra_deg'],
                    'dec_deg': trans_info['dec_deg'],
                } for trans_info in trans_info_set])
                errors.extend(identify_errors)
                for transient_name in new_transient_names:
                    trans_info = [trans_info for trans_info in trans_info_set
                                  if trans_info['name'] == transient_name][0]
                    trans_name = trans_info["name"]
                    try:
                        # This appears redundant to the lower-lever validation of new Transient objects,
                        # but it allows us to provide the user specific error information.
                        Transient.validate_name(trans_name)
                    except ValidationError as err:
                        logger.error(err.message)
                        errors.append(err.message)
                        continue
                    try:
                        Transient.objects.create(**trans_info)
                        defined_transient_names += [trans_name]
                    except Exception as err:
                        err_msg = f'Error creating transient: {err}'
                        logger.error(err_msg)
                        errors.append(err_msg)
                # Trigger processing of new transients
                import_transient_list.delay(defined_transient_names)

    else:
        form = TransientUploadForm()

    context = {
        "form": form,
        "errors": errors,
        "defined_transient_names": defined_transient_names,
        "imported_transient_names": imported_transient_names,
        "existing_transient_names": existing_transient_names,
    }
    return render(request, "add_transient.html", context)


def analytics(request):
    analytics_results = {}

    for aggregate in ["total", "not completed", "completed", "waiting"]:
        transients = TaskRegisterSnapshot.objects.filter(
            aggregate_type__exact=aggregate
        )
        transients_ordered = transients.order_by("-time")

        if transients_ordered.exists():
            transients_current = transients_ordered[0]
        else:
            transients_current = None

        analytics_results[f"{aggregate}_transients_current".replace(" ", "_")] = (
            transients_current
        )
        bokeh_processing_context = plot_timeseries()

    return render(
        request, "analytics.html", {**analytics_results, **bokeh_processing_context}
    )


@log_usage_metric()
@silk_profile(name="Transient result for some transient")
def results(request, transient_name):
    param_var_ptype = (
        [
            "{\\rm log}_{10}(M_{\\ast}/M_{\odot})\,",  # noqa
            "{\\rm log}_{10}({\\rm SFR})",  # noqa
            "{\\rm log}_{10}({\\rm sSFR})",  # noqa
            "{\\rm stellar\ age}",  # noqa
            "{\\rm log}_{10}(Z_{\\ast}/Z_{\odot})",  # noqa
            "{\\rm log}_{10}(Z_{gas}/Z_{\odot})\,",  # noqa
            "\\tau_2",
            "\delta",  # noqa
            "\\tau_1/\\tau_2",
            "Q_{PAH}",
            "U_{min}",
            "{\\rm log}_{10}(\gamma_e)\,",  # noqa
            "{\\rm log}_{10}(f_{AGN})\,",  # noqa
            "{\\rm log}_{10}(\\tau_{AGN})\,"  # noqa
        ],
        [
            "log_mass",
            "log_sfr",
            "log_ssfr",
            "log_age",
            "logzsol",
            "gas_logz",
            "dust2",
            "dust_index",
            "dust1_fraction",
            "duste_qpah",
            "duste_umin",
            "log_duste_gamma",
            "log_fagn",
            "log_agn_tau"
        ],
        [
            "normal",
            "normal",
            "normal",
            "normal",
            "Metallicity",
            "Metallicity",
            "Dust",
            "Dust",
            "Dust",
            "Dust",
            "Dust",
            "Dust",
            "AGN",
            "AGN"]
    )

    def user_warning(transient):
        # check for user warnings
        is_warning = False
        for u in transient.taskregister_set.all().values_list("user_warning", flat=True):
            is_warning |= u
        return {'warning': is_warning}

    def compile_workflow_status(transient):
        class workflow_diagram():
            def __init__(self, name='', message='', badge='', fill_color=''):
                self.name = name
                self.message = message
                self.badge = badge
                self.fill_color = fill_color
                self.fill_colors = {
                    'success': '#d5e8d4',
                    'error': '#f8cecc',
                    'warning': '#fff2cc',
                    'blank': '#aeb6bd',
                }

        # Compile a filtered and sorted TaskRegister object list for display.
        # Omit some utility tasks like thumbnail generation and cutout crop.
        transient_taskregister_set = []
        unsorted_taskregs = transient.taskregister_set.all()
        taskreg_order = [
            'Cutout download',
            'Transient MWEBV',
            'Host match',
            'Host information',
            'Host MWEBV',
            'Global aperture construction',
            'Global aperture photometry',
            'Validate global photometry',
            'Global host SED inference',
            'Local aperture photometry',
            'Validate local photometry',
            'Local host SED inference',
        ]
        for task_name in taskreg_order:
            try:
                transient_taskregister_set.append(
                    [taskreg for taskreg in unsorted_taskregs if taskreg.task.name == task_name][0])
            except IndexError:
                missing_tr = TaskRegister()
                missing_tr.task = Task(name=task_name)
                missing_tr.status = Status.objects.get(message__exact="not processed")
                transient_taskregister_set.append(missing_tr)
        workflow_diagrams = []
        for item in transient_taskregister_set:
            # Configure workflow diagram
            diagram_settings = workflow_diagram(
                name=item.task.name,
                message=item.status.message,
                badge=item.status.badge,
                fill_color=workflow_diagram().fill_colors[item.status.type],
            )
            workflow_diagrams.append(diagram_settings)

        # Determine CSS class for workflow processing status
        if transient.processing_status == "blocked":
            processing_status_badge_class = "badge bg-danger"
        elif transient.processing_status == "processing":
            processing_status_badge_class = "badge bg-warning"
        elif transient.processing_status == "completed":
            processing_status_badge_class = "badge bg-success"
        else:
            processing_status_badge_class = "badge bg-secondary"
        return {
            "transient": transient,
            "transient_taskregister_set": transient_taskregister_set,
            "workflow_diagrams": workflow_diagrams,
            "processing_status_badge_class": processing_status_badge_class,
        }

    def compile_photometry_results(transient, scope):
        '''Compile aperture photometry results'''
        assert scope in ["local", "global"]
        if scope == 'local':
            aperture_photometry = AperturePhotometry.objects.filter(
                transient=transient,
                aperture__type__exact=scope,
                flux__isnull=False,
                is_validated="true",
            ).order_by(
                'filter__wavelength_eff_angstrom'
            ).prefetch_related()
            return {
                "local_aperture_photometry": aperture_photometry,
            }
        elif scope == 'global':
            aperture_photometry = AperturePhotometry.objects.filter(
                transient=transient,
                aperture__type__exact=scope,
                flux__isnull=False,
            ).filter(
                Q(is_validated="true") | Q(is_validated="contamination warning")
            ).order_by(
                'filter__wavelength_eff_angstrom'
            ).prefetch_related()
            contam_warning = len(aperture_photometry.filter(is_validated="contamination warning")) > 0
            return {
                "global_aperture_photometry": aperture_photometry,
                "contam_warning": contam_warning,
            }

    def compile_sed_results(transient, category, scope):
        '''Compile SED results for data tables'''
        assert scope in ["local", "global"]
        assert category in ["base", "sfh"]

        sed_obj = SEDFittingResult.objects.filter(transient=transient, aperture__type__exact=scope)
        results = ()
        if category == 'base':
            # Compile spectral energy distribution results
            for param, var, ptype in zip(*param_var_ptype):
                if sed_obj.exists():
                    results += (
                        (
                            param,
                            sed_obj[0].__dict__[f"{var}_16"],
                            sed_obj[0].__dict__[f"{var}_50"],
                            sed_obj[0].__dict__[f"{var}_84"],
                            ptype,
                        ),
                    )
        elif category == 'sfh':
            # Compile star formation history results
            if sed_obj.exists():
                for sh in sed_obj[0].logsfh.all():
                    results += (
                        (
                            sh.logsfr_16,
                            sh.logsfr_50,
                            sh.logsfr_84,
                            sh.logsfr_tmin,
                            sh.logsfr_tmax
                        ),
                    )
        return results

    # Acquire the transient object or return 404 not found
    try:
        transient = Transient.objects.get(name__exact=transient_name)
    except Transient.DoesNotExist:
        return render(request, "transient_404.html", status=404)

    # Collect all cutouts with FITS files
    all_cutouts = Cutout.objects.filter(transient__name__exact=transient.name).filter(~Q(fits=""))
    filters = [cutout.filter.name for cutout in all_cutouts]
    filter_status = {
        filter_.name: ("yes" if filter_.name in filters else "no")
        for filter_ in Filter.objects.all()
    }

    # Compile aperture details
    global_aperture = select_aperture(transient)
    local_aperture = Aperture.objects.filter(type__exact="local", transient=transient)

    image_data = b''
    # Generate filter selection form and choose cutout to display
    cutout = select_best_cutout(transient.name)
    if request.method == "GET":
        filter_select_form = ImageGetForm(filter_choices=filters)
        # Choose the cutout from the available filters using the priority define in select_cutout_aperture()
        if cutout:
            # Download thumbnail image from object store
            # TODO: Replace this with something like Django's template fragment caching
            #       or perhaps serve the thumbnails directly from the bucket using an S3-to-HTTP proxy.
            s3 = ObjectStore()
            thumbnail_filepath = cutout.fits.name.replace(".fits", ".jpg")
            thumbnail_object_key = os.path.join(settings.S3_BASE_PATH, thumbnail_filepath.strip('/'))
            try:
                image_data = s3.get_object(path=thumbnail_object_key)
            except Exception as err:
                logger.info(f'''Error downloading thumbnail object: "{thumbnail_object_key}": {err}''')
                image_data = b''
            image_data_encoded = base64.b64encode(image_data).decode()
            bokeh_cutout_context = {}

    if request.method == "POST":
        # Selecting a new filter from the dropdown menu of the interactive image plot
        # triggers an HTTP POST request instead of a GET.
        # TODO: Replace this with an AJAX call to replace the image plot without a full page reload.
        filter_select_form = ImageGetForm(request.POST, filter_choices=filters)
        if filter_select_form.is_valid():
            filter = filter_select_form.cleaned_data["filters"]
            cutout = all_cutouts.filter(filter__name__exact=filter)[0]

    # If the thumbnail is not available, display the Bokeh cutout plot
    if request.method == "POST" or image_data == b'':
        try:
            bokeh_cutout_context = plot_cutout_image(
                cutout=cutout,
                transient=transient,
                global_aperture=global_aperture.prefetch_related(),
                local_aperture=local_aperture.prefetch_related(),
            )
        except Exception as err:
            logger.error(f'''Error rendering cutout plot: {err}''')
        image_data = b''
        image_data_encoded = base64.b64encode(image_data).decode()

    # Download the SED thumbnails if they exist
    s3 = ObjectStore()
    interactive_sed_plot = {}
    image_data_encoded_sed = {}
    for scope in ['local', 'global']:
        image_data = b''
        interactive_sed_plot[scope] = {}
        image_data_encoded_sed[scope] = ''
        sed_obj = SEDFittingResult.objects.filter(transient=transient, aperture__type__exact=scope)
        if sed_obj.exists():
            sed_filepath = sed_obj[0].posterior.name
            thumbnail_filepath = sed_filepath.replace(".h5", ".jpg")
            thumbnail_object_key = os.path.join(settings.S3_BASE_PATH, thumbnail_filepath.strip('/'))
            try:
                logger.debug(f'''Downloading thumbnail object: "{thumbnail_object_key}"...''')
                image_data = s3.get_object(path=thumbnail_object_key)
            except Exception as err:
                logger.info(f'''Error downloading thumbnail object: "{thumbnail_object_key}": {err}''')
                image_data = b''
            image_data_encoded_sed[scope] = base64.b64encode(image_data).decode()
            interactive_sed_plot[scope] = {}
        # If there are no SED plot thumbnails, render the interactive plots
        if image_data == b'':
            try:
                interactive_sed_plot[scope] = render_sed_plot(transient, scope)
            except Exception as err:
                logger.error(f'''Error rendering SED plot: {err}''')
                try:
                    interactive_sed_plot[scope] = plot_sed(
                        transient=transient,
                        type=scope,
                        sed_results_file=None,
                        sed_modeldata_file=None,
                    )
                except Exception as err2:
                    logger.error(f'''Error rendering SED plot: {err2}''')
                    interactive_sed_plot[scope] = {}

    # Construct the Django render() function context
    context = {
        **{
            "transient": transient,
            "filter_select_form": filter_select_form,
            "filter_status": filter_status,
            "local_aperture": local_aperture[0] if local_aperture.exists() else None,
            "global_aperture": global_aperture[0] if global_aperture.exists() else None,
            "local_sed_results": compile_sed_results(transient, 'base', 'local'),
            "global_sed_results": compile_sed_results(transient, 'base', 'global'),
            "local_sfh_results": compile_sed_results(transient, 'sfh', 'local'),
            "global_sfh_results": compile_sed_results(transient, 'sfh', 'global'),
            "is_auth": request.user.is_authenticated,
            "image_data_encoded": image_data_encoded,
            "image_data_encoded_sed_local": image_data_encoded_sed['local'],
            "image_data_encoded_sed_global": image_data_encoded_sed['global'],
        },
        **bokeh_cutout_context,
        **user_warning(transient),
        **compile_photometry_results(transient, 'local'),
        **compile_photometry_results(transient, 'global'),
        **compile_workflow_status(transient),
        **interactive_sed_plot['local'],
        **interactive_sed_plot['global'],
    }
    # Return rendered HTML content
    return render(request, "results.html", context)


def stream_sed_output_file(file_path):
    # Stream the data file from the S3 bucket
    s3 = ObjectStore()
    object_key = os.path.join(settings.S3_BASE_PATH, file_path.strip('/'))
    filename = os.path.basename(file_path)
    obj_stream = s3.stream_object(object_key)
    response = StreamingHttpResponse(streaming_content=obj_stream)
    response["Content-Disposition"] = f"attachment; filename={filename}"
    return response


def download_chains(request, slug, aperture_type):
    sed_result = get_object_or_404(
        SEDFittingResult, transient__name=slug, aperture__type=aperture_type
    )
    return stream_sed_output_file(sed_result.chains_file.name)


def download_modelfit(request, slug, aperture_type):
    sed_result = get_object_or_404(
        SEDFittingResult, transient__name=slug, aperture__type=aperture_type
    )
    return stream_sed_output_file(sed_result.model_file.name)


def download_percentiles(request, slug, aperture_type):
    sed_result = get_object_or_404(
        SEDFittingResult, transient__name=slug, aperture__type=aperture_type
    )
    return stream_sed_output_file(sed_result.percentiles_file.name)


def acknowledgements(request):
    context = {"acknowledgements": Acknowledgement.objects.all()}
    return render(request, "acknowledgements.html", context)


def team(request):
    context = {}
    return render(request, "team_members.html", context)


def home(request):
    # This view can only reached in development mode where the webserver proxy, which serves
    # static content and governs endpoints, either does not exist or can be bypassed.
    # In this case it is assumed that the home page should be rendered on-the-fly without
    # reliance on the periodic task in Celery Beat that typically updates the rendering.
    update_home_page_statistics()
    with open(os.path.join(settings.STATIC_ROOT, 'index.html'), 'r') as fp:
        html_content = fp.read()
    return HttpResponse(html_content)


@shared_task
def update_home_page_statistics():
    analytics_results = {}

    task_register_qs = TaskRegister.objects.filter(
        status__message="processed"
    ).prefetch_related()
    for aggregate, qs_value in zip(
        [
            "Basic Information",
            "Host Identification",
            "Host Photometry",
            "Host SED Fitting",
        ],
        [
            "Transients with Basic Information",
            "Transients with Matched Hosts",
            "Transients with Photometry",
            "Transients with SED Fitting",
        ],
    ):
        analytics_results[aggregate] = len(
            filter_transient_categories(
                Transient.objects.all(), qs_value, task_register=task_register_qs
            )
        )

    processed = len(Transient.objects.filter(
        Q(processing_status="blocked") | Q(processing_status="completed")))

    in_progress = len(Transient.objects.filter(
        Q(progress__lt=100) | Q(processing_status='processing')))

    # bokeh_processing_context = plot_pie_chart(analytics_results)
    bokeh_processing_context = plot_bar_chart(analytics_results)

    html_body = render_to_string(
        "index.html",
        {
            "processed": processed,
            "in_progress": in_progress,
            **bokeh_processing_context,
            "show_profile": True,
        },
    )
    with open(os.path.join(settings.STATIC_ROOT, 'index.html'), 'w') as fp:
        fp.write(html_body)


# @user_passes_test(lambda u: u.is_staff and u.is_superuser)
def flower_view(request):
    """passes the request back up to nginx for internal routing"""
    response = HttpResponse()
    path = request.get_full_path()
    path = path.replace("flower", "flower-internal", 1)
    response["X-Accel-Redirect"] = path
    return response


@login_required
@log_usage_metric()
def report_issue(request, item_id):
    item = TaskRegister.objects.get(pk=item_id)
    item.user_warning = True
    item.save()
    return HttpResponseRedirect(
        reverse_lazy("results", kwargs={"slug": item.transient.name})
    )


@login_required
@log_usage_metric()
def resolve_issue(request, item_id):
    item = TaskRegister.objects.get(pk=item_id)
    item.user_warning = False
    item.save()
    return HttpResponseRedirect(
        reverse_lazy("results", kwargs={"slug": item.transient.name})
    )


# Handler for 403 errors
def error_view(request, exception, template_name="403.html"):
    return render(request, template_name)


# Handler for 404 errors
def resource_not_found_view(request, exception, template_name="generic_404.html"):
    return render(request, template_name, status=404)


# View for the privacy policy
def privacy_policy(request):
    return render(request, "privacy_policy.html")


# View for the privacy policy
def healthz(request):
    return HttpResponse()


# Function for getting the SED data plot
def fetch_sed_plot(request):
    transient_name = request.GET.get('transient_name')
    # Acquire the transient object or return 404 not found
    try:
        transient = Transient.objects.get(name__exact=transient_name)
    except Transient.DoesNotExist:
        return JsonResponse(status=404, data={'message': 'Transient not found.'})
    scope = request.GET.get('scope')
    context = render_sed_plot(transient, scope)
    data = {
        f'bokeh_sed_{scope}_div': context[f'bokeh_sed_{scope}_div'],
        f'bokeh_sed_{scope}_script': context[f'bokeh_sed_{scope}_script'],
    }
    return JsonResponse(data)


# Function for getting the cutout FITS plot
def cutout_fits_plot(request):
    if request.method == 'GET':
        transient_name = request.GET.get('transient_name')
        filter = request.GET.get('filter')
        logger.info(f"{transient_name} and {filter}")

        # Acquire the transient object or return 404 not found
        try:
            transient = Transient.objects.get(name__exact=transient_name)
        except Transient.DoesNotExist:
            return JsonResponse(status=404, data={'message': 'Transient not found.'})

        global_aperture = select_aperture(transient)
        local_aperture = Aperture.objects.filter(type__exact="local", transient=transient)
        if filter == '':
            cutout = select_best_cutout(transient.name)
        else:
            cutout_name = f"{transient.name}_{filter}"
            logger.debug(cutout_name)
            cutout = Cutout.objects.get(name__exact=cutout_name)
            logger.info(cutout)
        bokeh_context = plot_cutout_image(
            cutout=cutout,
            transient=transient,
            global_aperture=global_aperture.prefetch_related(),
            local_aperture=local_aperture.prefetch_related(),
        )
        return JsonResponse(bokeh_context)
