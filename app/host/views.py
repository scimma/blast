import django_filters
from django.contrib.auth.decorators import login_required, permission_required
from django.contrib.auth.mixins import UserPassesTestMixin
from django.db.models import Q
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.http import StreamingHttpResponse
from django.shortcuts import get_object_or_404
from django.shortcuts import render
from django.urls import re_path
from django.urls import reverse_lazy
from django_tables2 import RequestConfig
from host.forms import ImageGetForm
from host.forms import TransientUploadForm
from host.host_utils import select_aperture
from host.host_utils import select_cutout_aperture
from host.models import Acknowledgement
from host.models import Aperture
from host.models import AperturePhotometry
from host.models import Cutout
from host.models import Filter
from host.models import SEDFittingResult
from host.models import TaskRegister
from host.models import TaskRegisterSnapshot
from host.models import Transient
from host.plotting_utils import plot_bar_chart
from host.plotting_utils import plot_cutout_image
from host.plotting_utils import plot_sed
from host.plotting_utils import plot_timeseries
from host.tables import TransientTable
from host.tasks import import_transient_list
from host.object_store import ObjectStore
from revproxy.views import ProxyView
from silk.profiling.profiler import silk_profile
from django.template.loader import render_to_string
import os
from django.conf import settings
from celery import shared_task
from host.decorators import log_usage_metric
import csv
import io
from host.log import get_logger
logger = get_logger(__name__)


def filter_transient_categories(qs, value, task_register=None):
    if task_register is None:
        task_register = TaskRegister.objects.all()
    if value == "Transients with Basic Information":
        qs = qs.filter(
            pk__in=task_register.filter(
                task__name="Transient information", status__message="processed"
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
    def identify_existing_transients(transient_names, ra_degs=None, dec_degs=None):
        # If no coordinates are provided, search only by transient name
        if not (ra_degs or dec_degs):
            existing_transients = Transient.objects.filter(name__in=transient_names)
            existing_transient_names = [existing_transient.name for existing_transient in existing_transients]
            for transient_name in existing_transient_names:
                logger.info(f'Transient already saved: "{transient_name}"')
            new_transient_names = [transient_name for transient_name in transient_names
                                   if transient_name not in existing_transient_names]
            return existing_transient_names, new_transient_names
        existing_transient_names = []
        new_transient_names = []
        for transient in zip(transient_names, ra_degs, dec_degs):
            transient_name = transient[0]
            transient_ra_deg = transient[1]
            transient_dec_deg = transient[2]
            arcsec_dec = 0.0002778                  # 1 arcsecond in decimal degrees
            arcsec_ra = 0.004167                    # 1 arcsecond when using right ascension units
            existing_transients = Transient.objects.filter(Q(name__exact=transient_name)
                                                           | (Q(ra_deg__gte=transient_ra_deg - arcsec_ra)
                                                              & Q(ra_deg__lte=transient_ra_deg + arcsec_ra)
                                                              & Q(dec_deg__gte=transient_dec_deg - arcsec_dec)
                                                              & Q(dec_deg__lte=transient_dec_deg + arcsec_dec)
                                                              ))
            for existing_transient in existing_transients:
                existing_transient_names.append(existing_transient.name)
            if not existing_transients:
                new_transient_names.append(transient_name)
        for transient_name in existing_transient_names:
            logger.info(f'Transient already saved: "{transient_name}"')
        return existing_transient_names, new_transient_names

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
                existing_transient_names, imported_transient_names = identify_existing_transients(transient_names)
                # Trigger import and processing of new transients
                import_transient_list.delay(imported_transient_names)

            info = form.cleaned_data["full_info"]
            if info:
                trans_info_set = []
                reader = csv.DictReader(io.StringIO(info), fieldnames=['name', 'ra', 'dec', 'redshift', 'specclass'])
                for transient in reader:
                    try:
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
                        }
                    except Exception as err:
                        err_msg = f'''Error parsing line "{transient}": {err}'''
                        logger.error(err_msg)
                        errors.append(err_msg)
                        continue
                    trans_info_set.append(trans_info)
                transient_names = [trans_info['name'] for trans_info in trans_info_set]
                ra_degs = [trans_info['ra_deg'] for trans_info in trans_info_set]
                dec_degs = [trans_info['dec_deg'] for trans_info in trans_info_set]
                existing_transient_names, new_transient_names = identify_existing_transients(
                    transient_names, ra_degs, dec_degs)
                for transient_name in new_transient_names:
                    trans_info = [trans_info for trans_info in trans_info_set
                                  if trans_info['name'] == transient_name][0]
                    if trans_info['name'].startswith("SN") or trans_info['name'].startswith("AT"):
                        trans_name = trans_info["name"]
                        err_msg = f'Error creating transient: {trans_name} starts with an illegal prefix (SN or AT)'
                        logger.error(err_msg)
                        errors.append(err_msg)
                        continue
                    try:
                        Transient.objects.create(**trans_info)
                        defined_transient_names += [trans_info['name']]
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
def results(request, slug):

    transients = Transient.objects.all()
    try:
        transient = transients.get(name__exact=slug)
    except Transient.DoesNotExist:
        return render(request, "transient_404.html", status=404)

    global_aperture = select_aperture(transient)

    local_aperture = Aperture.objects.filter(type__exact="local", transient=transient)
    local_aperture_photometry = AperturePhotometry.objects.filter(
        transient=transient,
        aperture__type__exact="local",
        flux__isnull=False,
        is_validated="true",
    ).order_by('filter__wavelength_eff_angstrom')
    global_aperture_photometry = AperturePhotometry.objects.filter(
        transient=transient, aperture__type__exact="global", flux__isnull=False
    ).filter(
        Q(is_validated="true") | Q(is_validated="contamination warning")
    ).order_by('filter__wavelength_eff_angstrom')
    contam_warning = (
        True
        if len(global_aperture_photometry.filter(is_validated="contamination warning"))
        else False
    )

    local_sed_obj = SEDFittingResult.objects.filter(
        transient=transient, aperture__type__exact="local"
    )
    global_sed_obj = SEDFittingResult.objects.filter(
        transient=transient, aperture__type__exact="global"
    )
    # ugly, but effective?
    local_sed_results, global_sed_results = (), ()
    for param, var, ptype in zip(
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
    ):

        if local_sed_obj.exists():
            local_sed_results += (
                (
                    param,
                    local_sed_obj[0].__dict__[f"{var}_16"],
                    local_sed_obj[0].__dict__[f"{var}_50"],
                    local_sed_obj[0].__dict__[f"{var}_84"],
                    ptype,
                ),
            )
        if global_sed_obj.exists():
            global_sed_results += (
                (
                    param,
                    global_sed_obj[0].__dict__[f"{var}_16"],
                    global_sed_obj[0].__dict__[f"{var}_50"],
                    global_sed_obj[0].__dict__[f"{var}_84"],
                    ptype,
                ),
            )
    local_sfh_results, global_sfh_results = (), ()
    if local_sed_obj.exists():
        for sh in local_sed_obj[0].logsfh.all():
            local_sfh_results += (
                (
                    sh.logsfr_16,
                    sh.logsfr_50,
                    sh.logsfr_84,
                    sh.logsfr_tmin,
                    sh.logsfr_tmax
                ),
            )

    if global_sed_obj.exists():
        for sh in global_sed_obj[0].logsfh.all():
            global_sfh_results += (
                (
                    sh.logsfr_16,
                    sh.logsfr_50,
                    sh.logsfr_84,
                    sh.logsfr_tmin,
                    sh.logsfr_tmax
                ),
            )

    def delete_cached_file(file_path):
        if not isinstance(file_path, str):
            return
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as err:
            logger.error(f'''Error deleting cached SED file "{file_path}": {err}''')

    def download_file_from_s3(file_path):
        try:
            # Download SED results files to local file cache
            object_key = os.path.join(settings.S3_BASE_PATH, file_path.strip('/'))
            s3.download_object(path=object_key, file_path=file_path)
            assert os.path.isfile(file_path)
        except Exception as err:
            logger.error(f'''Error downloading SED file "{file_path}": {err}''')

    if local_sed_obj.exists() or global_sed_obj.exists():
        s3 = ObjectStore()
    if local_sed_obj.exists():
        local_sed_file = local_sed_obj[0].posterior.name
        local_sed_hdf5_filepath = local_sed_file
        local_sed_modeldata_filepath = local_sed_file.replace(".h5", "_modeldata.npz")
        download_file_from_s3(local_sed_hdf5_filepath)
        download_file_from_s3(local_sed_modeldata_filepath)
    else:
        local_sed_file = None
        local_sed_hdf5_filepath = None
        local_sed_modeldata_filepath = None
    if global_sed_obj.exists():
        global_sed_file = global_sed_obj[0].posterior.name
        global_sed_hdf5_filepath = global_sed_file
        global_sed_modeldata_filepath = global_sed_file.replace(".h5", "_modeldata.npz")
        download_file_from_s3(global_sed_hdf5_filepath)
        download_file_from_s3(global_sed_modeldata_filepath)
    else:
        global_sed_file = None
        global_sed_hdf5_filepath = None
        global_sed_modeldata_filepath = None

    all_cutouts = Cutout.objects.filter(transient__name__exact=slug).filter(~Q(fits=""))
    filters = [cutout.filter.name for cutout in all_cutouts]
    all_filters = Filter.objects.all()

    filter_status = {
        filter_.name: ("yes" if filter_.name in filters else "no")
        for filter_ in all_filters
    }
    if request.method == "POST":
        form = ImageGetForm(request.POST, filter_choices=filters)
        if form.is_valid():
            filter = form.cleaned_data["filters"]
            cutout = all_cutouts.filter(filter__name__exact=filter)[0]
    else:
        form = ImageGetForm(filter_choices=filters)

        cutouts = Cutout.objects.filter(transient__name__exact=slug).filter(~Q(fits=""))
        # choose a cutout, if possible
        cutout = None
        choice = 0
        try:
            while cutout is None and choice <= 8:
                cutout = select_cutout_aperture(cutouts, choice=choice).filter(
                    ~Q(fits="")
                )
            if not len(cutout):
                cutout = None
            else:
                cutout = cutout[0]
        except IndexError:
            cutout = None

    bokeh_context = plot_cutout_image(
        cutout=cutout,
        transient=transient,
        global_aperture=global_aperture.prefetch_related(),
        local_aperture=local_aperture.prefetch_related(),
    )
    bokeh_sed_local_context = plot_sed(
        transient=transient,
        type="local",
        sed_results_file=local_sed_file,
    )
    bokeh_sed_global_context = plot_sed(
        transient=transient,
        type="global",
        sed_results_file=global_sed_file,
    )

    if local_aperture.exists():
        local_aperture = local_aperture[0]
    else:
        local_aperture = None

    if global_aperture.exists():
        global_aperture = global_aperture[0]
    else:
        global_aperture = None

    # check for user warnings
    is_warning = False
    for u in transient.taskregister_set.all().values_list("user_warning", flat=True):
        is_warning |= u

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

    transient_taskregister_set = transient.taskregister_set.all()
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

    context = {
        **{
            "transient": transient,
            "transient_taskregister_set": transient_taskregister_set,
            "workflow_diagrams": workflow_diagrams,
            "processing_status_badge_class": processing_status_badge_class,
            "form": form,
            "local_aperture_photometry": local_aperture_photometry.prefetch_related(),
            "global_aperture_photometry": global_aperture_photometry.prefetch_related(),
            "filter_status": filter_status,
            "local_aperture": local_aperture,
            "global_aperture": global_aperture,
            "local_sed_results": local_sed_results,
            "global_sed_results": global_sed_results,
            "local_sfh_results": local_sfh_results,
            "global_sfh_results": global_sfh_results,
            "warning": is_warning,
            "contam_warning": contam_warning,
            "is_auth": request.user.is_authenticated,
        },
        **bokeh_context,
        **bokeh_sed_local_context,
        **bokeh_sed_global_context,
    }
    # Purge temporary cached files
    delete_cached_file(local_sed_hdf5_filepath)
    delete_cached_file(local_sed_modeldata_filepath)
    delete_cached_file(global_sed_hdf5_filepath)
    delete_cached_file(global_sed_modeldata_filepath)
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


class FlowerProxyView(UserPassesTestMixin, ProxyView):
    # `flower` is Docker container, you can use `localhost` instead
    upstream = "http://{}:{}".format("0.0.0.0", 8888)
    url_prefix = "flower"
    rewrite = ((r"^/{}$".format(url_prefix), r"/{}/".format(url_prefix)),)

    def test_func(self):
        return self.request.user.is_superuser

    @classmethod
    def as_url(cls):
        return re_path(r"^(?P<path>{}.*)$".format(cls.url_prefix), cls.as_view())


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
