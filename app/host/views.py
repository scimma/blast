import django_filters
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import UserPassesTestMixin
from django.db.models import Q
from django.http import HttpResponse
from django.http import HttpResponseRedirect
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
from host.models import Status
from host.models import TaskRegister
from host.models import TaskRegisterSnapshot
from host.models import Transient
from host.plotting_utils import plot_bar_chart
from host.plotting_utils import plot_cutout_image
from host.plotting_utils import plot_sed
from host.plotting_utils import plot_timeseries
from host.tables import TransientTable
from host.tasks import import_transient_list
from revproxy.views import ProxyView
from silk.profiling.profiler import silk_profile
from host.workflow import transient_workflow


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
            )
            | Q(
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
            )
            | Q(
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
def transient_uploads(request):
    errors = []
    uploaded_transient_names = []

    # add transients -- either from TNS or from RA/Dec/redshift
    if request.method == "POST":
        form = TransientUploadForm(request.POST)

        if form.is_valid():
            info = form.cleaned_data["tns_names"]
            retrigger = form.cleaned_data["retrigger"]
            if info:
                transient_names = info.splitlines()
                # Since the async import task may take a long time, the returned list
                # of uploaded transients will not necessarily match the transients
                # successfully imported from TNS.
                uploaded_transient_names = transient_names
                import_transient_list.delay(transient_names, retrigger=retrigger)

            info = form.cleaned_data["full_info"]
            if info:
                for line in info.split("\n"):
                    # name, ra, dec, redshift, type
                    name, ra, dec, redshift, specclass = line.split(",")
                    redshift = None if redshift.lower() == "none" else redshift
                    specclass = None if specclass.lower() == "none" else specclass
                    info_dict = {
                        "name": name,
                        "ra_deg": ra,
                        "dec_deg": dec,
                        "redshift": redshift,
                        "spectroscopic_class": specclass,
                        "tns_id": 0,
                        "tns_prefix": "",
                        "added_by": request.user,
                    }
                    # check if exists; if not, add it
                    try:
                        Transient.objects.get(name=name)
                        errors += [f"Transient {name} already exists in the database"]
                    except Transient.DoesNotExist:
                        Transient.objects.create(**info_dict)
                        uploaded_transient_names += [name]
    else:
        form = TransientUploadForm()

    context = {
        "form": form,
        "errors": errors,
        "uploaded_transient_names": uploaded_transient_names,
    }
    return render(request, "transient_uploads.html", context)


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


def results(request, slug):

    transients = Transient.objects.all()
    transient = transients.get(name__exact=slug)

    global_aperture = select_aperture(transient)

    local_aperture = Aperture.objects.filter(type__exact="local", transient=transient)
    local_aperture_photometry = AperturePhotometry.objects.filter(
        transient=transient,
        aperture__type__exact="local",
        flux__isnull=False,
        is_validated="true",
    )
    global_aperture_photometry = AperturePhotometry.objects.filter(
        transient=transient, aperture__type__exact="global", flux__isnull=False
    ).filter(Q(is_validated="true") | Q(is_validated="contamination warning"))
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
    for param,var,ptype in zip(
            ["{\\rm log}_{10}(M_{\\ast}/M_{\odot})\,", "{\\rm log}_{10}({\\rm SFR})", "{\\rm log}_{10}({\\rm sSFR})", "{\\rm stellar\ age}", "{\\rm log}_{10}(Z_{\\ast}/Z_{\odot})", "{\\rm log}_{10}(Z_{gas}/Z_{\odot})\,",
             "\\tau_2", "\delta", "\\tau_1/\\tau_2", "Q_{PAH}", "U_{min}", "{\\rm log}_{10}(\gamma_e)\,",
             "{\\rm log}_{10}(f_{AGN})\,", "{\\rm log}_{10}(\\tau_{AGN})\,"],
            ["log_mass", "log_sfr", "log_ssfr", "log_age", "logzsol","gas_logz",
             "dust2","dust_index","dust1_fraction","duste_qpah", "duste_umin", "log_duste_gamma",
             "log_fagn","log_agn_tau"],
            ["normal","normal","normal","normal","Metallicity","Metallicity","Dust","Dust","Dust","Dust","Dust","Dust","AGN","AGN"]
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
        
    if local_sed_obj.exists():
        local_sed_file = local_sed_obj[0].posterior.name
    else:
        local_sed_file = None
    if global_sed_obj.exists():
        global_sed_file = global_sed_obj[0].posterior.name
    else:
        global_sed_file = None

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
    context = {
        **{
            "transient": transient,
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
    return render(request, "results.html", context)


def reprocess_transient(request, slug):
    transient_name = slug
    tasks = TaskRegister.objects.filter(transient__name=transient_name)
    for t in tasks:
        t.status = Status.objects.get(message="not processed")
        t.save()
    transient_workflow.delay(transient_name)

    return HttpResponseRedirect(reverse_lazy("results", kwargs={"slug": slug}))


def download_chains(request, slug, aperture_type):
    sed_result = get_object_or_404(
        SEDFittingResult, transient__name=slug, aperture__type=aperture_type
    )

    filename = sed_result.chains_file.name.split("/")[-1]
    response = HttpResponse(sed_result.chains_file, content_type="text/plain")
    response["Content-Disposition"] = f"attachment; filename={filename}"

    return response


def download_modelfit(request, slug, aperture_type):
    sed_result = get_object_or_404(
        SEDFittingResult, transient__name=slug, aperture__type=aperture_type
    )

    filename = sed_result.model_file.name.split("/")[-1]
    response = HttpResponse(sed_result.model_file, content_type="text/plain")
    response["Content-Disposition"] = f"attachment; filename={filename}"

    return response


def download_percentiles(request, slug, aperture_type):
    sed_result = get_object_or_404(
        SEDFittingResult, transient__name=slug, aperture__type=aperture_type
    )

    filename = sed_result.percentiles_file.name.split("/")[-1]
    response = HttpResponse(sed_result.percentiles_file, content_type="text/plain")
    response["Content-Disposition"] = f"attachment; filename={filename}"

    return response


def acknowledgements(request):
    context = {"acknowledgements": Acknowledgement.objects.all()}
    return render(request, "acknowledgements.html", context)


def home(request):
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

    #    transients = TaskRegisterSnapshot.objects.filter(
    #        aggregate_type__exact=aggregate
    #    )

    #    transients_ordered = transients.order_by("-time")

    #    if transients_ordered.exists():
    #        transients_current = transients_ordered[0].number_of_transients
    #    else:
    #        transients_current = None

    #    analytics_results[f"{aggregate}".replace("_", " ")] = transients_current

    # total = analytics_results["total"]
    # del analytics_results["total"]

    processed = len(
        Transient.objects.filter(
            Q(processing_status="blocked") | Q(processing_status="completed")
        )
    )
    in_progress = len(Transient.objects.filter(processing_status="processing"))

    # bokeh_processing_context = plot_pie_chart(analytics_results)
    bokeh_processing_context = plot_bar_chart(analytics_results)

    return render(
        request,
        "index.html",
        {
            "processed": processed,
            "in_progress": in_progress,
            **bokeh_processing_context,
        },
    )


# @user_passes_test(lambda u: u.is_staff and u.is_superuser)
def flower_view(request):
    """passes the request back up to nginx for internal routing"""
    response = HttpResponse()
    path = request.get_full_path()
    path = path.replace("flower", "flower-internal", 1)
    response["X-Accel-Redirect"] = path
    return response


def report_issue(request, item_id):
    item = TaskRegister.objects.get(pk=item_id)
    item.user_warning = True
    item.save()
    return HttpResponseRedirect(
        reverse_lazy("results", kwargs={"slug": item.transient.name})
    )


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
