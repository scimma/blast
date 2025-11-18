import os

from django.urls import include
from django.urls import path
from django.urls import re_path
from rest_framework.routers import DefaultRouter
from rest_framework.schemas import get_schema_view
from host.workflow import reprocess_transient_view
from host.tasks import retrigger_transient_view

schema_view = get_schema_view(title="Blast API")

from . import views
import api.views

base_path = os.environ.get("BASE_PATH", "").strip("/")
if base_path != "":
    base_path = f"""{base_path}/"""

urlpatterns = [
    path(f"""{base_path}transients/""", views.transient_list, name="transient_list"),
    path(f"""{base_path}add/""", views.add_transient, name="add_transient"),
    path(f"""{base_path}analytics/""", views.analytics),
    path(f"""{base_path}transients/<slug:slug>/""", views.results, name="results"),
    path(
        f"""{base_path}download_chains/<slug:slug>/<str:aperture_type>/""",
        views.download_chains,
        name="download_chains",
    ),
    path(
        f"""{base_path}download_modelfit/<slug:slug>/<str:aperture_type>/""",
        views.download_modelfit,
        name="download_modelfit",
    ),
    path(
        f"""{base_path}download_percentiles/<slug:slug>/<str:aperture_type>/""",
        views.download_percentiles,
        name="download_percentiles",
    ),
    path(f"""{base_path}acknowledgements/""", views.acknowledgements, name="acknowledgements"),
    path(f"""{base_path}team/""", views.team, name="team"),
    path(f"""{base_path}""", views.home),
    path(f"""{base_path}flower/""", views.flower_view),
    path(
        f"""{base_path}reprocess_transient/<slug:slug>/""",
        reprocess_transient_view,
        name="reprocess_transient",
    ),
    path(
        f"""{base_path}retrigger_transient/<slug:slug>/""",
        retrigger_transient_view,
        name="retrigger_transient",
    ),
    path(
        f"""{base_path}report_issue/<item_id>""",
        views.report_issue,
        name="report_issue",
    ),
    path(
        f"""{base_path}resolve_issue/<item_id>""",
        views.resolve_issue,
        name="resolve_issue",
    ),
    path(f"""{base_path}privacy""", views.privacy_policy, name='privacy'),
    path(f"""{base_path}healthz""", views.healthz, name='healthz'),
    path(f"""{base_path}cutout_fits_plot""", views.cutout_fits_plot, name='cutout_fits_plot')
]

router = DefaultRouter()

router.register(r"transient", api.views.TransientViewSet)
router.register(r"aperture", api.views.ApertureViewSet)
router.register(r"cutout", api.views.CutoutViewSet)
router.register(r"filter", api.views.FilterViewSet)
router.register(r"aperturephotometry", api.views.AperturePhotometryViewSet)
router.register(r"sedfittingresult", api.views.SEDFittingResultViewSet)
router.register(r"taskregister", api.views.TaskRegisterViewSet)
router.register(r"task", api.views.TaskViewSet)
router.register(r"host", api.views.HostViewSet)

# Login/Logout
api_url_patterns = [
    re_path(r"^api/", include(router.urls)),
    # TODO: As of v1.5.7, this api/schema endpoint is broken:
    #       "AssertionError: `inflection` must be installed for OpenAPI schema support."
    re_path(r"^api/schema/$", schema_view),
    re_path(r"^api-auth/", include("rest_framework.urls", namespace="rest_framework")),
]

urlpatterns += api_url_patterns

if os.environ.get("SILKY_PYTHON_PROFILER", "false").lower() == "true":
    urlpatterns += [path("silk/", include("silk.urls", namespace="silk"))]
