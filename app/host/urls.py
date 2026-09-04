import os

from django.urls import include
from django.urls import path
from rest_framework.schemas import get_schema_view
from host.workflow import reprocess_transient_view
from host.tasks import retrigger_transient_view
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

schema_view = get_schema_view(title="Blast API")

from . import views

base_path = os.environ.get("BASE_PATH", "").strip("/")
if base_path != "":
    base_path = f"""{base_path}/"""

urlpatterns = [
    path(f"""{base_path}transients/""", views.transient_list, name="transient_list"),
    path(f"""{base_path}add/""", views.add_transient, name="add_transient"),
    path(f"""{base_path}transients/<slug:transient_name>/""", views.results, name="results"),
    path(f"""{base_path}acknowledgements/""", views.acknowledgements, name="acknowledgements"),
    path(f"""{base_path}team/""", views.team, name="team"),
    path(f"""{base_path}""", views.home),
    path(
        f"""{base_path}reprocess_transient/<slug:transient_name>/""",
        reprocess_transient_view,
        name="reprocess_transient",
    ),
    path(
        f"""{base_path}retrigger_transient/<slug:transient_name>/""",
        retrigger_transient_view,
        name="retrigger_transient",
    ),
    path(
        f"""{base_path}issue_handling/<str:action>/<int:item_id>""",
        views.issue_handling,
        name="issue_handling",
    ),
    path(f"""{base_path}privacy""", views.privacy_policy, name='privacy'),
    path(f"""{base_path}healthz""", views.healthz, name='healthz'),
    path(f"""{base_path}cutout_fits_plot""", views.cutout_fits_plot, name='cutout_fits_plot'),
    path(f"""{base_path}fetch_sed_plot""", views.fetch_sed_plot, name='fetch_sed_plot'),
    path(f"""{base_path}fetch_host_spectrum_plot""", views.fetch_host_spectrum_plot, name='fetch_host_spectrum_plot'),
]


urlpatterns += [
    path('api/schema/openapi/', SpectacularAPIView.as_view(), name='schema'),  # Download of API Schema in YAML
    path('swagger-ui/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
]

if os.environ.get("SILKY_PYTHON_PROFILER", "false").lower() == "true":
    urlpatterns += [path("silk/", include("silk.urls", namespace="silk"))]
