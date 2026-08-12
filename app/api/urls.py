import os

from django.urls import path, re_path
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

from . import views

base_path = os.environ.get("BASE_PATH", "").strip("/")
if base_path != "":
    base_path = f"""{base_path}/"""

urlpatterns = [
    re_path(
        base_path + r"^dataset/(?P<transient_name>[a-zA-Z0-9_-]+)/export$",
        views.DatasetExportView.as_view(),
    ),
    re_path(
        base_path + r"^dataset/(?P<transient_name>[a-zA-Z0-9_-]+)/$",
        views.DatasetView.as_view(),
    ),
    path(base_path + 'alias/<str:alias>/', views.alias_handler_get_delete, ),
    path(base_path + 'alias/<str:alias>/<str:object_type>/<str:name>/', views.alias_handler_post),
]

# if os.environ.get("ALLOW_API_POST") == "YES":
#     urlpatterns.append(
#         path(
#             f"""{base_path}transient/post/name=<str:transient_name>&ra=<str:transient_ra>&dec=<str:transient_dec>""",
#             views.post_transient,
#         )
#     )

urlpatterns += [
    path('schema/openapi', SpectacularAPIView.as_view(), name='schema'),  # Download of API Schema in YAML
    path('schema/swagger-ui/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
]
