import os

from django.urls import re_path, include
from rest_framework.routers import DefaultRouter

import api.views

base_path = os.environ.get("BASE_PATH", "").strip("/")
if base_path != "":
    base_path = f"""{base_path}/"""

urlpatterns = [
    re_path(
        base_path + r"^dataset/(?P<transient_name>[a-zA-Z0-9_-]+)/export/$",
        api.views.DatasetExportView.as_view(),
    ),
    re_path(
        base_path + r"^dataset/(?P<transient_name>[a-zA-Z0-9_-]+)/$",
        api.views.DatasetView.as_view(),
    ),
]

router = DefaultRouter()

router.register(r"transient", api.views.TransientViewSet)
router.register(r"aperture", api.views.ApertureViewSet)
router.register(r"cutout", api.views.CutoutViewSet, basename="cutout")
router.register(r"filter", api.views.FilterViewSet)
router.register(r"aperturephotometry", api.views.AperturePhotometryViewSet)
router.register(r"sedfittingresult", api.views.SEDFittingResultViewSet, basename="sedfittingresult")
router.register(r"taskregister", api.views.TaskRegisterViewSet)
router.register(r"task", api.views.TaskViewSet)
router.register(r"host", api.views.HostViewSet)
router.register(r"alias", api.views.AliasViewSet)

# Login/Logout
api_url_patterns = [
    re_path("", include(router.urls)),
]

urlpatterns += api_url_patterns
