import os

from django.urls import path, re_path

from . import views

base_path = os.environ.get("BASE_PATH", "").strip("/")
if base_path != "":
    base_path = f"""{base_path}/"""

urlpatterns = [
    path(
        f"""{base_path}transient/get/<str:transient_name>""",
        views.get_transient_science_payload,
    ),
    path(
        f"""{base_path}transient/delete/<str:transient_name>/""",
        views.delete_transient_view,
    ),
    re_path(
        base_path + r"^transient/export/(?P<transient_name>[a-zA-Z0-9_-]+)/(?P<all>all/|)$",
        views.export_transient_view,
    ),
]

if os.environ.get("ALLOW_API_POST") == "YES":
    urlpatterns.append(
        path(
            f"""{base_path}transient/post/name=<str:transient_name>&ra=<str:transient_ra>&dec=<str:transient_dec>""",
            views.post_transient,
        )
    )
