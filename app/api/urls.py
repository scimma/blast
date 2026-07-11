import os

from django.urls import path, re_path

from . import views

base_path = os.environ.get("BASE_PATH", "").strip("/")
if base_path != "":
    base_path = f"""{base_path}/"""

urlpatterns = [
    re_path(
        base_path + r"^transient/delete/(?P<transient_name>[a-zA-Z0-9_-]+)/(?P<all>all/|)$",
        views.delete_transient_view,
    ),
    re_path(
        base_path + r"^transient/export/(?P<transient_name>[a-zA-Z0-9_-]+)/(?P<all>all/|)$",
        views.export_transient_view,
    ),
    path(base_path + 'alias/<str:alias>/', views.alias_handler),
    path(base_path + 'alias/<str:alias>/<str:object_type>/<str:name>/', views.alias_handler),
]

if os.environ.get("ALLOW_API_POST") == "YES":
    urlpatterns.append(
        path(
            f"""{base_path}transient/post/name=<str:transient_name>&ra=<str:transient_ra>&dec=<str:transient_dec>""",
            views.post_transient,
        )
    )
