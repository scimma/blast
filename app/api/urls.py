import os

from django.urls import path

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
    path(
        f"""{base_path}transient/export/<str:transient_name>/""",
        views.export_transient_view,
    ),
    # TO DO: Secure this endpoint with Django REST Framework permission_classes
    # path(
    #     f"""{base_path}workflow/<str:transient_name>""",
    #     views.launch_workflow,
    # ),
]

if os.environ.get("ALLOW_API_POST") == "YES":
    urlpatterns.append(
        path(
            f"""{base_path}transient/post/name=<str:transient_name>&ra=<str:transient_ra>&dec=<str:transient_dec>""",
            views.post_transient,
        )
    )
