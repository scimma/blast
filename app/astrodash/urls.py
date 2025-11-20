from django.urls import path
from astrodash import views

app_name = "astrodash"

urlpatterns = [
    path("analysis-options", views.analysis_options, name="analysis_options"),
    path("template-statistics", views.template_statistics, name="template_statistics"),
    path("template-spectrum", views.template_spectrum, name="template_spectrum"),
    path("line-list", views.template_line_list, name="line_list"),
    path("line-list/elements", views.line_list_elements, name="line_list_elements"),
    path("line-list/element/<str:element>", views.line_list_element, name="line_list_element"),
    path("line-list/filter", views.line_list_filter, name="line_list_filter"),
    path("process", views.process_spectrum, name="process_spectrum"),
    path("estimate-redshift", views.estimate_redshift, name="estimate_redshift"),
    path("models/upload", views.upload_model, name="upload_model"),
    path("models", views.list_models, name="list_models"),
    path("models/<uuid:model_id>", views.get_model_info_view, name="get_model_info"),
    path("models/<uuid:model_id>/delete", views.delete_model, name="delete_model"),
    path("models/<uuid:model_id>/update", views.update_model, name="update_model"),
    path("models/owner/<str:owner>", views.list_models_by_owner, name="models_by_owner"),
    path("batch-process", views.batch_process, name="batch_process"),
]
