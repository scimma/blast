# users/urls.py
from django.urls import include
from django.urls import re_path

urlpatterns = [re_path(r"^accounts/", include("django.contrib.auth.urls"))]
