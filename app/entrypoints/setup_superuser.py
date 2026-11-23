import os
from django.contrib.auth import get_user_model

User = get_user_model()  # get the currently active user model,

username = os.environ.get("DJANGO_SUPERUSER_USERNAME", "")
email = os.environ.get("DJANGO_SUPERUSER_EMAIL", "")
password = os.environ.get("DJANGO_SUPERUSER_PASSWORD", "")

if not User.objects.filter(username=username).exists():
    User.objects.create_superuser(
        username=username,
        email=email,
        password=password,
    )
