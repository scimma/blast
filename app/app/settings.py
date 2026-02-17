import os
from pathlib import Path

######################################################################
# Blast application config
#
APP_VERSION = '1.9.0'
# Data paths
DUSTMAPS_DATA_ROOT = os.environ.get("DUSTMAPS_DATA_ROOT", "/data/dustmaps")
CUTOUT_ROOT = os.environ.get("CUTOUT_ROOT", "/data/cutout_cdn")
SED_OUTPUT_ROOT = os.environ.get("SED_OUTPUT_ROOT", "/data/sed_output")
SBI_TRAINING_ROOT = os.environ.get("SBI_TRAINING_ROOT", "/data/sbi_training_sets")
PROST_OUTPUT_ROOT = os.environ.get("PROST_OUTPUT_ROOT", "/tmp/prost_output")
SBIPP_ROOT = os.environ.get("SBIPP_ROOT", "/data/sbipp")
SBIPP_PHOT_ROOT = os.environ.get("SBIPP_PHOT_ROOT", "/data/sbipp_phot")
TRANSMISSION_CURVES_ROOT = os.environ.get("TRANSMISSION_CURVES_ROOT", "/data/transmission")
TNS_STAGING_ROOT = os.environ.get("TNS_STAGING_ROOT", "/data/tns_staging")
# Email address for support requests
SUPPORT_EMAIL = os.getenv('SUPPORT_EMAIL', "devnull@example.com")
# Workflow task options
TNS_INGEST_TIMEOUT = int(os.environ.get("TNS_INGEST_TIMEOUT", "120"))
QUERY_TIMEOUT = int(os.environ.get("QUERY_TIMEOUT", "60"))
TNS_SIMULATE = os.environ.get("TNS_SIMULATE", "false").lower() in ["true", "t", "1"]
CUTOUT_OVERWRITE = os.environ.get("CUTOUT_OVERWRITE", "False").lower() in ["true", "t", "1"]
# Set JOB_SCRATCH_MAX_SIZE to 0 to determine scratch volume capacity using os.statvfs
JOB_SCRATCH_MAX_SIZE = int(float(os.getenv('JOB_SCRATCH_MAX_SIZE', str(20 * 1024**3))))  # 20 GiB
JOB_SCRATCH_FREE_SPACE = int(float(os.getenv('JOB_SCRATCH_FREE_SPACE', str(5 * 1024**3))))  # 5 GiB
# S3_ENDPOINT_URL example: "https://js2.jetstream-cloud.org:8001"
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "")
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_BASE_PATH = os.getenv("S3_BASE_PATH", "")
S3_LOGS_PATH = os.getenv("S3_LOGS_PATH", "")
# Usage metrics collection system
USAGE_METRICS_LOGROLLER_ENABLED = os.getenv("USAGE_METRICS_LOGROLLER_ENABLED", "true").lower() in ["true", "t"]
USAGE_METRICS_LOGROLLER_FREQUENCY = int(os.getenv('USAGE_METRICS_LOGROLLER_FREQUENCY', '3600'))
USAGE_METRICS_LOGS_PER_ARCHIVE = int(os.getenv('USAGE_METRICS_LOGS_PER_ARCHIVE', '1000'))
USAGE_METRICS_IGNORE_REQUESTS = [
    {
        'path': '/add/',
        'method': 'GET',
    },
]

######################################################################
# Django apps and middlewares
#
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "host",
    "crispy_forms",
    "django_tables2",
    "bootstrap4",
    "crispy_bootstrap4",
    "django_celery_beat",
    "api",
    "users",
    "django_cron",
    "django_filters",
    'django_celery_results',  # TODO: This can be removed if using Redis as Celery backend
    "latexify",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

######################################################################
# Generic application config
#
WSGI_APPLICATION = "app.wsgi.application"
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = os.environ.get("SECRET_KEY", "django-insecure-tn6@rg(#694!6p^c!^0ekz5d)jyxk(dxtx-z9m2%$h&w$p0#+)")
DEBUG = os.environ.get("DJANGO_DEBUG", "false").lower() == "true"
# Internationalization & timezone
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_L10N = True
USE_TZ = True
# Static files (CSS, JavaScript, Images)
STATIC_URL = "/static/"
STATIC_ROOT = os.path.join(os.path.dirname(BASE_DIR), "app/static/")
MEDIA_URL = "/cutouts/"
MEDIA_ROOT = os.path.join(os.path.dirname(BASE_DIR), "data")

######################################################################
# Logging config
#
LOGGING = {
    'version': 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    'loggers': {}
}

######################################################################
# Template config
#
CRISPY_TEMPLATE_PACK = "bootstrap4"
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [
            os.path.join(BASE_DIR, "host", "templates", "host"),
            os.path.join(BASE_DIR, "users", "templates", "registration"),
        ],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "users.context_processors.user_profile",
            ]
        },
    }
]

######################################################################
# Database
#
DATABASES = {
    "default": {
        'ENGINE': os.getenv('DB_ENGINE', 'django.db.backends.postgresql'),
        'NAME': os.getenv('DB_NAME', 'blast'),
        'USER': os.getenv('DB_USER', 'blast'),
        'PASSWORD': os.getenv('DB_PASS', 'password'),
        'HOST': os.getenv('DB_HOST', '127.0.0.1'),
        'PORT': os.getenv('DB_PORT', '5432'),
    },
}
# Default primary key field type.
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

######################################################################
# Webserver config
#
WSGI_APPLICATION = "app.wsgi.application"
HOSTNAMES = os.environ.get("DJANGO_HOSTNAMES", "localhost").split(",")
ALLOWED_HOSTS = ["*"]
CORS_ORIGIN_WHITELIST = ["*"]
CSRF_TRUSTED_ORIGINS = ["http://localhost", "http://localhost:8000", "http://localhost:4000"]
for hostname in HOSTNAMES:
    CSRF_TRUSTED_ORIGINS.append(f"""https://{hostname}""")
CSRF_COOKIE_SECURE = True
ROOT_URLCONF = "app.urls"
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"
    },
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

AUTHENTICATION_BACKENDS = (
    'django.contrib.auth.backends.ModelBackend',
    'app.auth_backend.CustomOIDCAuthenticationBackend',
)

######################################################################
# Celery config
#
CELERY_BEAT_SCHEDULER = "django_celery_beat.schedulers:DatabaseScheduler"
CELERY_TIMEZONE = "UTC"
CELERY_IMPORTS = [
    "host.tasks",
    "host.system_tasks",
    "host.transient_tasks",
]
CELERY_TASK_ROUTES = {
    'Global Host SED Fitting': {'queue': 'sed'},
    'Local Host SED Fitting': {'queue': 'sed'},
}
REDIS_SERVICE = os.environ.get('REDIS_SERVICE', 'redis')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))
# If running Redis in high-availability mode using Sentinel, there must be a master group name set
REDIS_MASTER_GROUP_NAME = os.environ.get('REDIS_MASTER_GROUP_NAME', '')
REDIS_OR_SENTINEL = 'sentinel' if REDIS_MASTER_GROUP_NAME else 'redis'
# Caching config
CACHES = {
    'default': {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": f"{REDIS_OR_SENTINEL}://{REDIS_SERVICE}:{REDIS_PORT}",
    }
}
# Backends & brokers
CELERY_BROKER_URL = f"{REDIS_OR_SENTINEL}://{REDIS_SERVICE}:{REDIS_PORT}"
CELERY_BROKER_TRANSPORT_OPTIONS = {'master_name': REDIS_MASTER_GROUP_NAME}
# Results backend
CELERY_RESULT_BACKEND = f"{REDIS_OR_SENTINEL}://{REDIS_SERVICE}:{REDIS_PORT}"
CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS = {
    'master_name': REDIS_MASTER_GROUP_NAME,
    'retry_policy': {
        'timeout': 5.0
    }
}
CELERYD_REDIRECT_STDOUTS_LEVEL = "INFO"

######################################################################
# Django Rest Framework config
#
INSTALLED_APPS.append('rest_framework')
REST_FRAMEWORK = {
    "DEFAULT_PERMISSION_CLASSES": [
        f'rest_framework.permissions.{os.environ.get("API_AUTHENTICATION")}',
    ],
    "DEFAULT_FILTER_BACKENDS": ["django_filters.rest_framework.DjangoFilterBackend"],
}

######################################################################
# OpenID Connect config
#
INSTALLED_APPS.append('mozilla_django_oidc')
# Configure the OIDC client
OIDC_RP_CLIENT_ID = os.environ.get("OIDC_CLIENT_ID", "")
OIDC_RP_CLIENT_SECRET = os.environ.get("OIDC_CLIENT_SECRET", "")
OIDC_RP_SCOPES = "openid profile email"
OIDC_OP_AUTHORIZATION_ENDPOINT = os.environ.get('OIDC_OP_AUTHORIZATION_ENDPOINT', '')
OIDC_OP_TOKEN_ENDPOINT = os.environ.get('OIDC_OP_TOKEN_ENDPOINT', '')
OIDC_OP_USER_ENDPOINT = os.environ.get('OIDC_OP_USER_ENDPOINT', '')
# Required for keycloak
OIDC_RP_SIGN_ALGO = os.environ.get('OIDC_RP_SIGN_ALGO', 'RS256')
OIDC_OP_JWKS_ENDPOINT = os.environ.get('OIDC_OP_JWKS_ENDPOINT', '')
OIDC_OP_LOGOUT_URL_METHOD = "app.auth_backend.execute_logout"
# OIDC_USERNAME_ALGO = 'app_base.auth_backends.generate_username'
LOGIN_URL = '/oidc/authenticate'
LOGIN_REDIRECT_URL = "/add"
LOGOUT_REDIRECT_URL = os.environ.get('OIDC_OP_LOGOUT_ENDPOINT', '/')
# ALLOW_LOGOUT_GET_METHOD tells mozilla-django-oidc that the front end can logout with a GET
# which allows the front end to use location.href to /auth/logout to logout.
ALLOW_LOGOUT_GET_METHOD = True
# Our django backend is deployed behind nginx/guncorn. By default Django ignores
# the X-FORWARDED request headers generated. mozilla-django-oidc calls
# Django's request.build_absolute_uri method in such a way that the https
# request produces an http redirect_uri. So, we need to tell Django not to ignore
# the X-FORWARDED header and the protocol to use:
USE_X_FORWARDED_HOST = True
LOGGING['loggers']['mozilla_django_oidc'] = {
    'handlers': ['console'],
    'level': 'INFO'
}

######################################################################
# Django Silk profiler
#
SILKY_PYTHON_PROFILER = os.environ.get("SILKY_PYTHON_PROFILER", "false").lower() == "true"
SILKY_INTERCEPT_PERCENT = int(os.environ.get("SILKY_INTERCEPT_PERCENT", "0"))
INSTALLED_APPS.append('silk')
MIDDLEWARE.append('silk.middleware.SilkyMiddleware')
