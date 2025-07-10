import functools
import re

from django.utils import timezone
from django.conf import settings

# from .models import ExternalResourceCall
from .models import UsageMetricsLog
import json


# def log_resource_call(resource_name):
#     """
#     Decorator which saves metadata about a call to an external resource.

#     Args:
#         resource_name (str): Name of the external resource being requested.
#     Returns:
#         Decorator function
#     """

#     def decorator_save(func):
#         @functools.wraps(func)
#         def wrapper_save(*args, **kwargs):
#             value = func(*args, **kwargs)
#             status = value.get("response_message")
#             call = ExternalResourceCall(
#                 resource_name=resource_name,
#                 response_status=status,
#                 request_time=timezone.now(),
#             )
#             call.save()
#             return value

#         return wrapper_save

#     return decorator_save


def log_usage_metric():
    """
    Decorator to log a usage metric based on the request.

    Returns:
        Decorator function.
    """
    def decorator_save(func):
        @functools.wraps(func)
        def wrapper_save(*args, **kwargs):
            request = args[0]
            value = func(*args, **kwargs)
            # Do not record ignored requests
            for ignore_url in settings.USAGE_METRICS_IGNORE_REQUESTS:
                if request.path == ignore_url['path'] and request.method == ignore_url['method']:
                    return value
            # Filter the submitted data object for POST requests
            submitted_data = ''
            if (request.method == "POST"):
                post_data = {k: v for k, v in request.POST.copy().items() if v}
                post_data.pop("csrfmiddlewaretoken", None)
                tns_names = []
                if 'tns_names' in post_data:
                    tns_names = re.split(r'\r\n|\n|\r', post_data['tns_names'])
                    post_data['tns_names'] = tns_names
                full_info = []
                if 'full_info' in post_data:
                    full_info = re.split(r'\r\n|\n|\r', post_data['full_info'])
                    post_data['full_info'] = full_info
                submitted_data = json.dumps(post_data)
            # Create and save the data to a new usage metric log object
            UsageMetricsLog(
                request_url=request.path,
                request_method=request.method,
                request_time=timezone.now(),
                submitted_data=submitted_data,
                request_user=request.user,
                request_ip=request.META["REMOTE_ADDR"],
            ).save()
            return value
        return wrapper_save
    return decorator_save
