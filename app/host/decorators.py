import functools
import time

from django.utils import timezone

from .models import ExternalResourceCall
from .models import UsageMetricsLogs
import json


def log_resource_call(resource_name):
    """
    Decorator which saves metadata about a call to an external resource.

    Args:
        resource_name (str): Name of the external resource being requested.
    Returns:
        Decorator function
    """

    def decorator_save(func):
        @functools.wraps(func)
        def wrapper_save(*args, **kwargs):
            value = func(*args, **kwargs)
            status = value.get("response_message")
            call = ExternalResourceCall(
                resource_name=resource_name,
                response_status=status,
                request_time=timezone.now(),
            )
            call.save()
            return value

        return wrapper_save

    return decorator_save


def log_process_time(process_name):
    """
    Decorator to time how long a process takes.

    Args:
        process_name (str): Name of the process being timed.
    Returns:
        Decorator function.
    """

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
            request_body = request.method
            if (request.method == "POST"):
                request_body += "\n"
                post_info = request.POST.copy()
                post_info = {k:v for k,v in post_info.items() if v}
                post_info.pop("csrfmiddlewaretoken", None)
                request_body += json.dumps(post_info)
            call = UsageMetricsLogs(
                request_url = request.path,
                request_time=timezone.now(),
                submitted_data = request_body,
                request_user = request.user,
                request_ip = request.META["REMOTE_ADDR"],
            )
            if not (request.path == "/transient_uploads/" and request.method == "GET"):
                call.save()
            return value
        return wrapper_save
    return decorator_save