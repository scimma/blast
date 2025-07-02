import functools
import time

from django.utils import timezone

from .models import ExternalResourceCall


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
