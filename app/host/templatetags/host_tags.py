from django import template
from django.conf import settings

register = template.Library()


# See https://docs.djangoproject.com/en/5.1/howto/custom-template-tags/
@register.simple_tag(name="app_version")
def app_version(prefix):
    return f'''{prefix}{settings.APP_VERSION}'''


@register.simple_tag(name="service_monitor_url")
def service_monitor_url():
    return settings.SERVICE_MONITOR_URL or ""


@register.simple_tag(name="support_email")
def support_email():
    return settings.SUPPORT_EMAIL


@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)
