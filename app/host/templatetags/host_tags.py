from django import template
from django.conf import settings

register = template.Library()


# See https://docs.djangoproject.com/en/5.1/howto/custom-template-tags/
@register.simple_tag(name="app_version")
def app_version(prefix):
    return f'''{prefix}{settings.APP_VERSION}'''


@register.simple_tag(name="support_email")
def support_email():
    return settings.SUPPORT_EMAIL
