from django.contrib import admin

from .models import Status
from .models import Task
from .models import UsageMetricsLog

# Register your models here.
admin.site.register(Task)
admin.site.register(Status)
admin.site.register(UsageMetricsLog)
