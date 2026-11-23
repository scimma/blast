from django_celery_beat.models import IntervalSchedule
from django_celery_beat.models import PeriodicTask
from host.tasks import periodic_tasks
from host.views import update_home_page_statistics

for taskrunner in periodic_tasks:
    task = taskrunner.task_name

    interval, created = IntervalSchedule.objects.get_or_create(
        every=taskrunner.task_frequency_seconds, period=IntervalSchedule.SECONDS
    )

    search = PeriodicTask.objects.filter(
        name=taskrunner.task_name,
    )
    if not search:
        PeriodicTask.objects.create(
            interval=interval,
            name=taskrunner.task_name,
            task=taskrunner.task_function_name,
            enabled=taskrunner.task_initially_enabled,
        )

# Render the initial version of the static home page
update_home_page_statistics()

# Schedule a periodic task to update the rendered home page
interval, created = IntervalSchedule.objects.get_or_create(
    every=300, period=IntervalSchedule.SECONDS
)
search = PeriodicTask.objects.filter(
    name='render_home_page',
)
if search:
    search.delete()
PeriodicTask.objects.create(
    interval=interval,
    name='render_home_page',
    task='host.views.update_home_page_statistics',
    enabled=True,
)
