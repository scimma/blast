from django.db import migrations


def remove_registered_tasks(apps, schema_editor):
    TaskRegister = apps.get_model("host", "TaskRegister")
    for registered_task in TaskRegister.objects.filter(task__name="Log transient processing status"):
        registered_task.delete()


def remove_task(apps, schema_editor):
    Task = apps.get_model("host", "Task")
    Task.objects.get(name='Log transient processing status').delete()


class Migration(migrations.Migration):

    dependencies = [
        ('host', '0029_tasklock'),
    ]

    operations = [
        migrations.RunPython(remove_registered_tasks),
        migrations.RunPython(remove_task),
    ]
