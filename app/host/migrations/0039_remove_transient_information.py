from django.db import migrations


def remove_registered_tasks(apps, schema_editor):
    TaskRegister = apps.get_model("host", "TaskRegister")
    for registered_task in TaskRegister.objects.filter(task__name="Transient information"):
        registered_task.delete()


def remove_task(apps, schema_editor):
    Task = apps.get_model("host", "Task")
    Task.objects.get(name='Transient information').delete()


class Migration(migrations.Migration):

    dependencies = [
        ('host', '0038_alter_transient_name'),
    ]

    operations = [
        migrations.RunPython(remove_registered_tasks),
        migrations.RunPython(remove_task),
    ]
