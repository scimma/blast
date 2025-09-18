from django.db import migrations

def load_tasks(apps, schema_editor):
    Task = apps.get_model("host", "Task")
    for name in [
        'Local aperture photometry photo-z',
        'Validate local photometry photo-z',
        'Local host SED inference photo-z'
    ]:
        Task(name=name).save()

class Migration(migrations.Migration):

    dependencies = [
        ('host', '0036_usagemetricslog_request_user_agent'),
    ]

    operations = [
        migrations.RunPython(load_tasks),
    ]
