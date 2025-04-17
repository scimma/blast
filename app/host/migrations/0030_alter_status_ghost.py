from django.db import migrations


def update_status(apps, schema_editor):
    Status = apps.get_model("host", "Status")
    for obj in Status.objects.filter(message__exact='no GHOST match'):
        obj.message = 'no host match'
        obj.save()


class Migration(migrations.Migration):

    dependencies = [
        ('host', '0029_tasklock'),
    ]

    operations = [
        migrations.RunPython(update_status),
    ]
