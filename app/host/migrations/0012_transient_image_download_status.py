# Generated by Django 3.2.9 on 2022-01-28 17:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('host', '0011_rename_call_time_externalresourcecall_request_time'),
    ]

    operations = [
        migrations.AddField(
            model_name='transient',
            name='image_download_status',
            field=models.CharField(default='not processed', max_length=20),
        ),
    ]