# Generated by Django 3.2.9 on 2023-12-26 03:06
from django.db import migrations
from django.db import models


class Migration(migrations.Migration):

    dependencies = [
        ("host", "0016_transient_added_by"),
    ]

    operations = [
        migrations.AlterField(
            model_name="aperturephotometry",
            name="is_validated",
            field=models.CharField(blank=True, max_length=40, null=True),
        ),
    ]