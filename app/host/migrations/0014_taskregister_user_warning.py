# Generated by Django 3.2.9 on 2023-11-19 22:14
from django.db import migrations
from django.db import models


class Migration(migrations.Migration):
    dependencies = [
        ("host", "0013_sedfittingresult_mass_surviving_ratio"),
    ]

    operations = [
        migrations.AddField(
            model_name="taskregister",
            name="user_warning",
            field=models.BooleanField(blank=True, null=True),
        ),
    ]
