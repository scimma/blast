# Generated by Django 5.0.8 on 2025-03-08 00:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('host', '0027_transient_image_trim_status'),
    ]

    operations = [
        migrations.AddField(
            model_name='aperture',
            name='software_version',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddField(
            model_name='aperturephotometry',
            name='software_version',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddField(
            model_name='cutout',
            name='software_version',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddField(
            model_name='host',
            name='software_version',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddField(
            model_name='sedfittingresult',
            name='software_version',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddField(
            model_name='starformationhistoryresult',
            name='software_version',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddField(
            model_name='transient',
            name='software_version',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
    ]
