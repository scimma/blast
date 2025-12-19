from django.db import migrations, models
import django.utils.timezone
import uuid


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="SpectrumRecord",
            fields=[
                ("id", models.CharField(primary_key=True, max_length=64, default=uuid.uuid4, serialize=False, editable=False)),
                ("osc_ref", models.CharField(blank=True, null=True, max_length=128, db_index=True)),
                ("file_name", models.CharField(blank=True, null=True, max_length=255)),
                ("x", models.JSONField()),
                ("y", models.JSONField()),
                ("redshift", models.FloatField(blank=True, null=True)),
                ("meta", models.JSONField(blank=True, null=True)),
                ("created_at", models.DateTimeField(default=django.utils.timezone.now)),
            ],
            options={
                "db_table": "astrodash_spectra",
                "ordering": ["-created_at"],
            },
        ),
        migrations.CreateModel(
            name="UserModelRecord",
            fields=[
                ("id", models.UUIDField(primary_key=True, default=uuid.uuid4, serialize=False, editable=False)),
                ("name", models.CharField(blank=True, null=True, max_length=255)),
                ("description", models.TextField(blank=True, null=True)),
                ("owner", models.CharField(blank=True, null=True, max_length=255, db_index=True)),
                ("model_path", models.CharField(max_length=512)),
                ("class_mapping_path", models.CharField(max_length=512)),
                ("input_shape_path", models.CharField(max_length=512)),
                ("meta", models.JSONField(blank=True, null=True)),
                ("created_at", models.DateTimeField(default=django.utils.timezone.now)),
            ],
            options={
                "db_table": "astrodash_user_models",
                "ordering": ["-created_at"],
            },
        ),
    ]
