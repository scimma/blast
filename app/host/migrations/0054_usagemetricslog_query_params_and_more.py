"""UsageMetricsLog and Alias model update migration script

Add a new JSONField query_params field to the the UsageMetricsLog model and convert the submitted_data field
from TextField to JSONField.

Replaces original Alias primary key with alias string itself. Should not need conversion logic because the
alias values are already unique.
"""

from django.db import migrations, models
import json
import host.models


def convert_submitted_data(apps, schema_editor):
    UsageMetricsLog = apps.get_model("host", "UsageMetricsLog")
    # Use .iterator() for more efficient one-pass processing of large tables
    for record in UsageMetricsLog.objects.all().iterator():
        raw_value = record.submitted_data
        if raw_value is None or raw_value.strip() == "":
            record.submitted_data_json = None
            # Using update_fields= performs a smaller UPDATE operation
            record.save(update_fields=["submitted_data_json"])
            continue
        try:
            parsed_value = json.loads(raw_value)
        except (TypeError, ValueError) as exc:
            # !r tells an f-string to format the value using its repr() representation instead of its normal str()
            # representation, useful in error messages because it makes invisible or ambiguous characters visible.
            raise RuntimeError(f"UsageMetricsLog {record.pk} contains invalid JSON: {raw_value!r}") from exc
        record.submitted_data_json = parsed_value
        record.save(update_fields=["submitted_data_json"])


class Migration(migrations.Migration):

    dependencies = [
        ('host', '0053_host_spectrum_task'),
    ]

    operations = [
        migrations.AddField(
            model_name="usagemetricslog",
            name="submitted_data_json",
            field=models.JSONField(null=True, blank=True),
        ),
        migrations.RunPython(convert_submitted_data),
        migrations.RemoveField(
            model_name="usagemetricslog",
            name="submitted_data",
        ),
        migrations.RenameField(
            model_name="usagemetricslog",
            old_name="submitted_data_json",
            new_name="submitted_data",
        ),
        migrations.AlterField(
            model_name='usagemetricslog',
            name='submitted_data',
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name='usagemetricslog',
            name='query_params',
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.RemoveField(
            model_name='alias',
            name='id',
        ),
        migrations.AlterField(
            model_name='alias',
            name='alias',
            field=models.CharField(max_length=64, primary_key=True, serialize=False, unique=True,
                                   validators=[host.models.Alias.validate_name]),
        ),
    ]
