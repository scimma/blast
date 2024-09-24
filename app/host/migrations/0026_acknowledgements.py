from django.db import migrations
from django.core import serializers


def load_fixtures(apps, schema_editor):
    for fixture_file in [
        '/app/host/fixtures/initial/setup_acknowledgements.yaml',
    ]:
        # print(f'''  Loading fixture "{fixture_file}"...''')
        with open(fixture_file) as fp:
            # Inspired by https://stackoverflow.com/a/25981899
            objects = serializers.deserialize('yaml', fp, ignorenonexistent=True)
            for obj in objects:
                obj.save()


class Migration(migrations.Migration):

    dependencies = [
        ('host', '0025_sedfittingresult_dust1_fraction_16_and_more'),
    ]

    operations = [
        migrations.RunPython(load_fixtures),
    ]
