from django.db import migrations
from django.core import serializers


def load_fixtures(apps, schema_editor):
    for fixture_file in [
        '/app/astrodash/fixtures/initial/initial_spectrum.yaml',
        '/app/astrodash/fixtures/initial/initial_user_model.yaml',
    ]:
        print(f'''  Loading fixture "{fixture_file}"...''')
        with open(fixture_file) as fp:
            # Inspired by https://stackoverflow.com/a/25981899
            objects = serializers.deserialize('yaml', fp, ignorenonexistent=True)
            for obj in objects:
                obj.save()


class Migration(migrations.Migration):

    dependencies = [
        ('astrodash', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(load_fixtures),
    ]
