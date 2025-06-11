from django.db import migrations


def update_acknowledgements(apps, schema_editor):
    Acknowledgement = apps.get_model("host", "Acknowledgement")
    # Replace GHOST citation with Prost
    Acknowledgement.objects.get(name__exact='GHOST').delete()
    Acknowledgement(
        name='Prost',
        description='Used to match transients to host galaxies',
        repository_url='https://github.com/alexandergagliano/Prost',
        website_url='https://astro-prost.readthedocs.io/',
        paper_url='',
        doi='',
    ).save()


class Migration(migrations.Migration):

    dependencies = [
        ('host', '0032_host_photometric_redshift_err_host_redshift_err'),
    ]

    operations = [
        migrations.RunPython(update_acknowledgements),
    ]
