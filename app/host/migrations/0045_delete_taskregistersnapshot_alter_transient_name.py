from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('host', '0044_transient_name_validator'),
    ]

    operations = [
        migrations.DeleteModel(
            name='TaskRegisterSnapshot',
        ),
    ]
