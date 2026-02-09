import host.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('host', '0043_rename_invalid_transients'),
    ]

    operations = [
        migrations.AlterField(
            model_name='transient',
            name='name',
            field=models.CharField(max_length=64, unique=True, validators=[host.models.Transient.validate_name]),
        ),
    ]
