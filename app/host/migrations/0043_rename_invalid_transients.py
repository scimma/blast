from django.db import migrations
from django.core.exceptions import ValidationError
import re


def validate_name_with_options(name):
    '''This is mostly redundant with the Transient model methods, but those methods cannot be called
       in migration scripts. One key difference is the omission of the prohibition against "SN" and "AT"
       prefixes. Existing transients with these prefixes can retain them.'''
    trans_name_max_length = 64
    if len(name) > trans_name_max_length:
        raise ValidationError(f'''Invalid transient identifier: "{name}" is longer than the max length '''
                              f'''of {trans_name_max_length} characters.''')
    if not bool(re.match(r"[a-zA-Z0-9]+[a-zA-Z0-9_-]*[a-zA-Z0-9]+\Z", name)):
        raise ValidationError(f'''Invalid transient identifier: "{name}" must begin and end with alphanumeric '''
                              '''characters, and may include underscores and hyphens. Spaces are not allowed.''')
    if name.find('--') > -1 or name.find('__') > -1:
        raise ValidationError(f'''Invalid transient identifier: "{name}" may not contain consecutive '''
                              '''underscores or hyphens.''')


def rename_invalid_transients(apps, schema_editor):
    '''Iterate over all transients to validate their names, renaming invalid ones and saving the original value
       as the new display_name field value.'''
    Transient = apps.get_model("host", "Transient")
    all_transients = Transient.objects.all()
    len_all_transients = len(all_transients)
    # Initialize list of all transient names to ensure that new names are unique.
    transient_names = [tr.name for tr in all_transients]
    for trans_idx, transient in enumerate(all_transients):
        try:
            validate_name_with_options(transient.name)
        except ValidationError:
            original_name = transient.name
            # Replace invalid characters with underscores. This must be done before
            # collapsing consecutive underscores.
            replace_invalid_chars = ''
            for char in transient.name:
                if re.search(r'[a-zA-Z0-9_-]', char):
                    replace_invalid_chars += char
                else:
                    replace_invalid_chars += '_'
            transient.name = replace_invalid_chars
            # Collapse consecutive hyphens and underscores
            while transient.name.find('--') > -1 or transient.name.find('__') > -1:
                transient.name = transient.name.replace('--', '-').replace('__', '_')
            # Strip leading/trailing hyphens and underscores
            transient.name = transient.name.strip('-_')
            # Ensure that new name is unique.
            transient_names.remove(original_name)
            idx = 1
            base_name = transient.name
            while transient.name in transient_names:
                transient.name = f'{base_name}_{idx}'
                idx += 1
            try:
                validate_name_with_options(transient.name)
            except ValidationError:
                print(f'''ERROR: [{trans_idx + 1}/{len_all_transients}] Renaming transient "{original_name}" '''
                      f'''to "{transient.name}" failed.''')
            else:
                transient_names.append(transient.name)
                transient.display_name = original_name
                transient.save()
                print(f'''[{trans_idx + 1}/{len_all_transients}] Transient name "{original_name}" '''
                      f'''renamed to "{transient.name}".''')
        else:
            continue


class Migration(migrations.Migration):

    dependencies = [
        ('host', '0042_transient_display_name'),
    ]

    operations = [
        migrations.RunPython(rename_invalid_transients),
    ]
