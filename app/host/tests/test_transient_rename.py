from django.test import TestCase
from ..models import Transient
from django.core.exceptions import ValidationError
import re


class RenameTransientTest(TestCase):
    fixtures = ["../fixtures/test/setup_test_transient_rename.yaml"]

    def test_rename_transients(self):
        all_transients = Transient.objects.all()
        # Initialize list of all transient names to ensure that new names are unique.
        transient_names = [tr.name for tr in all_transients]
        for transient in all_transients:
            try:
                Transient.validate_name(transient.name)
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
                    transient.name = transient.name.replace('--', '-')
                    transient.name = transient.name.replace('__', '_')
                # Strip leading/trailing hyphens and underscores
                transient.name = transient.name.strip('-_')

                # Ensure that new name is unique.
                transient_names.remove(original_name)
                idx = 1
                base_name = transient.name
                while transient.name in transient_names:
                    transient.name = f'{base_name}_{idx}'
                    idx += 1
                transient_names.append(transient.name)
                print(f'''Transient name "{original_name}" renamed to "{transient.name}".''')
                # Verify that the new name is valid
                self.assertIsNone(Transient.validate_name(transient.name))
            else:
                continue
