from django.contrib.auth.context_processors import auth
import base64
import binascii
from host.log import get_logger
logger = get_logger(__name__)


def user_profile(request):
    def decoder(b64_string):
        return base64.urlsafe_b64decode(b64_string.encode('utf-8')).decode('utf-8')

    def check_perms(user, perm):
        if isinstance(perm, str):
            perms = (perm,)
        else:
            perms = perm
        return user.has_perms(perms)

    user = auth(request)['user']
    username_b64decoded = ''
    # Attempt to base64-decode the username assuming stripped padding; see
    # "app/app/auth_backend.py::generate_username()" for encoding function.
    possible_padding = ['', '=', '==']
    for padded_string in [f'''{user.username}{pad}''' for pad in possible_padding]:
        try:
            username_b64decoded = decoder(padded_string)
        except binascii.Error:
            # If incorrectly padded, try the next possible padding.
            pass
        except Exception as err:
            # Do not treat this as an error, because in general Django usernames are not
            # base64-encoded.
            logger.debug(f'''Error decoding username "{user.username}": {err}''')
            username_b64decoded = ''
        else:
            break
    context = {
        'username_b64decoded': username_b64decoded,
        'has_perm_retrigger_transient': check_perms(user, "host.retrigger_transient"),
        'has_perm_reprocess_transient': check_perms(user, "host.reprocess_transient"),
    }
    return context
