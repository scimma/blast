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
        # First check if the user has the permission (even anon users)
        if user.has_perms(perms):
            return True
        return False

    user = auth(request)['user']
    username_b64decoded = ''
    for padded_string in [f'''{user.username}{pad}''' for pad in ['', '=', '==']]:
        try:
            username_b64decoded = decoder(padded_string)
        except binascii.Error:
            pass
        except Exception as err:
            logger.error(f'''Error decoding username: {err}''')
            username_b64decoded = ''
        else:
            break

    logger.debug(f'''Decoded username: {username_b64decoded}''')

    context = {
        'username_b64decoded': username_b64decoded,
        'has_perm_retrigger_transient': check_perms(user, "host.retrigger_transient"),
        'has_perm_reprocess_transient': check_perms(user, "host.reprocess_transient"),
    }
    logger.debug(context)
    return context
