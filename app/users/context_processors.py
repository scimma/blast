from django.contrib.auth.context_processors import auth
import base64
import binascii
from host.log import get_logger
logger = get_logger(__name__)


def user_profile(request):
    def decoder(b64_string):
        return base64.urlsafe_b64decode(b64_string.encode('utf-8')).decode('utf-8')

    user = auth(request)['user']
    username_b64decoded = ''
    for padded_string in [f'''{user.username}{pad}''' for pad in ['', '=', '==']]:
        try:
            username_b64decoded = decoder(padded_string)
        except binascii.Error:
            pass
        else:
            break

    logger.debug(f'''Decoded username: {username_b64decoded}''')
    return {'username_b64decoded': username_b64decoded}
