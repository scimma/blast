"""
Adding OIDC authentication to the project. 
References: https://gitlab.com/nsf-muses/calculation-engine/-/blob/main/app/app_base/auth_backends.py
"""

from mozilla_django_oidc.auth import OIDCAuthenticationBackend
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
import unicodedata
import base64
import re

from host.log import get_logger
logger = get_logger(__name__)


def generate_username(identifier):
    '''Convert identifier into a valid username'''

    def is_valid(username):
        # Using Python 3 and Django 1.11+, usernames can contain alphanumeric
        # (ascii and unicode), _, @, +, . and - characters. So we normalize
        # it and slice at 150 characters.
        valid_username = re.compile(r"^[\w.@+-]+\Z")
        return valid_username.match(username) and len(username) <= 150

    # Replace contiguous space characters with periods, and
    # convert to lowercase with no surrounding space.
    username = re.compile(r"\s+").sub('.', identifier.strip().lower())
    # Reduce instance of multiple periods with single periods. For example,
    # the name "J. Edgar Hoover" would otherwise convert to "j..edgar.hoover"
    username = re.compile(r"\.+").sub('.', username)
    if not is_valid(username):
        # Note: stripping the base64 padding "=" will render the string invalid for base64 decoding
        username = base64.urlsafe_b64encode(username.encode('utf-8')).decode('utf-8').strip('=')
    username = unicodedata.normalize('NFKC', username)[:150]
    # Require conformance to Django standard
    assert is_valid(username)
    return username


def execute_logout(request):
    """
    OIDC logout. Currently a placeholder.
    """
    try:
        request.user.auth_token.delete()
    except (AttributeError, ObjectDoesNotExist):
        pass

    return settings.LOGOUT_REDIRECT_URL


class CustomOIDCAuthenticationBackend(OIDCAuthenticationBackend):
    def __init__(self):
        super().__init__(self)

    def verify_claims(self, claims):
        # Require only that the "sub" claim is provided.
        return claims.get('sub', '')

    def create_user(self, claims):
        """ Overrides the default authentication backend so that Django users are
            created even when the profile and email information is not provided.
        """
        logger.debug(f'''OIDC claims: {claims}''')
        user_info = {
            'username': self.get_username_from_claims(claims),
            'email': self.get_email_from_claims(claims),
            'first_name': claims.get('given_name', ''),
            'last_name': claims.get('family_name', ''),
        }
        logger.debug(f'''New Django user: {user_info}''')
        new_user = self.UserModel.objects.create(
            username=user_info['username'],
            email=user_info['email'],
            first_name=user_info['first_name'],
            last_name=user_info['last_name'],
        )
        return new_user

    def filter_users_by_claims(self, claims):
        """ Return all users matching the specified email.
            If nothing found matching the email, then try the
            unique identifier provided by the OIDC provider.
        """
        users = []
        # Require the "sub" claim. Fail authentication if absent.
        sub = claims.get('sub', '')
        if not sub:
            return self.UserModel.objects.none()
        # # Try associating existing user by email first.
        # email = claims.get('email', '')
        # if email:
        #     users = self.UserModel.objects.filter(email__iexact=email)
        # # If email association fails, use username.
        if len(users) < 1:
            users = self.UserModel.objects.filter(username__iexact=generate_username(sub))
        return users

    def update_user(self, user, claims):
        logger.debug(f'''OIDC claims: {claims}''')
        if "given_name" in claims:
            user.first_name = claims.get('given_name')
        if "family_name" in claims:
            user.last_name = claims.get('family_name')
        user.email = self.get_email_from_claims(claims)
        user.save()
        return user

    def get_username_from_claims(self, claims):
        """Generate username from claims"""
        if "username" in claims:
            return generate_username(claims.get("username"))
        if "preferred_username" in claims:
            return generate_username(claims.get("preferred_username"))
        if "name" in claims:
            return generate_username(claims.get("name"))
        if "family_name" in claims or "given_name" in claims:
            name = f'''{claims.get("given_name", '')} {claims.get("family_name", '')}'''
            return generate_username(name)
        if "email" in claims:
            return generate_username(claims.get("email"))
        if "email_list" in claims:
            email = claims.get("email_list")
            if isinstance(email, list):
                email = email[0]
            return generate_username(email)
        # Use the "sub" claim as a last resort
        return generate_username(claims.get("sub"))

    def get_email_from_claims(self, claims):
        email = ''
        if "email" in claims:
            email = claims.get("email")
        elif "email_list" in claims:
            email = claims.get("email_list")
        if isinstance(email, list):
            email = email[0]
        return email
