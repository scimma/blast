"""
Adding OIDC authentication to the project. 
References: https://gitlab.com/nsf-muses/calculation-engine/-/blob/main/app/app_base/auth_backends.py
"""

from mozilla_django_oidc.auth import OIDCAuthenticationBackend
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
import unicodedata
import re

from host.log import get_logger
logger = get_logger(__name__)

class KeycloakOIDCAuthenticationBackend(OIDCAuthenticationBackend):
    def __init__(self):
        super().__init__(self)

    def create_user(self, claims):
        """ Overrides Authentication Backend so that Django users are
            created with the keycloak preferred_username.
            If nothing found matching the email, then try the username.
        """
        ''' As of 2025/06/25, CILogon provides these claims:
            {
                'name': 'Jon Bon Jovi',
                'given_name': 'Jon Bon',
                'family_name': 'Jovi',
                'email': 'rockstar@example.com'
                'sub': 'http://cilogon.org/serverA/users/12345678',
                'aud': 'cilogon:/client_id/71dd2716fc1f3460d49afbce1d74f068',
                'acr': 'https://refeds.org/profile/mfa',
                'azp': 'cilogon:/client_id/71dd2716fc1f3460d49afbce1d74f068',
                'iss': 'https://cilogon.org',
                'jti': 'https://cilogon.org/oauth2/idToken//ac2...389/17...62',
            }
        '''
        logger.debug(f'''OIDC claims: {claims}''')
        email = self.get_email(claims)
        user_info = {
            'username': self.get_username(claims),
            'email': email,
            'first_name': claims.get('given_name', ''),
            'last_name': claims.get('family_name', ''),
        }
        new_user = self.UserModel.objects.create(
            username=user_info['username'],
            email=user_info['email'],
            first_name=user_info['first_name'],
            last_name=user_info['last_name'],
        )
        return new_user

    def filter_users_by_claims(self, claims):
        """ Return all users matching the specified email.
            If nothing found matching the email, then try the username
        """
        email = claims.get('email')
        preferred_username = claims.get('preferred_username')

        if not email:
            return self.UserModel.objects.none()
        users = self.UserModel.objects.filter(email__iexact=email)

        if len(users) < 1:
            if not preferred_username:
                return self.UserModel.objects.none()
            users = self.UserModel.objects.filter(username__iexact=preferred_username)
        return users

    def update_user(self, user, claims):
        logger.debug(f'''OIDC claims: {claims}''')
        email = claims.get("email")
        user.first_name = claims.get('given_name', '')
        user.last_name = claims.get('family_name', '')
        user.email = email
        user.save()
        return user

    def get_username(self, claims):
        """Extract username from claims"""
        regex = re.compile(r"\s+")
        if "username" in claims:
            username = claims.get("username")
        elif "preferred_username" in claims:
            username = claims.get("preferred_username")
        elif "name" in claims:
            name = regex.sub('.', claims.get("name")).strip().lower()
            username = generate_username(name)
        elif "family_name" in claims and "given_name" in claims:
            name = f'''{claims.get("given_name")} {claims.get("family_name")}'''
            name = regex.sub('.', name).strip().lower()
            username = generate_username(name)
        else:
            # Use the email address as a last resort
            username = generate_username(self.get_email(claims))
        return username

    def get_email(self, claims):
        """Extract email from claims"""
        email = ""
        if "email" in claims:
            email = claims.get("email")
        elif "email_list" in claims:
            email = claims.get("email_list")

        if isinstance(email, list):
            email = email[0]
        return email


def execute_logout(request):
    """
    OIDC logout. Currently a placeholder.
    """
    try:
        request.user.auth_token.delete()
    except (AttributeError, ObjectDoesNotExist):
        pass

    return settings.LOGOUT_REDIRECT_URL


def generate_username(identifier):
    # Using Python 3 and Django 1.11+, usernames can contain alphanumeric
    # (ascii and unicode), _, @, +, . and - characters. So we normalize
    # it and slice at 150 characters.
    return unicodedata.normalize('NFKC', identifier)[:150]
