"""
Adding OIDC authentication to the project. 
References: https://gitlab.com/nsf-muses/calculation-engine/-/blob/main/app/app_base/auth_backends.py
"""

from mozilla_django_oidc.auth import OIDCAuthenticationBackend
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
import unicodedata


class KeycloakOIDCAuthenticationBackend(OIDCAuthenticationBackend):
    def __init__(self):
        super().__init__(self)

    
    def create_user(self, claims):
        """ Overrides Authentication Backend so that Django users are
            created with the keycloak preferred_username.
            If nothing found matching the email, then try the username.
        """
        email = self.get_email(claims)
        user_info = {
            'username':claims.get("preferred_username"),
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
        email = claims.get("email")
        user.first_name = claims.get('given_name', '')
        user.last_name = claims.get('family_name', '')
        user.email = email
        user.save()
        return user
    
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

def generate_username(email):
    # Using Python 3 and Django 1.11+, usernames can contain alphanumeric
    # (ascii and unicode), _, @, +, . and - characters. So we normalize
    # it and slice at 150 characters.
    return unicodedata.normalize('NFKC', email)[:150]