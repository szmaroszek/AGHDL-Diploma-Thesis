from django.contrib.auth.base_user import BaseUserManager
from django.utils.translation import ugettext_lazy
from django.forms import ValidationError


class CustomUserManager(BaseUserManager):

    def create_user(self, email, password=None,):
    
        if not email:
            raise ValueError(ugettext_lazy('The Email must be set'))

        email = self.normalize_email(email)
        user = self.model(email=email,)

        user.set_password(password)
        user.save()

        return user

    def create_superuser(self, email, password=None, is_staff=True, is_superuser=True, is_active=True):
    
        if not email:
            raise ValueError(ugettext_lazy('The Email must be set'))

        if not is_staff:
            raise ValueError(ugettext_lazy('Superuser must have is_staff=True.'))
        if not is_superuser:
            raise ValueError(ugettext_lazy('Superuser must have is_superuser=True.'))

        email = self.normalize_email(email)
        user = self.model(
            email=email, 
            is_staff=is_staff, 
            is_superuser=is_superuser, 
            is_active=is_active
            )       
        
        user.set_password(password)
        user.save()
        return user