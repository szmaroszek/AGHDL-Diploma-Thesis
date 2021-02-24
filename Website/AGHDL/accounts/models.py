from django.db import models
from django.contrib.auth.models import AbstractUser
from .managers import CustomUserManager




class CustomUser(AbstractUser):
    username = None
    email = models.EmailField(max_length=254, unique=True)
    tags = models.CharField(max_length=10000, null=True)


    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    objects = CustomUserManager()

    def __str__(self):
        return self.email
    