from django.db import models
from django.urls import reverse
# Create your models here.


class Photo(models.Model):
    owner = models.ForeignKey('accounts.CustomUser', on_delete=models.CASCADE, null=True)
    img = models.ImageField(upload_to='photos/uploads')
    title = models.CharField(max_length=200, null=True, blank=True) 
    description = models.CharField(max_length=5000, null=True, blank=True) 
    tags = models.CharField(max_length=1000, null=True, blank=True)
    tags_auto = models.CharField(max_length=1000, null=True, blank=True)
    date_added = models.DateTimeField(auto_now_add=True)
    date_modified = models.DateTimeField(auto_now=True)
    likes = models.ManyToManyField('accounts.CustomUser', related_name='photo_likes')

    def __str__(self):
        return self.description

    def get_absolute_url(self):
        return reverse('photos:photo_detail', args=str(self.id))
    

