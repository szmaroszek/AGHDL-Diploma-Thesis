from django import forms
from .models import Photo


class PhotoForm(forms.ModelForm):
    class Meta:
        model = Photo
        fields = ('img', 'title', 'description', 'tags',)

        widgets = {
            'title' : forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter title..'}),
            'description' : forms.Textarea(attrs={'class': 'form-control', 'placeholder': 'Enter description..'}),
            'tags' : forms.Textarea(attrs={'class': 'form-control', 'placeholder': 'Enter tags e.g., #something '}),
        }