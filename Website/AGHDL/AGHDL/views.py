from django.views.generic import TemplateView, ListView
from photos.models import Photo


class HomePage(ListView):
    model = Photo
    template_name = 'home.html'

class ThanksPage(TemplateView):
    template_name = 'thanks.html'

