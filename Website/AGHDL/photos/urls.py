from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = 'photos'
urlpatterns = [
    path(r'user_photos/',
        views.ShowUserPhotos.as_view(),
        name='user_photos'),
    
    path(r'add_photo/',
        views.model_form_upload,
        name='add_photo'),

    path(r'photo/<int:pk>',
        views.PhotoDetailView.as_view(),
        name='photo_detail'),

    path(r'like/<int:pk>',
        views.LikeView,
        name='like_photo'),
    
    path(r'unlike/<int:pk>',
        views.UnLikeView,
        name='unlike_photo'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)