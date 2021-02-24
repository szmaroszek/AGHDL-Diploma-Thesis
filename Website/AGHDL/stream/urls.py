from django.urls import path
from . import views

app_name = 'stream'
urlpatterns = [
    path(r'stream/',
        views.StreamView.as_view(),
        name='stream'),
    
    path(r'stream_live/',
        views.live_feed,
        name='stream_live'),
    
    path(r'stop_stream/',
        views.stop_stream,
        name='stop_streaming'),
    
]
