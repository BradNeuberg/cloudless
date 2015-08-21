from django.conf.urls import url

from . import views

urlpatterns = [
    url('api/getImage', views.getImage),
    url('annotate', views.annotate)
]
