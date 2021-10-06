# app/urls.py

from django.urls import path
from .views import index, result

urlpatterns = [
    path('', index, name='index'),
    path('result/', result, name='result'),
]
        