from django.urls import path
from .views import ImproveImageQualityView

urlpatterns = [
    path('', ImproveImageQualityView.as_view(), name='improve_quality'),
]
