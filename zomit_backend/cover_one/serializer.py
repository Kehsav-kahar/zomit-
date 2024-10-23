from rest_framework import serializers
from .models import cover_one

class two_d_cover_serializers(serializers.ModelSerializer):
    class Meta:
        model = cover_one
        fields = ['id', 'cover_model', 'cover_template', 'created_at']  # Include the new fields
