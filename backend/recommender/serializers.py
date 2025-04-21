from rest_framework import serializers
from recommender.models import Book, Customer

class BookSerializer(serializers.ModelSerializer):
    keywords = serializers.SerializerMethodField()

    class Meta:
        model = Book
        fields = [
            'isbn',
            'title',
            'author',
            'description',
            'keywords'
        ]

    def get_keywords(self, obj):
        return obj.keywords or []


class CustomerSerializer(serializers.ModelSerializer):
    history = BookSerializer(many=True)
    recommendations = BookSerializer(many=True)
    
    class Meta:
        model = Customer
        fields = '__all__'