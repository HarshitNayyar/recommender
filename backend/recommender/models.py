from django.db import models

# Create your models here.

class Book(models.Model):
    isbn = models.CharField(max_length=100)
    title = models.CharField(max_length=100)
    author = models.CharField(max_length=100)
    year = models.CharField(max_length=100)
    genre = models.CharField(max_length=100)
    publisher = models.CharField(max_length=100)
    language = models.CharField(max_length=100)
    description = models.CharField(max_length=500)
    keywords = models.JSONField(default=list, blank=True) # Memorizza lista di stringhe

    def __str__(self):
       return f'{self.title} | {self.author}'

class Customer(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
    history = models.ManyToManyField(Book, related_name="readers", blank=True)
    recommendations = models.ManyToManyField(Book, related_name="possible_readers", blank=True)

    def get_recommendation_titles(self):
        return [book.title for book in self.recommendations.all()]

    def __str__(self):
        return f'{self.first_name} {self.last_name}'
    
class Messages(models.Model):
    message = models.JSONField()
    
    class Meta:
        verbose_name_plural = "Messages"

    def __str__(self):
        return f'Json in arrivo: {self.message}'