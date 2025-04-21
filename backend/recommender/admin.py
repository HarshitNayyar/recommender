# Register your models here.
# vim: set fileencoding=utf-8 :
from django.contrib import admin

import recommender.models as models


class BookAdmin(admin.ModelAdmin):

    list_display = (
        'id',
        'isbn',
        'title',
        'author',
        'year',
        'genre',
        'publisher',
        'language',
        'description',
        'keywords',
    )
    list_filter = (
        'id',
        'isbn',
        'title',
        'author',
        'year',
        'genre',
        'publisher',
        'language',
        'description',
        'keywords',
    )


class CustomerAdmin(admin.ModelAdmin):
    list_display = ('id', 'first_name', 'last_name', 'email', 'display_history', 'display_recommendations')
    readonly_fields = ('display_history', 'display_recommendations')  # <-- campi readonly nel form

    def display_history(self, obj):
        return ", ".join(book.title for book in obj.history.all())
    display_history.short_description = 'History'

    def display_recommendations(self, obj):
        return ", ".join(book.title for book in obj.recommendations.all())
    display_recommendations.short_description = 'Recommendations'

    exclude = ('history', 'recommendations')




class MessagesAdmin(admin.ModelAdmin):

    list_display = ('id', 'message')
    list_filter = ('id', 'message')


def _register(model, admin_class):
    admin.site.register(model, admin_class)


_register(models.Book, BookAdmin)
_register(models.Customer, CustomerAdmin)
_register(models.Messages, MessagesAdmin)
