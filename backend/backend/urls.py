from django.contrib import admin
from django.urls import path, include
from rest_framework import routers
from recommender.views import BookViewSet, CustomerViewSet, BookCreateBulk, CustomerCreateBulk, CustomerRecommendationsView

router = routers.DefaultRouter()
router.register(r'books', BookViewSet)
router.register(r'customers', CustomerViewSet)

urlpatterns = [
     path('admin/', admin.site.urls),
     path('api/', include(router.urls)),
     path('api/bulk_add_book/', BookCreateBulk.as_view(), name='bulk_add_book'),
     path('api/bulk_add_customer/', CustomerCreateBulk.as_view(), name='bulk_add_customer'),
     path('api/get_recommendations/<int:user_id>', CustomerRecommendationsView.as_view(), name='get_recommendation')
]
