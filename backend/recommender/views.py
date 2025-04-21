from rest_framework import viewsets, status
from recommender.serializers import BookSerializer, CustomerSerializer
from recommender.models import Book, Customer
from rest_framework.response import Response
from rest_framework.exceptions import NotFound
from rest_framework.views import APIView
import json
from .cluster_recommender import get_recommendations_for_user

# Serializers define the API representation

class BookViewSet(viewsets.ViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer

    # ---- getBookById ----
    def retrieve(self, request, pk=None):
        if not pk:
            return Response({"detail": "Book ID is required."}, status=status.HTTP_400_BAD_REQUEST)
        
        book = Book.objects.get(pk=pk)
        serializer = BookSerializer(book)
        return Response(serializer.data)
    
    # ---- getAllBooks ----
    def list(self, request):
        queryset = Book.objects.all()
        serializer = BookSerializer(queryset, many=True)
        return Response(serializer.data)

    # ---- createBook ----
    def create(self, request):
        serializer = self.serializer_class(data=request.data)

        if serializer.is_valid():
            book = serializer.save()
            book.save()
            
            return Response(BookSerializer(book).data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    # ---- updateBook ----
    def update(self, request, pk=None):
        if not pk:
            return Response({"detail": "Book ID is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            book = Book.objects.get(pk=pk)
        except Book.DoesNotExist:
            raise NotFound(detail="Book not found.")
        
        serializer = self.serializer_class(book, data=request.data, partial=True)
        if serializer.is_valid():
            book = serializer.save()
            return Response(BookSerializer(book).data, status=status.HTTP_200_OK)
        else:
            print("Serializer errors:", serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    # ---- deleteBook ----  
    def destroy(self, request, pk=None):
        if not pk:
            return Response({"detail": "Book ID is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            book = Book.objects.get(pk=pk)
            book.delete()
            return Response({"message": "Book deleted successfully."}, status=status.HTTP_204_NO_CONTENT)
        
        except Book.DoesNotExist:
            return Response({"error": "Book not found."}, status=status.HTTP_404_NOT_FOUND)

class CustomerViewSet(viewsets.ViewSet):
    queryset = Customer.objects.all()
    serializer_class = CustomerSerializer

    # ---- getCustomerById ----
    def retrieve(self, request, pk=None):
        if not pk:
            return Response({"detail": "Customer ID is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            customer = Customer.objects.get(pk=pk)
        except Customer.DoesNotExist:
            raise NotFound(detail="Customer not found.")
        
        serializer = CustomerSerializer(customer)
        return Response(serializer.data)
    
    # ---- getAllCustomers ----
    def list(self, request):
        queryset = Customer.objects.all()
        serializer = CustomerSerializer(queryset, many=True)
        return Response(serializer.data)

    # ---- createCustomer ----
    def create(self, request):
        serializer = self.serializer_class(data=request.data)

        if serializer.is_valid():
            customer = serializer.save()
            return Response(CustomerSerializer(customer).data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    # ---- updateCustomer ----
    def update(self, request, pk=None):
        if not pk:
            return Response({"detail": "Customer ID is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            customer = Customer.objects.get(pk=pk)
        except Customer.DoesNotExist:
            raise NotFound(detail="Customer not found.")
        
        serializer = self.serializer_class(customer, data=request.data, partial=True)
        if serializer.is_valid():
            customer = serializer.save()
            return Response(CustomerSerializer(customer).data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    # ---- deleteCustomer ---- 
    def destroy(self, request, pk=None):
        if not pk:
            return Response({"detail": "Customer ID is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            customer = Customer.objects.get(pk=pk)
            customer.delete()
            return Response({"message": "Customer deleted successfully."}, status=status.HTTP_204_NO_CONTENT)
        except Customer.DoesNotExist:
            return Response({"error": "Customer not found."}, status=status.HTTP_404_NOT_FOUND)

class BookCreateBulk(APIView):
    def post(self, request):
        try:
            # Validate the incoming data using request.data (DRF automatically parses JSON)
            data = request.data
            
            # Validate the JSON structure (it should be a list of books)
            if not isinstance(data, list):
                return Response({'error': 'Invalid JSON structure. Expected a list of books.'}, status=status.HTTP_400_BAD_REQUEST)
            
            books_to_add = []
            for book_data in data:
                # Ensure each book has the necessary fields
                if 'isbn' not in book_data or 'title' not in book_data:
                    return Response({'error': 'Missing required fields for one or more books.'}, status=status.HTTP_400_BAD_REQUEST)
                
                books_to_add.append(Book(
                    isbn=book_data.get('isbn'),
                    title=book_data.get('title'),
                    author=book_data.get('author'),
                    year=book_data.get('year'),
                    genre=book_data.get('genre'),
                    publisher=book_data.get('publisher'),
                    language=book_data.get('language'),
                    description=book_data.get('description'),
                    keywords=book_data.get('keywords', [])
                ))

            # Bulk insert the books into the database
            Book.objects.bulk_create(books_to_add)
            
            # Return a success response with the number of books added
            return Response({'message': f'{len(books_to_add)} books successfully added.'}, status=status.HTTP_201_CREATED)
        
        except json.JSONDecodeError:
            return Response({'error': 'Invalid JSON format.'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
class CustomerCreateBulk(APIView):
    def post(self, request):
        try:
            # Ottieni i dati dalla richiesta
            data = request.data
            
            # Verifica che i dati siano una lista di clienti
            if not isinstance(data, list):
                return Response({'error': 'Struttura JSON non valida. Si aspettano una lista di clienti.'}, status=status.HTTP_400_BAD_REQUEST)
            
            customers_to_add = []
            for customer_data in data:
                # Assicurati che ogni cliente abbia i campi necessari
                if 'first_name' not in customer_data or 'email' not in customer_data:
                    return Response({'error': 'Mancano campi obbligatori per uno o più clienti.'}, status=status.HTTP_400_BAD_REQUEST)
                
                # Ottieni i libri dalla lista di ISBN
                history_isbns = customer_data.get('history', [])
                books = Book.objects.filter(isbn__in=history_isbns)  # Trova i libri con gli ISBN specificati

                # Crea il cliente e aggiungi i libri alla sua storia
                customer = Customer(
                    first_name=customer_data.get('first_name'),
                    last_name=customer_data.get('last_name'),
                    email=customer_data.get('email'),
                )

                customer.save()  # Salva prima il cliente

                # Associa i libri al cliente
                customer.history.set(books)  # Usa il metodo set() per una relazione ManyToManyField

                customers_to_add.append(customer)
            
            # Restituisci una risposta di successo
            return Response({'message': f'{len(customers_to_add)} clienti creati e la loro storia associata ai libri.'}, status=status.HTTP_201_CREATED)
        
        except json.JSONDecodeError:
            return Response({'error': 'Formato JSON non valido.'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class CustomerRecommendationsView(APIView):
    def get(self, request, user_id):
        try:
            customer = Customer.objects.get(id=user_id)
        except Customer.DoesNotExist:
            return Response({"error": "Utente non trovato."}, status=status.HTTP_404_NOT_FOUND)

        # ✅ Usa direttamente la funzione che restituisce gli ID dei libri
        recommended_book_ids = get_recommendations_for_user(user_id, top_n=5)

        if not recommended_book_ids:
            return Response({"message": "Nessuna raccomandazione disponibile per l'utente."}, status=status.HTTP_200_OK)

        # 🔍 Recupera gli oggetti Book corrispondenti
        recommended_books = Book.objects.filter(id__in=recommended_book_ids)
        # print(f'recommended_books', recommended_books)

        # ✅ Salva i libri raccomandati nel campo ManyToMany
        customer.recommendations.set(recommended_books)
        customer.save()

        # 📦 Serializza i libri consigliati
        recommendations_data = BookSerializer(recommended_books, many=True).data

        return Response({
            "user_id": user_id,
            "recommendations": recommendations_data
        }, status=status.HTTP_200_OK)

