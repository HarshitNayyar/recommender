# Recommender System V3

## Notes:

This is my first machine learning project, where I integrated content-based filtering with collaborative filtering for a library recommendation system. The solution combines user preferences with similar users' behaviors to suggest books. If I were to redo this, I would probably choose FastAPI or Flask for building the microservice instead of Django, as they are faster and more lightweight frameworks, better suited for handling machine learning models and APIs.

This recommender system was integrated as a microservice for a Library Management System, which allows seamless book recommendations based on user history and preferences. The microservice architecture is scalable and easily extendable, making it ideal for future features like real-time recommendation updates and more sophisticated clustering algorithms.

This project demonstrates my ability to implement machine learning models in a production environment, apply JMS messaging for inter-service communication, and integrate various technologies to provide personalized recommendations.

## Hybridization of Filtering

The book recommendation filtering, previously based solely on content-based filtering strategy, has been hybridized by adding a collaborative filtering module.

To implement the hybridization, we had two strategies available:

### Sequential Hybridization
L'ibridizzazione sequenziale prevede che il modulo **Collaborative** Filtering faccia una **prima selezione** di libri a seconda dei profili utenti e successivamente il modulo **Content-Based** filtering filtra questa lista, producendo la lista di raccomandazioni pertinenti.

### Parallel Hybridization
Parallel hybridization involves the implementation of two separate modules: one for Content-Based Filtering and one for Collaborative Filtering. Both modules produce a list of recommendations, which are then passed through a Rank Aggregation algorithm to combine a list of relevant recommendations.

We chose this hybridization strategy as it was easier to implement initially. If necessary, we will move to the sequential strategy in the future.

#### Addition of Collaborative Filtering Module
For the addition of the collaborative filtering module, we chose two different strategies based on different needs:
- Global Popularity: Based on the total number of reservations of a book (absolute popularity). This is used as a fallback in the case of a cold start (an user without any reserved books).
- User Similarity: Based on the similarity between users. This is used for recommendations based on the behavior of similar users.

## Integration with Library Management System Project

### Added JMS Service for Book, Reservation, and Customer:
We added several JMS services, so when a book, reservation, or customer is added, modified, or deleted, an appropriate message is sent to the recommendation microservice queue. These messages are then managed by events and RESTful APIs in Django.

### Added RESTful API in Django:
The microservice was integrated with a Dockerized MySQL database to manage the persistence of data. If I had to do it all over again I'd probably choose FastAPI or Flask.

## Database Integration
The microservice was integrated with a Dockerized MySQL database to manage the persistence of data.

## Core Recommender

#### 1. Data Extraction
**From**: Django Database (Book, Customer)
**To**: Pandas DataFrame

```python
base_dir = os.path.dirname(os.path.abspath(__file__))
cache_path = os.path.join(base_dir, 'cache', 'bert_embeddings.npy')
```

---

#### 2. Preprocessing  
Preparing data for embedding:

- Filling NaN values (fill_nan_values)
- Converting history from ID to ISBN
- Preparing text for embedding (concatenation with weights)

---

#### 3. Features embedding  
Convert the book catalog into embedding vectors:

```python
get_bert_feature_vectors(books)
```

- Generate embeddings using BERT (sentence-transformers)
- Local cache (.npy) to avoid regeneration if data remains unchanged

---

#### 4. User embedding  
Takes the list of users associated with their history, an ISBN-to-book index dictionary, and a numpy array of book semantic embeddings to produce a unique semantic embedding representing the user's preferences.

```python
get_user_embeddings(users_histories, isbn_to_book_index, feature_vectors)
# isbn_to_book_index: dizionario {ISBN: indice corrispondente nel dataframe}
```

- Calculate the average of embeddings for books read by each user

---

#### 5. Clustering utenti  
Grouping users based on similar tastes (embeddings):

```python
cluster_users_dbscan_best_eps(user_embeddings, **dbscan_kwargs)
```

- Dimensionality reduction with PCA (Principal Component Analysis):
  - PCA is used to reduce the output dimensions of BERT (MiniLM) from 384 to 10, improving clustering efficiency.
- Automatic search for the optimal eps value for DBSCAN (find_best_dbscan_eps)
- Assigning users to clusters (including outliers labeled as -1)

Note: dbscan_kwargs is a dictionary of parameters passed dynamically to the function.

---

#### 6. Cluster Labeling  
Assigning a descriptive label to each cluster (e.g., "Fantasy, Adventure"):

```python
get_cluster_label(cluster_isbns, books, top_n=3)
```

- Extracting most frequent genres and keywords
- Generating a representative name for each cluster

---

#### 7. Result Visualization (Debug)
- Printing the users per cluster (name, surname, ID), including outliers.

---

#### 8. Personalized Recommendation (Core) 
Suggesting books to users with similar tastes (collaborative approach):

```python
get_recommendations_for_user(user_id, top_n=5)
```

- Retrieve the user's cluster membership
- Exclude books already read
- Suggest the most popular books in the cluster

---

#### Output  
- List of recommended book IDs.

## Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt     --extra-index-url https://download.pytorch.org/whl/cpu
```
