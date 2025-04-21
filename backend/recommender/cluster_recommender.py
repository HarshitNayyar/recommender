import os
import pandas as pd
import numpy as np
from django.core.cache import cache
import pickle
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from collections import Counter
from recommender.models import Book, Customer  

# ---- Dataframes ----

books_queryset = Book.objects.all().values('id', 'isbn', 'title', 'author', 'genre', 'description', 'keywords')
customers_queryset = Customer.objects.all().values('id', 'first_name', 'last_name', 'history')

books = pd.DataFrame(list(books_queryset))
customers = pd.DataFrame(list(customers_queryset))

# Mapping from book ID to ISBN
if 'id' in books.columns:
    id_to_isbn = dict(zip(books['id'], books['isbn']))
else:
    print("The 'id' column doesn't exist in the DataFrame")

# Conversion column 'history' from ID to ISBN
def convert_ids_to_isbns(history):
    if isinstance(history, list):
        return [id_to_isbn.get(book_id) for book_id in history if book_id in id_to_isbn]
    elif isinstance(history, int):
        return [id_to_isbn.get(history)]
    return []

if 'history' in customers.columns:
    customers['history'] = customers['history'].apply(convert_ids_to_isbns)
else:
    print("The 'history' column doesn't exist in the DataFrame")

# ---- Data Cleaning ----
def fill_nan_values(books, selected_features):
    for feature in selected_features:
        books[feature] = books[feature].fillna('')
    return books

# ---- Feature Embedding ----

"""
Input:
    - books: dataframe containing the book data (including 'genre', 'keywords', 'description', 'author', 'title')
    - weight: weight factor for keyword (default: 2)
    - cache_path: path for the cache file to save/load embeddings (default: 'cache/bert_embeddings.npy')

Azioni:
    - Select the relevant columns for the embedding generation: ['genre', 'keywords', 'description', 'author', 'title']
    - Fills the NaN values in the selected columns using the fill_nan_values function
    - Concatenates the textual data into a single string for each book, giving more weight to the keywords.
    - Checks if a cache file with pre-saved embeddings exists:
        - If the cache file exists, loads and returns the embeddings from that file.
        - If the cache file does not exist, generates the embedding vectors using the BERT model ('all-MiniLM-L6-v2').
    - Saves the generated embeddings in the cache file for future use.

Output:
    - Numpy array containing the BERT embedding for the books
"""

def get_bert_feature_vectors(books, weight=2, cache_key='bert_embeddings', timeout=60 * 60 * 24): # 24 ore
    selected_features = ['genre', 'keywords', 'description', 'author', 'title']
    books = fill_nan_values(books, selected_features)

    books['keywords'] = books['keywords'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    combined = (
        books['keywords'] * weight + ' ' +
        books['description'] + ' ' +
        books['genre'] + ' ' +
        books['author'] + ' ' +
        books['title']
    )
    '''
    The cache should be refreshed in the following cases:
    - A new book is added;
    - A book is modified in any way;
    - A book is removed;
    - The weight in get_bert_feature_vectors is modified;
    - The BERT model used is changed;
    - The structure of selected_features is changed;

    Perhaps by implementing a flag or an auto-versioning system, the refresh could be automated.

    '''

# 📦 Tries cache Redis
    cached = cache.get(cache_key)
    if cached is not None:
        print(f"✅ Embedding loaded from Redis cache (key: '{cache_key}')")
        return pickle.loads(cached)

    print("🔄 Cache miss → generating embeddings with BERT...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(combined.tolist(), convert_to_tensor=False)
    embeddings = np.array(embeddings)  # 🔐 Coerenza formato

    cache.set(cache_key, pickle.dumps(embeddings), timeout=timeout)
    print(f"💾 Embedding saved in Redis (key: '{cache_key}')")

    return embeddings


# ---- Cluster Label ----
"""
Input:
    - cluster_isbns: set of ISBNs belonging to a cluster
    - books: dataframe containing book data (including 'isbn', 'genre', and 'keywords')
    - top_n: maximum number of most frequent keywords to include in the cluster label (default: 3)

Actions:
    - Filters the books in the `books` dataframe that match the ISBNs in the cluster
    - Extracts the genres (`genre`) and keywords (`keywords`) of the books in the cluster
    - Merges the keywords with the genres into a single list
    - Removes non-string or empty items from the list
    - Counts the frequency of words in the resulting list and selects the `top_n` most common ones
    - Concatenates the most frequent words into a comma-separated string

Output:
    - A string containing the most representative keywords of the cluster
    - If no valid keywords are found, returns the string 'Generico'
"""

def get_cluster_label(cluster_isbns, books, top_n=3):
        cluster_books = books[books['isbn'].isin(cluster_isbns)]
        texts = cluster_books['genre'].tolist() + cluster_books['keywords'].explode().tolist()
        texts = [text for text in texts if isinstance(text, str) and text]
        word_counts = Counter(texts)
        common_words = [word for word, _ in word_counts.most_common(top_n)]
        return ', '.join(common_words) if common_words else 'Generico'

# ---- User Embedding ----
"""
Input: 
    - users_histories: dictionary {user_id: list of ISBNs read}
    - isbn_to_book_index: dictionary {ISBN: corresponding index in the dataframe}
    - feature_vectors: numpy array with the vector representations of the books

Actions:
    - For each user in the `users_histories` list:
      - Converts the read ISBNs into corresponding indices in the dataframe using `isbn_to_book_index`
      - If the user has no valid ISBNs, a warning is shown and the user is skipped
      - Calculates the user's embedding by averaging the `feature_vectors` of the books read

Output:
    - user_embeddings: dictionary {user_id: user embedding (average of the books read)}
"""

def get_user_embeddings(users_histories, isbn_to_book_index, feature_vectors):
    user_embeddings = {}
    for user_id, isbn_list in users_histories.items():
        book_df_indices = [isbn_to_book_index[isbn] for isbn in isbn_list if isbn in isbn_to_book_index]
        if not book_df_indices:
            print(f"⚠️ User {user_id} doesn't have valid ISBN.")
            continue
        user_embeddings[user_id] = np.mean(feature_vectors[book_df_indices], axis=0)
    return user_embeddings

# ---- Finds best eps ----
"""
Input:  
    - vectors: numpy array with the feature vectors  
    - eps_range: tuple (min, max) for the eps value in DBSCAN (default: (0.1, 1.0))  
    - step: increment for eps during the search (default: 0.05)  
    - min_samples: minimum number of points required to form a cluster in DBSCAN (default: 2)  
    - n_pca_components: number of PCA components for dimensionality reduction (default: 10)  
    - verbose: if True, prints details about the results for each eps value  

Actions:  
    - Normalizes the input vectors  
    - Applies PCA to reduce the dimensionality to `n_pca_components`  
    - For each `eps` value in the specified range:  
        - Applies DBSCAN with 'cosine' metric  
        - Counts the number of clusters obtained  
        - If there are fewer than 2 clusters, ignores the result  
        - Calculates the silhouette score and updates the best model if the score is higher  
        - If `verbose` is True, prints the details  
    - If no eps value produces at least 2 valid clusters, raises an exception  

Output:  
    - best_labels: array with the cluster labels for the best DBSCAN model found  
"""

# TESTING PARAMETERS: eps_range=(0.1, 1.0), step=0.05, min_samples=4
def find_best_dbscan_eps(vectors, eps_range=(0.1, 1.0), step=0.05, min_samples=2, n_pca_components=10, verbose=False):
    best_score = -1
    best_model = None
    best_labels = None

    vectors_norm = normalize(vectors)
    pca = PCA(n_components=n_pca_components)
    vectors_reduced = pca.fit_transform(vectors_norm)

    for eps in np.arange(*eps_range, step):
        model = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = model.fit_predict(vectors_reduced)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters < 2:
            continue
        try:
            score = silhouette_score(vectors_reduced, labels, metric='cosine')
            if score > best_score:
                best_score = score
                best_model = model
                best_labels = labels  # <--- Saves best labels
            if verbose:
                print(f"✅ eps={eps:.2f} → silhouette={score:.3f}, cluster={n_clusters}")
        except Exception as e:
            print(f"❌ Error silhouette (eps={eps:.2f}): {e}")

    if best_model is None:
        raise ValueError("❌ No eps value ha produced at least 2 valid clusters.")

    return best_labels  

# ---- Clustering users with DBSCAN + auto-eps ----
"""
Input:  
    - user_embeddings: dictionary {user_id: user embedding (numpy array)}  
    - **dbscan_kwargs: additional parameters to pass to `find_best_dbscan_eps`  

Actions:  
    - Extracts the user IDs and their embeddings as numpy arrays  
    - Applies `find_best_dbscan_eps` to find the best eps value for DBSCAN  
    - Assigns the resulting cluster to each user  

Output:  
    - user_clusters: dictionary {user_id: assigned cluster}  
"""

def cluster_users_dbscan_best_eps(user_embeddings, **dbscan_kwargs):
    user_ids = list(user_embeddings.keys())
    vectors = np.array([user_embeddings[uid] for uid in user_ids])

    labels = find_best_dbscan_eps(vectors, **dbscan_kwargs)

    user_clusters = dict(zip(user_ids, labels))

    return user_clusters

# ---- Main func to call ----
"""
Input:  
    - No input parameters (data is loaded from JSON files)  

Actions:  
    - Loads book data (books.json) and customer data (customers.json)  
    - Cleans ISBNs by removing whitespace  
    - Generates book feature vectors using `get_bert_feature_vectors`  
    - Creates a dictionary `isbn_to_book_index` to map ISBNs to dataframe indices  
    - Extracts the reading history of users (users_histories) from the customer dataset  
    - Calculates user embeddings based on the books read using `get_user_embeddings`  
    - Clusters users with `cluster_users_dbscan_best_eps`, optimizing `eps` for DBSCAN  
    - Assigns the resulting clusters to users in the customer dataframe  

    - Uses the `get_cluster_label` function to automatically label clusters:
        - Retrieves the genres and keywords of the books read by users in the cluster  
        - Counts the most frequent words and selects the `top_n` to describe the cluster  

    - Prints the results, assigning labels to clusters:
        - Regular clusters are given a name based on the most common keywords  
        - Outliers (cluster -1) are labeled as "*️⃣ Outlier"  
        - For each cluster, it prints the assigned users with first name, last name, and ID  

Output:  
    - (Currently returns nothing, but could return `user_clusters` if uncommented)  
"""

def cluster_users_from_users_histories():
    # Ensure that the ISBN column is in the correct format (string and stripped of spaces)
    if 'isbn' in books.columns:
        books['isbn'] = books['isbn'].astype(str).str.strip()
    else:
        print("La colonna 'isbn' non esiste nel DataFrame")

    # Generate the feature vectors for the books (you should already have this function implemented)
    feature_vectors = get_bert_feature_vectors(books)

    # Create a mapping from ISBN to DataFrame index for easy lookup
    isbn_to_book_index = {isbn: idx for idx, isbn in enumerate(books['isbn'])}

    # Convert the history column to a list of ISBNs if necessary
    # The assumption here is that the 'history' column could be a string of comma-separated ISBNs
    # If it's already a list, no transformation will occur
    # The resulting dictionary will be {user_id: [isbn1, isbn2, ...]}
    users_histories = customers.set_index('id')['history'].apply(
        lambda x: x.split(',') if isinstance(x, str) else x).to_dict()
    
    # Optional: print the users_histories to debug and ensure it contains the correct data
    print(f"User histories: {users_histories}")

    # Generate embeddings for users based on their book history (you should already have this function implemented)
    user_embeddings = get_user_embeddings(users_histories, isbn_to_book_index, feature_vectors)

    # Cluster the users using DBSCAN (you should already have this function implemented)
    user_clusters = cluster_users_dbscan_best_eps(
        user_embeddings,
        eps_range=(0.05, 1.0),   # Range for the eps parameter
        step=0.05,               # Step size for eps parameter
        min_samples=4,           # Minimum number of samples in a cluster
        n_pca_components=10,     # Number of PCA components to use
        verbose=True             # Verbose logging
    )
    
    # Add the resulting clusters to the customers DataFrame
    customers['cluster'] = customers['id'].map(user_clusters)

    # Print the clustering results with automatic labeling
    for cluster_label in sorted(customers['cluster'].dropna().unique()):
        cluster_users = customers[customers['cluster'] == cluster_label]
        unique_users = cluster_users.drop_duplicates(subset='id')

        if cluster_label == -1:
            cluster_name = f"*️⃣ Outlier"
        else:
            from itertools import chain
            # Recupera gli ISBN unici dei libri letti dagli utenti in questo cluster
            cluster_isbns = set(chain.from_iterable(unique_users['history']))
            cluster_description = get_cluster_label(cluster_isbns, books)
            cluster_name = f"✨ Cluster {cluster_label} - '{cluster_description}'"

        print(f"\n{cluster_name} ({len(unique_users)} users):")
        for _, user in unique_users.iterrows():
            print(f"  - {user['first_name']} {user['last_name']} (id={user['id']})")


    return user_clusters

"""
Input:  
    - user_id: The ID of the user for whom recommended books are generated  
    - top_n: The maximum number of recommended books to return (default: 5)  

Actions:  
    - Performs user clustering using the `cluster_users_from_users_histories` function  
    - Retrieves the cluster of the specified user  
    - If the user does not belong to any valid cluster (cluster -1 or absent), returns an empty list  
    - Extracts the user's reading history (a set of ISBNs read)  
    - Retrieves users from the user's cluster, excluding the user itself  
    - Creates a unified list of ISBNs read by users in the cluster  
    - Selects books that have been read by users in the cluster but not the specified user  
    - Filters the recommended books based on ISBNs present in the books dataframe  
    - Returns the IDs of the first `top_n` recommended books  

Output:  
    - A list of book IDs recommended for the user  
"""

def get_recommendations_for_user(user_id, top_n=5):
    
    clusters = cluster_users_from_users_histories()
    user_cluster = clusters.get(user_id, None)

    if user_cluster is None or user_cluster == -1:
        return []

    user_history = set(customers[customers['id'] == user_id]['history'].values[0])
    cluster_users = customers[(customers['cluster'] == user_cluster) & (customers['id'] != user_id)]
    cluster_history = set(sum(cluster_users['history'], []))
    recommended_books = cluster_history - user_history

    # Get the IDs of the recommended books
    recommended_book_ids = books[books['isbn'].isin(recommended_books)]['id'].head(top_n).tolist()
    
    return recommended_book_ids