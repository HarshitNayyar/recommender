# Recommender System V3

## Ibridizzazione del filtering

Il filtering delle raccomandazioni dei libri, precedentemente basato unicamente sulla strategia di content-based filtering, è stato ibridizzato con l'aggiunta di un modulo di collaborative filtering.

Per attuare l'ibridizzazione avevamo due strategie a disposizione:

### Ibridizzazione sequenziale
L'ibridizzazione sequenziale prevede che il modulo **Collaborative** Filtering faccia una **prima selezione** di libri a seconda dei profili utenti e successivamente il modulo **Content-Based** filtering filtra questa lista, producendo la lista di raccomandazioni pertinenti.

### Ibridizzazione parallela
L'ibridizzazione parallela prevede l'implementazione di **due moduli**, uno **Content-Based** Filtering, uno **Collaborative** Filtering, che entrambi producono una lista di raccomandazioni. Entrambe le liste verranno poi passate in un algoritmo di **Rank Aggregation** per combinare una lista di raccomandazioni pertinenti.

Abbiamo scelto questa strategia di ibridizzazione poiché più semplice da implementare inizialmente, successivamente, se sarà necessario passeremo alla stategia sequenziale.

#### Aggiunta del modulo collaborative filtering
Per l'aggiunta del modulo collaborative abbiamo scelto due strategie diverse per due necessità diverse:
- Popolarità globale: basata sul **numero totale di prenotazioni del libro** (popolarità assoluta). Utilizzata come fallback in caso di **cold start** (utente senza libri prenotati).
- Similarità tra utenti: basata sulla **similità tra utenti**. Usato per le raccomandazioni basate sul comportamento di utenti simili.

## Integrazione con progetto Biblioteca

### Aggiunta servizio JMS per Book, Reservation e Customer
Abbiamo aggiunto vari servizi JMS che al momento dell'aggiunta, della modifica o dell'eliminazione di un Book, Reservation o Customer, mandano un messaggio appropriato alla coda del microservizio di raccomandazione.
Questi messaggi vengono poi gestiti in maniera appropriata da eventi e RESTful API in **Django**.

### Aggiunta di RESTful API in Django

## Integrazione Database
Abbiamo integrato il microservizio di raccomandazione con un database MySQL dockerizzato che gestisce la permanenza dei dati.

## Core Recommender

### Linea guida del sistema di raccomandazione e clustering utenti

---

#### 1. Estrazione dati  
**Da**: Database Django (`Book`, `Customer`)  
**Verso**: DataFrame Pandas

```python
base_dir = os.path.dirname(os.path.abspath(__file__))
cache_path = os.path.join(base_dir, 'cache', 'bert_embeddings.npy')
```

---

#### 2. Preprocessing  
Preparazione dei dati per l'embedding:

- Pulizia dei valori `NaN` (`fill_nan_values`)
- Conversione della cronologia (`history`) da ID a ISBN
- Preparazione del testo per l'embedding (concatenazione con pesi)

---

#### 3. Features embedding  
Conversione del catalogo di libri in vettori embeddings:

```python
get_bert_feature_vectors(books)
```

- Generazione degli embedding con BERT (`sentence-transformers`)
- Cache locale (`.npy`) per evitare rigenerazioni se i dati rimangono invariati

---

#### 4. User embedding  
Prende la lista degli utenti associata alla propria history, un dizionario ISBN libro associato all'indice del DataFrame dei libri e un array numpy con gli embeddings semantici dei libri, per produrre un **embedding semantico unico che rappresenta i gusti dell'utente**. 

```python
get_user_embeddings(users_histories, isbn_to_book_index, feature_vectors)
# isbn_to_book_index: dizionario {ISBN: indice corrispondente nel dataframe}
```

- Calcolo della media degli embedding dei libri letti da ciascun utente

---

#### 5. Clustering utenti  
Raggruppamento degli utenti in base a gusti simili (embedding):

```python
cluster_users_dbscan_best_eps(user_embeddings, **dbscan_kwargs)
```

- **Riduzione dimensionale** con PCA (Principal Component Analysis):
  - La PCA è una tecnica di riduzione della dimensionalità che trasforma un insieme di variabili correlate in un nuovo set di variabili non correlate, chiamate **componenti principali**, che sono le direzioni lungo le quali i dati variano di più, selezionate in ordine decrescente di varianza, riducendo così il numero di dimensioni mantenendo il più possibile l'informazione. Molto utilizzata per semplificare i dati e migliorare le prestazioni di modelli di machine learning.
  - In questo contesto serve a ridurre le dimensioni dell’output BERT (MiniLM) da 384 a 10, migliorando l’efficienza e l'efficacia del clustering.
- Ricerca automatica del valore ottimale di `eps` per DBSCAN (`find_best_dbscan_eps`)
- Assegnazione degli utenti ai cluster (inclusi outlier con etichetta `-1`)

Nota: `**dbscan_kwargs` è un dizionario di parametri passati dinamicamente alla funzione.

---

#### 6. Etichettatura dei cluster  
Attribuzione di un'etichetta descrittiva ai cluster (es. "Fantasy, Avventura"):

```python
get_cluster_label(cluster_isbns, books, top_n=3)
```

- Estrazione dei generi e parole chiave più frequenti
- Generazione del nome rappresentativo per ogni cluster

---

#### 7. Visualizzazione risultati (debug)  
- Stampa a console degli utenti per cluster (nome, cognome, ID), inclusi gli outlier

---

#### 8. Raccomandazione personalizzata (core)  
Suggerimento di libri a utenti con gusti simili (approccio collaborativo):

```python
get_recommendations_for_user(user_id, top_n=5)
```

- Recupero del cluster di appartenenza dell'utente
- Esclusione dei libri già letti
- Suggerimento dei libri più popolari nel cluster

---

#### Output  
- Lista di `book_ids` consigliati


```sh
pip install -r requirements.txt     --extra-index-url https://download.pytorch.org/whl/cpu
```