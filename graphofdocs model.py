#change the password on line 112, and have graph data science library installed in neo4j
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Neo4j Database Connection
class Neo4jHandler:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            return session.run(query, parameters)

# Preprocessing and Graph Construction
def clean_data(document):
    # Simple cleaning: lowercasing and removing non-alphanumeric characters
    return ' '.join(word.lower() for word in document.split() if word.isalnum())

def create_graph_of_words(terms, document_label, neo4j_handler):
    for i in range(len(terms) - 1):
        term1, term2 = terms[i], terms[i + 1]
        query = (
            "MERGE (w1:Word {text: $term1}) "
            "MERGE (w2:Word {text: $term2}) "
            "MERGE (w1)-[:CONNECTS]->(w2) "
            "MERGE (d:Document {label: $document_label}) "
            "MERGE (w1)-[:INCLUDES]->(d) "
            "MERGE (w2)-[:INCLUDES]->(d)"
        )
        neo4j_handler.run_query(query, parameters={"term1": term1, "term2": term2, "document_label": document_label})

def run_pagerank(neo4j_handler):
    # PageRank to identify the most important nodes
    query = (
        "CALL gds.pageRank.stream('Word', {relationshipQuery: 'CONNECTS'}) "
        "YIELD nodeId, score "
        "RETURN gds.util.asNode(nodeId).text AS word, score "
        "ORDER BY score DESC"
    )
    return list(neo4j_handler.run_query(query))

def calculate_jaccard_similarity(doc_vectors):
    similarities = []
    for i in range(len(doc_vectors)):
        for j in range(i + 1, len(doc_vectors)):
            similarity = jaccard_score(doc_vectors[i], doc_vectors[j], average='macro')
            similarities.append((i, j, similarity))
    return similarities

def create_document_similarity_subgraph(similarities, neo4j_handler):
    for doc1, doc2, score in similarities:
        query = (
            "MATCH (d1:Document {label: $doc1_label}), (d2:Document {label: $doc2_label}) "
            "MERGE (d1)-[:IS_SIMILAR {score: $score}]->(d2)"
        )
        neo4j_handler.run_query(query, parameters={"doc1_label": doc1, "doc2_label": doc2, "score": score})

def form_communities_of_similar_documents(neo4j_handler):
    # Use Louvain algorithm in Neo4j
    query = (
        "CALL gds.louvain.stream({nodeProjection: 'Document', relationshipProjection: 'IS_SIMILAR'}) "
        "YIELD nodeId, community "
        "RETURN gds.util.asNode(nodeId).label AS document, community "
        "ORDER BY community"
    )
    result = neo4j_handler.run_query(query)
    return pd.DataFrame(result.data(), columns=['document', 'community'])

# Main Graph of Docs Function
def graph_of_docs(neo4j_handler, dataset):
    for document_label, document in dataset.items():
        document = clean_data(document)
        terms = document.split()  # Tokenize text into words
        create_graph_of_words(terms, document_label, neo4j_handler)

    # Run PageRank
    run_pagerank(neo4j_handler)

    # Vectorize the documents
    document_labels = list(dataset.keys())
    vectorizer = CountVectorizer()
    doc_vectors = vectorizer.fit_transform(dataset.values()).toarray()

    # Calculate similarities and create subgraph
    similarities = calculate_jaccard_similarity(doc_vectors)
    create_document_similarity_subgraph(similarities, neo4j_handler)

    # Form document communities
    communities = form_communities_of_similar_documents(neo4j_handler)

    return communities

# Load the dataset
def load_dataset():
    from sklearn.datasets import fetch_20newsgroups

    # Fetch the dataset
    newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    # Create a dictionary with labels and documents
    documents = newsgroups_data.data
    labels = newsgroups_data.target
    return dict(zip(labels, documents))

# Main Execution
def main():
    neo4j_handler = Neo4jHandler("bolt://localhost:7687", "neo4j", "password")
    
    # Load dataset
    dataset = load_dataset() 

    # Create Graph of Docs and communities
    communities = graph_of_docs(neo4j_handler, dataset)

    # Train a classifier
    le = LabelEncoder()
    y = le.fit_transform(list(dataset.keys()))  # Encode document labels
    X_train, X_test, y_train, y_test = train_test_split(communities['document'], y, test_size=0.3, random_state=42)

    # Train KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train.values.reshape(-1, 1), y_train)
    y_pred = knn.predict(X_test.values.reshape(-1, 1))

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    neo4j_handler.close()

if __name__ == "__main__":
    main()
