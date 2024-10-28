#without neo4j
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import re

# Stopwords set

stop_words = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", 
    "at", "by", "for", "with", "about", "against", "between", "into", "through", 
    "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", 
    "once", "here", "there", "when", "where", "why", "how", "all", "any", 
    "both", "each", "few", "more", "most", "other", "some", "such", "no", 
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", 
    "t", "can", "will", "just", "don", "should", "now"
])


# Preprocessing and Graph Construction
def clean_data(document):
    # Remove non-alphanumeric characters and lowercasing
    document = re.sub(r'\W+', ' ', document.lower())
    # Split document into words, filter out stopwords
    #print(word for word in document.split() if word not in stop_words)
    #print("hm")
    return ' '.join([word for word in document.split() if word not in stop_words])

def create_word_graph(terms):
    word_graph = {}
    for i in range(len(terms) - 1):
        term1, term2 = terms[i], terms[i + 1]
        if term1 not in word_graph:
            word_graph[term1] = []
        if term2 not in word_graph:
            word_graph[term2] = []
        word_graph[term1].append(term2)
        word_graph[term2].append(term1)
    return word_graph

# PageRank-like algorithm for word importance
def run_pagerank(word_graph, num_iterations=10, d=0.85):
    pagerank = {word: 1 for word in word_graph}
    N = len(word_graph)
    
    for _ in range(num_iterations):
        new_pagerank = {}
        for word, neighbors in word_graph.items():
            rank_sum = sum(pagerank[neighbor] / len(word_graph[neighbor]) for neighbor in neighbors)
            new_pagerank[word] = (1 - d) / N + d * rank_sum
        pagerank = new_pagerank
    
    # Return words ranked by their PageRank scores
    return sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

# Calculate Jaccard similarity between document vectors
def calculate_jaccard_similarity(doc_vectors):
    similarities = []
    for i in range(len(doc_vectors)):
        for j in range(i + 1, len(doc_vectors)):
            similarity = jaccard_score(doc_vectors[i], doc_vectors[j], average='macro')
            similarities.append((i, j, similarity))
    return similarities

# Create similarity matrix for documents
def create_document_similarity_matrix(similarities, num_docs):
    similarity_matrix = np.zeros((num_docs, num_docs))
    for doc1, doc2, score in similarities:
        similarity_matrix[doc1][doc2] = score
        similarity_matrix[doc2][doc1] = score
    return similarity_matrix

# KMeans clustering as a substitute for Louvain
def form_communities_of_similar_documents(similarity_matrix, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    communities = kmeans.fit_predict(similarity_matrix)
    return communities

# Main Graph of Docs Function
def graph_of_docs(dataset):
    word_graph = {}
    document_labels = list(dataset.keys())
    vectorizer = CountVectorizer()
    doc_vectors = vectorizer.fit_transform(dataset.values()).toarray()

    # Create word-document graph and compute word importance
    for document_label, document in dataset.items():
        document = clean_data(document)
        terms = document.split()
        doc_word_graph = create_word_graph(terms)
        word_graph.update(doc_word_graph)

    ranked_words = run_pagerank(word_graph)
    print("Top words by PageRank:\n", ranked_words[:10])

    # Calculate document similarity
    similarities = calculate_jaccard_similarity(doc_vectors)
    similarity_matrix = create_document_similarity_matrix(similarities, len(doc_vectors))

    # Form communities of similar documents
    communities = form_communities_of_similar_documents(similarity_matrix, num_clusters=5)

    return communities, similarity_matrix

# Load the 20 Newsgroups dataset
def load_20newsgroups_dataset():
    from sklearn.datasets import fetch_20newsgroups

    # Fetch the training and test subsets
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    # Combine into dictionaries: label -> document
    train_data = dict(zip(newsgroups_train.target, newsgroups_train.data))
    test_data = dict(zip(newsgroups_test.target, newsgroups_test.data))

    return train_data, test_data

# Main Execution
def main():
    # Load the 20 Newsgroups dataset (both train and test sets)
    train_dataset, test_dataset = load_20newsgroups_dataset()

    # Encode labels for training and testing datasets
    train_labels = list(train_dataset.keys())
    test_labels = list(test_dataset.keys())
    
    le = LabelEncoder()
    y_train_labels = le.fit_transform(train_labels)
    y_test_labels = le.transform(test_labels)

    # Train a classifier using the similarity matrix of the training set
    communities, similarity_matrix_train = graph_of_docs(train_dataset)

    # Prepare the test document vectors (this will be needed for testing later)
    vectorizer = CountVectorizer()
    doc_vectors_test = vectorizer.fit_transform(test_dataset.values()).toarray()

    # Calculate Jaccard similarity for test documents against the training similarity matrix
    similarities_test = calculate_jaccard_similarity(doc_vectors_test)
    similarity_matrix_test = create_document_similarity_matrix(similarities_test, len(doc_vectors_test))

    # Train KNN Classifier using training data similarity matrix
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(similarity_matrix_train, y_train_labels)

    # Predict on the test set
    y_pred = knn.predict(similarity_matrix_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test_labels, y_pred)
    print(y_test_labels)
    print(y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

# Execute the code
main()
