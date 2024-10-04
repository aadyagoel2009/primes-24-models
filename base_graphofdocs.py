import pandas as pd
import numpy as np
import re
import networkx as nx
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Stop words set
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

# Preprocessing Function
def clean_data(document):
    document = re.sub(r'\W+', ' ', document.lower())
    return ' '.join([word for word in document.split() if word not in stop_words])

# Word Graph Creation
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

# PageRank-like algorithm using networkx
def run_pagerank(word_graph):
    G = nx.Graph()
    
    # Adding edges to the graph
    for word, neighbors in word_graph.items():
        for neighbor in neighbors:
            G.add_edge(word, neighbor)
    
    # Run PageRank using networkx's built-in function
    pagerank = nx.pagerank(G)
    
    # Return sorted words by PageRank score
    return sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

# Build Word-Document Graph
def build_word_document_graph(train_dataset, top_n_words=100):
    word_graph = defaultdict(list)
    doc_graph = defaultdict(list)
    all_word_importances = defaultdict(dict)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words=stop_words, ngram_range=(1, 2))
    doc_vectors = vectorizer.fit_transform(train_dataset.values()).toarray()

    # Build word-document graph
    for doc_id, document in train_dataset.items():
        clean_doc = clean_data(document)
        words = clean_doc.split()
        doc_word_graph = create_word_graph(words)

        # Run pagerank to find important words in the document
        ranked_words = run_pagerank(doc_word_graph)[:top_n_words]
        important_words = [word for word, _ in ranked_words]
        
        # Add important words as nodes and connect them to their document
        for word in important_words:
            word_graph[word].append(doc_id)
            doc_graph[doc_id].append(word)

            # Keep track of importance of word for this doc_id
            all_word_importances[word][doc_id] = all_word_importances[word].get(doc_id, 0) + 1

    return word_graph, doc_graph, all_word_importances

# Create Class Importance Nodes
def create_class_importance_nodes(y_train_labels, word_graph):
    # Dictionary where each word has a list of importance values for each class
    class_importances = {word: [0] * len(set(y_train_labels)) for word in word_graph.keys()}

    # Update class importances based on frequency of word in documents of each class
    for word, doc_ids in word_graph.items():
        for doc_id in doc_ids:
            class_label = y_train_labels[doc_id]
            class_importances[word][class_label] += 1

    return class_importances

# SVM Classifier Training
def train_classifier(X_train, y_train):
    svm = SVC(kernel='linear', C=1, random_state=42)
    svm.fit(X_train, y_train)
    return svm

# Load the 20 Newsgroups dataset
def load_20newsgroups_dataset():
    from sklearn.datasets import fetch_20newsgroups

    # Fetch the dataset
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    # Create train and test dictionaries with labels and documents
    train_documents = newsgroups_train.data
    train_labels = newsgroups_train.target
    test_documents = newsgroups_test.data
    test_labels = newsgroups_test.target

    train_dataset = dict(zip(range(len(train_labels)), train_documents))
    test_dataset = dict(zip(range(len(test_labels)), test_documents))

    return train_dataset, test_dataset, train_labels, test_labels

# Main Execution
def main():
    # Load the dataset (train and test split)
    train_dataset, test_dataset, y_train_labels, y_test_labels = load_20newsgroups_dataset()

    # Build the word-document graph
    word_graph, doc_graph, word_importances = build_word_document_graph(train_dataset)

    # Create class importance nodes
    class_importances = create_class_importance_nodes(y_train_labels, word_graph)

    # Prepare test set vectors
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=stop_words, ngram_range=(1, 2))
    doc_vectors_test = vectorizer.fit_transform(test_dataset.values()).toarray()

    # Train SVM Classifier using the word-document graph (using doc_vectors)
    svm = train_classifier(doc_vectors_test, y_test_labels)

    # Predict on the test set
    y_pred = svm.predict(doc_vectors_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test_labels, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

main()
