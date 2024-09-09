from neo4j import GraphDatabase
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Neo4j Database Connection
uri = "bolt://localhost:7687" 
user = "neo4j"
password = "primesproject"

driver = GraphDatabase.driver(uri, auth=(user, password))

def preprocess_text(text):
    # Simple preprocessing: remove punctuation and stopwords
    # For simplicity, we'll use a basic preprocessing approach
    return ' '.join(word.lower() for word in text.split() if word.isalnum())

def create_graph_of_words(terms, document_label, database):
    with database.session() as session:
        for term in terms:
            session.run("MERGE (w:Word {name: $term})", term=term)
        for i, term1 in enumerate(terms):
            for term2 in terms[i+1:]:
                session.run("""
                    MERGE (w1:Word {name: $term1})
                    MERGE (w2:Word {name: $term2})
                    MERGE (w1)-[:CONNECTS]-(w2)
                """, term1=term1, term2=term2)
        session.run("""
            MERGE (d:Document {label: $document_label})
            WITH d
            UNWIND $terms AS term
            MATCH (w:Word {name: term})
            MERGE (d)-[:INCLUDES]->(w)
        """, document_label=document_label, terms=terms)

def run_pagerank(database):
    with database.session() as session:
        result = session.run("""
            CALL algo.pageRank.stream('Word', 'CONNECTS', {graph: 'cypher', write: true})
            YIELD nodeId, score
            RETURN algo.getNodeById(nodeId).name AS word, score
            ORDER BY score DESC
        """)
        return list(result)

def create_document_similarity_subgraph(database):
    with database.session() as session:
        result = session.run("""
            MATCH (d1:Document)-[:INCLUDES]->(w:Word)<-[:INCLUDES]-(d2:Document)
            WITH d1, d2, COUNT(w) AS common_words
            WHERE d1 <> d2
            MERGE (d1)-[r:SIMILAR]->(d2)
            SET r.score = common_words
        """)
        return result.summary().counters

def form_communities_of_similar_documents(database):
    with database.session() as session:
        result = session.run("""
            CALL algo.louvain.stream('Document', 'SIMILAR', {graph: 'cypher', write: true})
            YIELD nodeId, community
            RETURN algo.getNodeById(nodeId).label AS document, community
            ORDER BY community
        """)
        return pd.DataFrame(list(result), columns=['document', 'community'])

def graph_of_docs():
    dataset = pd.read_csv('20_newsgroups.csv')  # Load your dataset
    with driver.session() as session:
        for _, row in dataset.iterrows():
            document_label = row['label']
            document = preprocess_text(row['text'])
            terms = document.split()
            create_graph_of_words(terms, document_label, session)
        
        run_pagerank(session)
        create_document_similarity_subgraph(session)
        communities = form_communities_of_similar_documents(session)
        
        print("Communities of documents:\n", communities)
        
        # Implement classification using communities and other methods as described
        # Placeholder for text categorization
        # Further code for text classification goes here...

graph_of_docs()

driver.close()