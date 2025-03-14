import os 
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import math
import json 
from utils import process_text, parse_doc, save_json, load_json, download_nltk
import re
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords




class Indexer:
    def __init__(self, xml_file, output_dir):
        self.xml_file = xml_file
        self.output_dir = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        
        self.documents = {}
        self.inverted_index = defaultdict(dict)  # Changed to dict for tf-idf scores
        self.doc_lengths = {}
        self.term_frequencies = {}
        self.document_frequencies = {}
        self.total_docs = 0
        self.total_terms = 0
        self.avg_doc_length = 1  # Initialize to 1 instead of 0
        self.idf_scores = {}
        self.doc_vectors = {}  # For document vectors
        self.doc_count = 0
        self.k1 = 1.5
        self.b = 0.75

    
    def process_docs(self):
        print("Starting to process the documents...")

        self.documents, temp_inverted_index = parse_doc(self.xml_file)
        self.total_docs = len(self.documents)

        # Calculate document lengths and term frequencies
        for doc_id, term_counts in self.documents.items():
            doc_length = sum(term_counts.values())
            self.doc_lengths[doc_id] = doc_length
            self.total_terms += doc_length
            self.term_frequencies[doc_id] = term_counts

        # Calculate IDF scores and build inverted index with weights
        for term, doc_ids in temp_inverted_index.items():
            doc_freq = len(doc_ids)
            self.document_frequencies[term] = doc_freq
            
            # Calculate IDF score
            self.idf_scores[term] = math.log((self.total_docs + 1) / (doc_freq + 0.5))

            # Calculate BM25-style weights for each term-document pair
            for doc_id in doc_ids:
                tf = self.term_frequencies[doc_id].get(term, 0)
                doc_len = self.doc_lengths[doc_id]
                
                # BM25 term weight calculation
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                weight = (numerator / denominator) * self.idf_scores[term]
                
                self.inverted_index[term][doc_id] = weight

        # Calculate average document length
        total_length = sum(self.doc_lengths.values())
        num_docs = len(self.doc_lengths)
        self.avg_doc_length = total_length / num_docs if num_docs > 0 else 1

        # Calculate document vectors using the weighted terms
        for doc_id in self.documents:
            doc_vector = defaultdict(float)
            doc_terms = self.term_frequencies[doc_id]
            
            for term, freq in doc_terms.items():
                if term in self.inverted_index:
                    doc_vector[term] = self.inverted_index[term].get(doc_id, 0)
            
            # Normalize the document vector
            magnitude = math.sqrt(sum(score ** 2 for score in doc_vector.values()))
            if magnitude > 0:
                doc_vector = {term: score/magnitude for term, score in doc_vector.items()}
            
            self.doc_vectors[doc_id] = dict(doc_vector)

        print("Finished processing the documents...")
        print(f"Total number of documents: {self.total_docs}")
        print(f"Total number of unique terms: {len(self.inverted_index)}")
        print(f"Average document length: {self.avg_doc_length:.2f}")

    def save_index(self):
        print("Saving the index...")
    
        save_json(self.documents, os.path.join(self.output_dir, "documents.json"))
        
        # Save all the computed data structures
        save_json(dict(self.inverted_index), os.path.join(self.output_dir, "inverted_index.json"))
        
        save_json(self.doc_lengths, os.path.join(self.output_dir, "doc_lengths.json"))
        
        save_json(self.term_frequencies, os.path.join(self.output_dir, "term_frequencies.json"))
        
        save_json(dict(self.document_frequencies), os.path.join(self.output_dir, "document_frequencies.json"))
        save_json(self.idf_scores, os.path.join(self.output_dir, "idf_scores.json"))
        save_json(self.doc_vectors, os.path.join(self.output_dir, "doc_vectors.json"))

        metadata = {
                "total_docs": self.total_docs,
                "total_terms": self.total_terms,
                "avg_doc_length": self.avg_doc_length
            }

        save_json(metadata, os.path.join(self.output_dir, "metadata.json"))

        print("Finished saving the index...")
        
    def build_index(self):
        self.process_docs()
        self.calculate_avg_doc_length()  # Calculate average before BM25
        self.calculate_bm25_scores()
        self.save_index()
        return self 
    
    def calculate_avg_doc_length(self):
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        else:
            self.avg_doc_length = 1  # Fallback value if no documents
        print(f"Average document length: {self.avg_doc_length}")

    def calculate_bm25_scores(self):
        for term in self.inverted_index:
            for doc_id in self.inverted_index[term]:
                tf = self.inverted_index[term][doc_id]
                doc_len = self.doc_lengths.get(doc_id, 0)
                
                # Safe division with fallback
                length_ratio = doc_len / self.avg_doc_length if self.avg_doc_length != 0 else 1
                denominator = tf + self.k1 * (1 - self.b + self.b * length_ratio)
                
                if denominator != 0:
                    self.inverted_index[term][doc_id] = tf / denominator
                else:
                    self.inverted_index[term][doc_id] = 0

def process_document(doc_text):
    """Enhanced document processing"""
    # Convert to lowercase
    text = doc_text.lower()
    
    # Remove special characters but keep meaningful ones
    text = re.sub(r'[^a-z0-9\-\']', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and stem
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    # Process tokens
    processed_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 1:
            stemmed = stemmer.stem(token)
            if stemmed:
                processed_tokens.append(stemmed)
    
    return processed_tokens

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    xml_path = os.path.join(data_dir, "cran.all.1400.xml")
    output_dir = os.path.join(data_dir, "index")
    
    indexer = Indexer(xml_path, output_dir)
    indexer.build_index()
    
    

