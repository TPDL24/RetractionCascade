import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Sentence-BERT (SBERT) model
sbert_model = SentenceTransformer('stsb-roberta-base')

# Function to compute Jaccard similarity between author names and cited references
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# Function to compute cosine similarity between abstracts or citation sentences
def cosine_similarities(sentence_embeddings1, sentence_embeddings2):
    similarities = cosine_similarity(sentence_embeddings1, sentence_embeddings2)
    return np.mean(similarities)

# Define weights for different features
weights = {
    'author_names': 0.3,
    'cited_references': 0.3,
    'common_references': 0.3,
    'abstracts': 0.1
}

# Function to compute weighted sum of similarities
def compute_weighted_sum(similarities):
    weighted_sum = 0
    for feature, weight in weights.items():
        weighted_sum += weight * similarities[feature]
    return weighted_sum

# Function to normalize final score
def normalize_score(score):
    return score / sum(weights.values())

# Read data from CSV file
data = pd.read_csv('your_file.csv')  # Replace 'your_file.csv' with the actual file path

# Example data (replace with actual data)
author_names1 = set(data['Author Names'][0].split(','))
author_names2 = set(data['Author Names'][1].split(','))
cited_references1 = set(data['Cited References'][0].split(','))
cited_references2 = set(data['Cited References'][1].split(','))
common_references = set(data['Common References'][0].split(','))
abstract1 = data['Abstract'][0]
abstract2 = data['Abstract'][1]

citation_sentence1 = data['Citation Sentence'][0]
citation_sentence2 = data['Citation Sentence'][1]

# Compute Jaccard similarity for author names and cited references
author_similarity = jaccard_similarity(author_names1, author_names2)
cited_references_similarity = jaccard_similarity(cited_references1, cited_references2)

# Convert abstracts and citation sentences to sentence embeddings
abstract_embeddings1 = sbert_model.encode([abstract1])
abstract_embeddings2 = sbert_model.encode([abstract2])
citation_sentence_embeddings1 = sbert_model.encode([citation_sentence1])
citation_sentence_embeddings2 = sbert_model.encode([citation_sentence2])

# Compute cosine similarity for abstracts and citation sentences
abstract_similarity = cosine_similarities(abstract_embeddings1, abstract_embeddings2)
citation_sentence_similarity = cosine_similarities(citation_sentence_embeddings1, citation_sentence_embeddings2)

# Compute weighted sum of similarities
similarities = {
    'author_names': author_similarity,
    'cited_references': cited_references_similarity,
    'common_references': cosine_similarity([list(common_references)], [list(common_references)])[0][0], # Assuming cosine similarity for common references
    'abstracts': abstract_similarity,
    'citation_sentences': citation_sentence_similarity
}

weighted_sum = compute_weighted_sum(similarities)

# Normalize final score
final_score = normalize_score(weighted_sum)

print("Final Score:", final_score)
