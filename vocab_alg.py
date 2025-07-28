import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def vocabulary_similarity(essay, reference_df, vectorizer=None):
    if vectorizer is None:
        vectorizer = CountVectorizer()
    reference_docs = reference_df.squeeze().tolist()
    combined_texts = [essay] + reference_docs
    word_freq_matrix = vectorizer.fit_transform(combined_texts)
    essay_vector = word_freq_matrix[0]
    reference_vectors = word_freq_matrix[1:]
    similarity_scores = cosine_similarity(essay_vector, reference_vectors).flatten()
    return [(idx, score) for idx, score in enumerate(similarity_scores)]
essay = "The quick brown fox jumps over the lazy dog. This is a test sentence."
reference_data = np.array([
    "The fast brown fox leaps over a lazy dog. This sentence is for testing.",
    "The sun shines brightly in the morning. A completely unrelated sentence.",
    "Foxes are quick and dogs are lazy in this simple sentence."
])
reference_df = pd.DataFrame(reference_data, columns=["Text"])
similarity_scores = vocabulary_similarity(essay, reference_df)
for idx, score in similarity_scores:
    print(f"Similarity to reference document {idx + 1}: {score:.2f}")
