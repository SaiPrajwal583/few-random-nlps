import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def vocabulary_similarity(essay, reference_df, vectorizer=None):
    """
    Compute vocabulary similarity between an essay and a reference database.
    
    Args:
        essay (str): The input essay as a string.
        reference_df (pd.DataFrame): DataFrame with reference documents (each document in a row or column).
        vectorizer (CountVectorizer): Optional pre-initialized vectorizer for word frequency.

    Returns:
        list: A list of tuples containing document index and similarity score.
    """
    if vectorizer is None:
        # Initialize a CountVectorizer to convert text to word frequency vectors
        vectorizer = CountVectorizer()

    # Convert the reference documents from the DataFrame into a list
    reference_docs = reference_df.squeeze().tolist()

    # Combine the essay and the reference documents for vectorization
    combined_texts = [essay] + reference_docs
    
    # Create the word frequency matrix (rows: documents, columns: vocabulary)
    word_freq_matrix = vectorizer.fit_transform(combined_texts)

    # Split the essay vector from the reference document vectors
    essay_vector = word_freq_matrix[0]  # First row is the essay
    reference_vectors = word_freq_matrix[1:]  # Remaining rows are reference documents

    # Compute cosine similarity between the essay and each document
    similarity_scores = cosine_similarity(essay_vector, reference_vectors).flatten()

    # Return the similarity scores as a list of (index, score) pairs
    return [(idx, score) for idx, score in enumerate(similarity_scores)]

# Example usage:
# Sample essay
essay = "The quick brown fox jumps over the lazy dog. This is a test sentence."

# Sample reference database in the form of a NumPy DataFrame
reference_data = np.array([
    "The fast brown fox leaps over a lazy dog. This sentence is for testing.",
    "The sun shines brightly in the morning. A completely unrelated sentence.",
    "Foxes are quick and dogs are lazy in this simple sentence."
])

# Convert to DataFrame
reference_df = pd.DataFrame(reference_data, columns=["Text"])

# Compute vocabulary similarity
similarity_scores = vocabulary_similarity(essay, reference_df)

# Print similarity scores
for idx, score in similarity_scores:
    print(f"Similarity to reference document {idx + 1}: {score:.2f}")
