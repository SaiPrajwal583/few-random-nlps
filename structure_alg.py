from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer, util
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
# Initialize the TF-IDF vectorizer globally
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 most frequent words
reference_texts = None  # Global variable to store reference texts
print("TF-IDF Vectorizer initialized successfully.")

def get_sbert_embeddings(sentences):
    """
    Compute SBERT embeddings for a list of sentences.
    """
    try:
        embeddings = sbert_model.encode(sentences,convert_to_tensor=True,normalize_embeddings=True)
    except Exception as e:
        print(f"Error in get_sbert_embeddings: {str(e)}")
        return None

def scale_similarity_stricter(cosine_similarity_score):
    """
    Scale cosine similarity from [-1, 1] to [0, 10], but apply a stricter non-linear transformation.
    This compresses lower scores more than higher ones.
    """
    try:
        # Scale the cosine similarity to a [0, 10] range using a stricter non-linear transformation
        adjusted_score = (cosine_similarity_score + 1) ** 2  # Squaring makes the scoring stricter
        return 5 * adjusted_score  # Maintain the scale in the [0, 10] range
    except Exception as e:
        print(f"Error in scaling similarity score: {str(e)}")
        return None

def compute_essay_similarity_to_reference_texts_stricter(essay):
    """
    Compute similarity between an essay and the most similar document from globally loaded reference texts using SBERT,
    with stricter scoring.
    """
    global reference_texts  # Use the globally loaded reference texts

    try:
        if reference_texts is None:
            raise ValueError("Reference texts not loaded. Please load them before computing similarity.")

        # Compute SBERT embeddings for the essay
        try:
            essay_embedding = sbert_model.encode([essay], convert_to_tensor=True, normalize_embeddings=True)[0]
            print("SBERT embedding computed for essay.")
        except Exception as e:
            print(f"Error computing essay embedding: {str(e)}")
            return None

        # Compute SBERT embeddings for the reference texts
        try:
            reference_embeddings = sbert_model.encode(reference_texts, convert_to_tensor=True, normalize_embeddings=True)
            print("SBERT embeddings computed for reference texts.")
        except Exception as e:
            print(f"Error computing reference embeddings: {str(e)}")
            return None

        # Compute cosine similarity
        try:
            similarity_scores = util.cos_sim(essay_embedding, reference_embeddings)[0].cpu().numpy()
            highest_score = float(np.max(similarity_scores))
            scaled_score = scale_similarity_stricter(highest_score)
            if scaled_score is None:
                raise ValueError("Failed to scale similarity score.")
        except Exception as e:
            print(f"Error computing cosine similarity: {str(e)}")
            return None

        return round(scaled_score, 2)

    except Exception as e:
        print(f"Unexpected error in compute_essay_similarity_to_reference_texts_stricter: {str(e)}")
        return None
def load_bookcorpus_dataset(num_documents=10):
    """
    Load a subset of the BookCorpus dataset and store it globally.
    """
    global reference_texts  # Store reference texts in a global variable
    try:
        dataset = load_dataset('bookcorpus', split=f'train[:{num_documents}]')
        reference_texts = dataset['text'][:num_documents]
        print(f"Dataset loaded successfully: {len(reference_texts)} documents.")
    except Exception as e:
        print(f"Error loading BookCorpus dataset: {str(e)}")
        reference_texts = None


# Example usage
if __name__ == "__main__":
    try:
        # Example essay for comparison
        essay = "The quick brown fox jumps over the lazy dog. It is a commonly used sentence for testing."

        # Load the dataset globally
        load_bookcorpus_dataset(num_documents=10)

        # Compute the highest similarity score using stricter scoring
        highest_similarity_score = compute_essay_similarity_to_reference_texts_stricter(essay)

        if highest_similarity_score is not None:
            print(f"Highest similarity score (stricter): {highest_similarity_score:.2f}")
        else:
            print("Error in similarity computation.")
    
    except Exception as e:
        print(f"Error in the main execution: {str(e)}")
