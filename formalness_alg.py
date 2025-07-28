import spacy
import pandas as pd
import json
from difflib import SequenceMatcher
import numpy as np

# Load spaCy's small English model
nlp = spacy.load('en_core_web_sm')

def extract_sentence_structure(paragraph):
    """
    Extracts the part-of-speech (POS) tag sequence for each sentence in a paragraph.
    
    Args:
        paragraph (str): Input paragraph as a string.
    
    Returns:
        np.ndarray: A 2D numpy array of POS tag sequences for each sentence in the paragraph.
    """
    doc = nlp(paragraph)
    structures = []
    for sent in doc.sents:
        pos_tags = [token.pos_ for token in sent]  # Extract POS tags for the sentence
        structures.append(pos_tags)
    return np.array(structures, dtype=object)

def compare_structures(structure1, structure2):
    """
    Compare two POS tag sequences using SequenceMatcher to compute a similarity score.
    
    Args:
        structure1 (list): POS tag sequence of the first sentence.
        structure2 (list): POS tag sequence of the second sentence.
    
    Returns:
        float: Similarity score between 0 and 1.
    """
    seq1 = ' '.join(structure1)
    seq2 = ' '.join(structure2)
    return SequenceMatcher(None, seq1, seq2).ratio()

def paragraph_structure_similarity(input_paragraph, reference_paragraph):
    """
    Compute sentence structure similarity between an input paragraph and a reference paragraph.
    
    Args:
        input_paragraph (str): The input paragraph as a string.
        reference_paragraph (str): The reference paragraph as a string.
    
    Returns:
        float: A similarity score between 0 and 1 for sentence structures.
    """
    input_structure = extract_sentence_structure(input_paragraph)
    reference_structure = extract_sentence_structure(reference_paragraph)
    
    # Use broadcasting and vectorized operations for better performance
    similarities = np.zeros((len(input_structure), len(reference_structure)))
    
    for i, input_sent in enumerate(input_structure):
        for j, ref_sent in enumerate(reference_structure):
            similarities[i, j] = compare_structures(input_sent, ref_sent)
    
    best_similarities = np.max(similarities, axis=1)
    average_similarity = np.mean(best_similarities)
    
    return average_similarity

def compute_similarity(reference_paragraph, input_paragraph):
    """
    Wrapper function to compute similarity and scale it.
    
    Args:
        reference_paragraph (str): The reference paragraph.
        input_paragraph (str): The input paragraph.
    
    Returns:
        float: Scaled similarity score between 0 and 10.
    """
    similarity = paragraph_structure_similarity(input_paragraph, reference_paragraph)
    return similarity * 10

def sentence_structure_similarity(input_paragraph, reference_df):
    """
    Compute similarity scores between an input paragraph and each reference paragraph in a DataFrame.
    Return only the highest similarity score.
    
    Args:
        input_paragraph (str): The input paragraph.
        reference_df (pd.DataFrame): A DataFrame where each row contains a reference paragraph.
    
    Returns:
        float: The highest similarity score between 0 and 10.
    """
    input_structure = extract_sentence_structure(input_paragraph)
    
    def process_row(reference_paragraph):
        return compute_similarity(reference_paragraph, input_paragraph)
    
    # Use vectorized operations for efficiency
    scores = np.array([process_row(row) for row in reference_df['Paragraphs']])
    return np.max(scores)

# Load Formal.json data
with open('Formal.json', 'r') as file:
    formal_paragraphs = json.load(file)

# Convert to DataFrame
reference_df = pd.DataFrame({'Paragraphs': formal_paragraphs})

# Example usage
input_paragraph = "The quick brown fox jumps over the lazy dog. This is a common test sentence."
best_similarity_score = sentence_structure_similarity(input_paragraph, reference_df)

# Print the best similarity score
print(f"Highest similarity score: {best_similarity_score:.2f}/10")
