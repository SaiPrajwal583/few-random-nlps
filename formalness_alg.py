import spacy
import pandas as pd
import json
from difflib import SequenceMatcher
import numpy as np
nlp = spacy.load('en_core_web_sm')
def extract_sentence_structure(paragraph):
    doc = nlp(paragraph)
    structures = []
    for sent in doc.sents:
        pos_tags = [token.pos_ for token in sent]
        structures.append(pos_tags)
    return np.array(structures, dtype=object)
def compare_structures(structure1, structure2):
    seq1 = ' '.join(structure1)
    seq2 = ' '.join(structure2)
    return SequenceMatcher(None, seq1, seq2).ratio()
def paragraph_structure_similarity(input_paragraph, reference_paragraph):
    input_structure = extract_sentence_structure(input_paragraph)
    reference_structure = extract_sentence_structure(reference_paragraph)
    similarities = np.zeros((len(input_structure), len(reference_structure)))
    for i, input_sent in enumerate(input_structure):
        for j, ref_sent in enumerate(reference_structure):
            similarities[i, j] = compare_structures(input_sent, ref_sent)
    best_similarities = np.max(similarities, axis=1)
    average_similarity = np.mean(best_similarities)
    return average_similarity
def compute_similarity(reference_paragraph, input_paragraph):
    similarity = paragraph_structure_similarity(input_paragraph, reference_paragraph)
    return similarity * 10
def sentence_structure_similarity(input_paragraph, reference_df):
    input_structure = extract_sentence_structure(input_paragraph)
    def process_row(reference_paragraph):
        return compute_similarity(reference_paragraph, input_paragraph)
    scores = np.array([process_row(row) for row in reference_df['Paragraphs']])
    return np.max(scores)
with open('Formal.json', 'r') as file:
    formal_paragraphs = json.load(file)
reference_df = pd.DataFrame({'Paragraphs': formal_paragraphs})
input_paragraph = "The quick brown fox jumps over the lazy dog. This is a common test sentence."
best_similarity_score = sentence_structure_similarity(input_paragraph, reference_df)
print(f"Highest similarity score: {best_similarity_score:.2f}/10")
