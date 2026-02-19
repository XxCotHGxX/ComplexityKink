import json
import os
import re
import math
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_instruction_memorization(instruction, training_proxy_path):
    """
    Calculates instruction memorization control (M_mem).
    Uses TF-IDF Cosine Similarity against a proxy of common training examples.
    """
    if not os.path.exists(training_proxy_path):
        return 0.0
        
    with open(training_proxy_path, 'r', encoding='utf-8') as f:
        proxy_data = f.read().splitlines()
        
    if not proxy_data:
        return 0.0
        
    vectorizer = TfidfVectorizer(ngram_range=(1, 3)).fit(proxy_data + [instruction])
    vectors = vectorizer.transform([instruction] + proxy_data)
    
    # Similarity of instruction to the most similar proxy example
    similarities = cosine_similarity(vectors[0:1], vectors[1:])
    return np.max(similarities)

if __name__ == "__main__":
    # Placeholder for M_mem logic verification
    print("Memorization Control Logic Initialized.")
