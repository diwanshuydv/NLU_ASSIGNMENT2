"""
===============================================================================
analysis.py — Semantic Analysis of Word2Vec Embeddings
===============================================================================
Performs two key analyses on the trained Word2Vec models:

1. Nearest Neighbors: Reports the top-5 most similar words (cosine similarity)
   for the query words: research, student, phd, exam

2. Analogy Experiments: Tests at least 3 word analogies using the vector
   arithmetic approach (e.g., UG : BTech :: PG : ?)

Both CBOW and Skip-gram models are analyzed for comparison.
===============================================================================
"""

import os
from gensim.models import Word2Vec

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Default models to analyze (dim=100, window=5, neg=5)
CBOW_MODEL = os.path.join(MODELS_DIR, "cbow_d300_w5_n5.model")
SKIPGRAM_MODEL = os.path.join(MODELS_DIR, "skipgram_d300_w8_n20.model")

# Words to find nearest neighbors for (as specified in the assignment)
# Note: 'exam' is not in vocabulary; we use 'examination' as the closest match
QUERY_WORDS = ["research", "student", "phd", "examination"]

# Analogy experiments: (positive1, negative1, positive2) -> expected answer
# Format: word1 is to word2 as word3 is to ?
# Analogy vector: word2 - word1 + word3 ≈ answer
ANALOGIES = [
    # UG : B.Tech :: PG : ? (expected: M.Tech or similar)
    {
        "description": "UG : B.Tech :: PG : ?",
        "positive": ["b.tech", "pg"],
        "negative": ["ug"],
    },
    # professor : teaching :: student : ? (expected: learning or study)
    {
        "description": "professor : teaching :: student : ?",
        "positive": ["teaching", "student"],
        "negative": ["professor"],
    },
    # department : head :: institute : ? (expected: director)
    {
        "description": "department : head :: institute : ?",
        "positive": ["head", "institute"],
        "negative": ["department"],
    },
    # semester : examination :: degree : ? (expected: convocation or award)
    {
        "description": "semester : examination :: degree : ?",
        "positive": ["examination", "degree"],
        "negative": ["semester"],
    },
    # bachelor : science :: master : ? (expected: technology or engineering)
    {
        "description": "bachelor : science :: master : ?",
        "positive": ["science", "master"],
        "negative": ["bachelor"],
    },
]


def load_model(filepath: str) -> Word2Vec:
    """
    Loads a saved Word2Vec model from disk.
    
    Args:
        filepath: Path to the saved .model file.
    
    Returns:
        Loaded Word2Vec model.
    """
    print(f"[INFO] Loading model: {filepath}")
    return Word2Vec.load(filepath)


def nearest_neighbors(model: Word2Vec, word: str, topn: int = 5) -> list:
    """
    Finds the top-N nearest neighbors of a word using cosine similarity.
    
    Cosine similarity measures the cosine of the angle between two word
    vectors — values close to 1.0 indicate high semantic similarity.
    
    Args:
        model: Trained Word2Vec model.
        word: Query word to find neighbors for.
        topn: Number of nearest neighbors to return.
    
    Returns:
        List of (word, similarity_score) tuples, or empty list if word
        is not in the vocabulary.
    """
    if word not in model.wv:
        print(f"    [WARN] '{word}' not in vocabulary")
        return []
    return model.wv.most_similar(word, topn=topn)


def solve_analogy(model: Word2Vec, positive: list, negative: list,
                  topn: int = 5) -> list:
    """
    Solves a word analogy using vector arithmetic:
      word2 - word1 + word3 ≈ answer
    
    This relies on Word2Vec's ability to capture semantic relationships
    as linear offsets in the embedding space.
    
    Args:
        model: Trained Word2Vec model.
        positive: Words contributing positively to the query vector.
        negative: Words contributing negatively to the query vector.
        topn: Number of top results to return.
    
    Returns:
        List of (word, similarity_score) tuples.
    """
    # Check that all query words exist in the vocabulary
    for word in positive + negative:
        if word not in model.wv:
            print(f"    [WARN] '{word}' not in vocabulary — skipping analogy")
            return []
    
    return model.wv.most_similar(positive=positive, negative=negative, topn=topn)


def analyze_model(model: Word2Vec, model_name: str) -> None:
    """
    Runs the full semantic analysis on a single model:
      1. Nearest neighbors for all query words
      2. All configured analogy experiments
    
    Args:
        model: Trained Word2Vec model.
        model_name: Display name for this model (e.g., "CBOW" or "Skip-gram").
    """
    print(f"\n{'=' * 70}")
    print(f"SEMANTIC ANALYSIS: {model_name}")
    print(f"{'=' * 70}")
    
    # ---- Part 1: Nearest Neighbors ----
    print(f"\n--- Top-5 Nearest Neighbors (Cosine Similarity) ---")
    for word in QUERY_WORDS:
        neighbors = nearest_neighbors(model, word)
        if neighbors:
            print(f"\n  '{word}':")
            for i, (neighbor, score) in enumerate(neighbors, 1):
                print(f"    {i}. {neighbor:20s} (similarity: {score:.4f})")
        else:
            print(f"\n  '{word}': NOT IN VOCABULARY")
    
    # ---- Part 2: Analogy Experiments ----
    print(f"\n--- Analogy Experiments ---")
    for analogy in ANALOGIES:
        print(f"\n  Analogy: {analogy['description']}")
        results = solve_analogy(model, analogy["positive"], analogy["negative"])
        if results:
            print(f"  Top-5 answers:")
            for i, (word, score) in enumerate(results, 1):
                print(f"    {i}. {word:20s} (similarity: {score:.4f})")
        else:
            print(f"  -> Could not solve (missing vocabulary words)")


def main():
    """
    Main analysis pipeline:
      1. Load both CBOW and Skip-gram models
      2. Run nearest neighbor analysis for query words
      3. Run analogy experiments
      4. Compare results between the two model types
    """
    # ---- Load models ----
    if not os.path.exists(CBOW_MODEL):
        print(f"[ERROR] CBOW model not found at {CBOW_MODEL}. Run train.py first.")
        return
    if not os.path.exists(SKIPGRAM_MODEL):
        print(f"[ERROR] Skip-gram model not found at {SKIPGRAM_MODEL}. Run train.py first.")
        return
    
    cbow = load_model(CBOW_MODEL)
    skipgram = load_model(SKIPGRAM_MODEL)
    
    # ---- Run analysis on both models ----
    analyze_model(cbow, "CBOW (dim=300, window=5, neg=5)")
    analyze_model(skipgram, "SKIP-GRAM (dim=300, window=8, neg=5)")
    
    # ---- Comparison summary ----
    print(f"\n{'=' * 70}")
    print("COMPARISON NOTES")
    print(f"{'=' * 70}")
    
    
    print("[DONE] Semantic analysis complete!")


if __name__ == "__main__":
    main()
