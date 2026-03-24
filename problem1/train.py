"""
===============================================================================
train.py — Word2Vec Model Training (CBOW & Skip-gram)
===============================================================================
Trains Word2Vec models using the gensim library on the preprocessed IIT Jodhpur
corpus. Experiments with different hyperparameter configurations:
  - Embedding dimensions: [50, 100, 200]
  - Context window sizes: [3, 5, 7]
  - Negative samples: [5, 10, 15]

Both CBOW (sg=0) and Skip-gram (sg=1) architectures are trained.
Results are saved in a summary table and models are persisted to disk.
===============================================================================
"""

import os
import time
import itertools
from gensim.models import Word2Vec
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
CORPUS_FILE = os.path.join(os.path.dirname(__file__), "data", "corpus.txt")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Hyperparameter grid for experimentation
EMBEDDING_DIMS = [100, 200, 300]      # Vector dimensionality
WINDOW_SIZES = [ 5, 8, 10]             # Context window (words on each side)
NEGATIVE_SAMPLES = [5, 10, 15]       # Number of negative samples
MIN_COUNT = 2                        # Ignore words appearing fewer than this
EPOCHS = 50                          # Training epochs for each model
WORKERS = 4                          # Parallel training threads


def load_corpus(filepath: str) -> list:
    """
    Loads the sentence corpus from file. Each line is a space-separated
    list of tokens representing one sentence.
    
    Args:
        filepath: Path to the corpus file.
    
    Returns:
        List of sentences, each sentence is a list of string tokens.
    """
    sentences = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                sentences.append(tokens)
    print(f"[INFO] Loaded {len(sentences)} sentences from corpus")
    return sentences


def train_model(sentences: list, sg: int, vector_size: int,
                window: int, negative: int) -> tuple:
    """
    Trains a single Word2Vec model with the given hyperparameters.
    
    Args:
        sentences: List of tokenized sentences.
        sg: 0 for CBOW, 1 for Skip-gram.
        vector_size: Dimensionality of word vectors.
        window: Maximum distance between current and predicted word.
        negative: Number of negative samples per positive sample.
    
    Returns:
        Tuple of (trained_model, training_time_seconds).
    """
    model_type = "skip-gram" if sg else "cbow"
    print(f"  Training {model_type} | dim={vector_size}, window={window}, neg={negative}")
    
    start_time = time.time()
    
    # Initialize and train the Word2Vec model
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,   # Embedding dimensionality
        window=window,             # Context window size
        min_count=MIN_COUNT,       # Minimum word frequency threshold
        sg=sg,                     # 0=CBOW, 1=Skip-gram
        negative=negative,         # Number of negative samples
        epochs=EPOCHS,             # Number of training passes over corpus
        workers=WORKERS,           # Parallel threads
        seed=42,                   # Reproducibility
    )
    
    elapsed = time.time() - start_time
    print(f"    -> Done in {elapsed:.1f}s | Vocab size: {len(model.wv)}")
    
    return model, elapsed


def save_model(model: Word2Vec, model_name: str) -> str:
    """
    Saves a trained Word2Vec model to the models directory.
    
    Args:
        model: Trained Word2Vec model.
        model_name: Descriptive name for the model file.
    
    Returns:
        Path to the saved model file.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    filepath = os.path.join(MODELS_DIR, f"{model_name}.model")
    model.save(filepath)
    return filepath


def main():
    """
    Main training loop:
      1. Load the preprocessed corpus
      2. Train CBOW and Skip-gram models for all hyperparameter combinations
      3. Save all models to disk
      4. Report results in a formatted summary table
    """
    # ---- Load corpus ----
    if not os.path.exists(CORPUS_FILE):
        print("[ERROR] Corpus file not found. Run preprocess.py first.")
        return
    
    sentences = load_corpus(CORPUS_FILE)
    
    # ---- Train models with hyperparameter grid ----
    results = []  # Will store results for the summary table
    
    # Iterate over both model types: CBOW (sg=0) and Skip-gram (sg=1)
    for sg, model_type in [(0, "cbow"), (1, "skipgram")]:
        print(f"\n{'=' * 60}")
        print(f"Training {model_type.upper()} Models")
        print(f"{'=' * 60}")
        
        for dim, win, neg in itertools.product(EMBEDDING_DIMS, WINDOW_SIZES, NEGATIVE_SAMPLES):
            model, elapsed = train_model(sentences, sg, dim, win, neg)
            
            # Save model with a descriptive name
            model_name = f"{model_type}_d{dim}_w{win}_n{neg}"
            save_path = save_model(model, model_name)
            
            # Record results for the summary table
            results.append({
                "Model": model_type.upper(),
                "Dim": dim,
                "Window": win,
                "NegSamples": neg,
                "VocabSize": len(model.wv),
                "TrainTime(s)": round(elapsed, 2),
            })
    
    # ---- Print results summary table ----
    df = pd.DataFrame(results)
    print(f"\n{'=' * 80}")
    print("TRAINING RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(df.to_string(index=False))
    print(f"{'=' * 80}")
    
    # Save results to CSV for the report
    results_path = os.path.join(MODELS_DIR, "training_results.csv")
    df.to_csv(results_path, index=False)
    print(f"\n[SAVED] Results table saved to: {results_path}")
    
    # ---- Identify best models (use default config for downstream tasks) ----
    # We'll use dim=100, window=5, neg=5 as default for analysis
    print(f"\n[INFO] Default models for analysis (dim=100, window=5, neg=5):")
    print(f"  CBOW:     models/cbow_d100_w5_n5.model")
    print(f"  Skip-gram: models/skipgram_d100_w5_n5.model")
    print(f"\n[DONE] Total models trained: {len(results)}")


if __name__ == "__main__":
    main()
