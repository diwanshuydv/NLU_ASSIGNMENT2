"""
===============================================================================
visualize.py — Word Embedding Visualization (PCA & t-SNE)
===============================================================================
Projects word embeddings from both CBOW and Skip-gram models into 2D space
using PCA and t-SNE dimensionality reduction techniques.

Visualizes clusters of semantically related word groups to analyze how well
the models capture semantic structure. Word groups include:
  - Academic terms (department, faculty, professor, lecturer, etc.)
  - Student-related terms (student, hostel, mess, library, etc.)
  - Research terms (research, publication, journal, conference, etc.)
  - Program terms (btech, mtech, phd, degree, etc.)
  - Administrative terms (registrar, dean, director, senate, etc.)

Both PCA and t-SNE projections are created for each model type (CBOW/Skip-gram).
===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
VIS_DIR = os.path.join(os.path.dirname(__file__), "visualizations")

CBOW_MODEL = os.path.join(MODELS_DIR, "cbow_d300_w5_n5.model")
SKIPGRAM_MODEL = os.path.join(MODELS_DIR, "skipgram_d300_w8_n15.model")

# Semantic word groups for visualization — each group represents a cluster
# of semantically related words that we expect to appear close together
# in the embedding space.
WORD_GROUPS = {
    "Academic": [
        "department", "faculty", "professor", "teaching", "lecture",
        "course", "curriculum", "syllabus", "education", "academic",
    ],
    "Students": [
        "student", "hostel", "library", "campus", "semester",
        "admission", "scholarship", "graduate", "undergraduate", "enrollment",
    ],
    "Research": [
        "research", "publication", "journal", "conference", "thesis",
        "paper", "innovation", "laboratory", "science", "technology",
    ],
    "Programs": [
        "btech", "mtech", "phd", "degree", "diploma",
        "program", "specialization", "engineering", "postgraduate", "doctoral",
    ],
    "Administration": [
        "registrar", "dean", "director", "senate", "committee",
        "regulation", "policy", "examination", "grade", "credit",
    ],
}

# Color palette for each word group
COLORS = {
    "Academic": "#e74c3c",      # Red
    "Students": "#3498db",      # Blue
    "Research": "#2ecc71",      # Green
    "Programs": "#f39c12",      # Orange
    "Administration": "#9b59b6",# Purple
}


def get_word_vectors(model: Word2Vec, word_groups: dict) -> tuple:
    """
    Extracts word vectors for all words in the defined groups that exist
    in the model's vocabulary.
    
    Args:
        model: Trained Word2Vec model.
        word_groups: Dictionary mapping group names to lists of words.
    
    Returns:
        Tuple of (vectors_array, words_list, group_labels_list, colors_list)
        where each element corresponds to words found in the vocabulary.
    """
    vectors = []
    words = []
    group_labels = []
    colors = []
    
    for group_name, word_list in word_groups.items():
        for word in word_list:
            if word in model.wv:
                vectors.append(model.wv[word])
                words.append(word)
                group_labels.append(group_name)
                colors.append(COLORS[group_name])
            else:
                print(f"  [SKIP] '{word}' not in {group_name} — not in vocabulary")
    
    return np.array(vectors), words, group_labels, colors


def plot_embeddings(coords: np.ndarray, words: list, group_labels: list,
                    colors: list, title: str, output_path: str) -> None:
    """
    Creates a 2D scatter plot of word embeddings with color-coded groups
    and word labels.
    
    Args:
        coords: Nx2 array of 2D coordinates (from PCA or t-SNE).
        words: List of word strings.
        group_labels: List of group names for each word.
        colors: List of color strings for each word.
        title: Plot title.
        output_path: File path to save the plot.
    """
    plt.figure(figsize=(16, 12))
    
    # Plot each group separately to get legend entries
    unique_groups = list(dict.fromkeys(group_labels))  # Preserve order
    for group in unique_groups:
        mask = [g == group for g in group_labels]
        group_coords = coords[mask]
        plt.scatter(
            group_coords[:, 0], group_coords[:, 1],
            c=COLORS[group], label=group, s=100, alpha=0.7, edgecolors="white",
            linewidths=0.5
        )
    
    # Add word labels next to each point
    for i, word in enumerate(words):
        plt.annotate(
            word,
            xy=(coords[i, 0], coords[i, 1]),
            fontsize=8,
            fontweight="bold",
            alpha=0.8,
            xytext=(5, 5),
            textcoords="offset points",
        )
    
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.legend(fontsize=11, loc="best", framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {output_path}")


def visualize_model(model: Word2Vec, model_name: str, prefix: str) -> None:
    """
    Creates PCA and t-SNE visualizations for a single model.
    
    For each dimensionality reduction method, we:
      1. Extract word vectors for all defined semantic groups
      2. Reduce to 2D
      3. Create a labeled scatter plot with color-coded groups
    
    Args:
        model: Trained Word2Vec model.
        model_name: Display name (e.g., "CBOW").
        prefix: Filename prefix for saving plots (e.g., "cbow").
    """
    print(f"\n[VIS] Generating visualizations for {model_name}")
    
    # Get word vectors for our semantic groups
    vectors, words, group_labels, colors = get_word_vectors(model, WORD_GROUPS)
    
    if len(vectors) < 5:
        print(f"  [WARN] Only {len(vectors)} words found — too few for visualization")
        return
    
    print(f"  Found {len(vectors)} words in vocabulary")
    
    # ---- PCA Projection ----
    # PCA finds the directions of maximum variance in the data.
    # It is a linear method, so it preserves global structure but may
    # miss non-linear clusters.
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(vectors)
    
    # Report explained variance (how much information PCA retains)
    explained_var = pca.explained_variance_ratio_
    print(f"  PCA explained variance: {explained_var[0]:.3f}, {explained_var[1]:.3f}")
    
    pca_title = f"PCA — {model_name} Word Embeddings\n(Explained Var: {explained_var[0]:.2%} + {explained_var[1]:.2%})"
    pca_path = os.path.join(VIS_DIR, f"{prefix}_pca.png")
    plot_embeddings(pca_coords, words, group_labels, colors, pca_title, pca_path)
    
    # ---- t-SNE Projection ----
    # t-SNE is a non-linear method that preserves local neighborhood
    # structure. It is better at revealing clusters but the axes have
    # no meaningful interpretation. Perplexity controls the balance
    # between local and global structure.
    perplexity = min(30, len(vectors) - 1)  # Perplexity must be < n_samples
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        max_iter=1000,
        learning_rate="auto",
        init="pca",
    )
    tsne_coords = tsne.fit_transform(vectors)
    
    tsne_title = f"t-SNE — {model_name} Word Embeddings\n(perplexity={perplexity})"
    tsne_path = os.path.join(VIS_DIR, f"{prefix}_tsne.png")
    plot_embeddings(tsne_coords, words, group_labels, colors, tsne_title, tsne_path)


def main():
    """
    Main visualization pipeline:
      1. Load CBOW and Skip-gram models
      2. Generate PCA and t-SNE plots for each model
      3. Provide interpretation notes for the report
    """
    os.makedirs(VIS_DIR, exist_ok=True)
    
    # ---- Load models ----
    if not os.path.exists(CBOW_MODEL) or not os.path.exists(SKIPGRAM_MODEL):
        print("[ERROR] Models not found. Run train.py first.")
        return
    
    cbow = Word2Vec.load(CBOW_MODEL)
    skipgram = Word2Vec.load(SKIPGRAM_MODEL)
    
    # ---- Generate visualizations ----
    visualize_model(cbow, "CBOW (dim=100, win=5, neg=5)", "cbow")
    visualize_model(skipgram, "Skip-gram (dim=100, win=5, neg=5)", "skipgram")
    
    # ---- Print interpretation notes ----
    print(f"\n{'=' * 70}")
    print("INTERPRETATION GUIDE (for the report)")
    print(f"{'=' * 70}")
    print("""
Expected Clustering Behavior:
  - Words from the same semantic group (e.g., all 'Research' words) should
    cluster together in both PCA and t-SNE projections.
  - t-SNE should show tighter, more distinct clusters than PCA since it
    preserves local neighborhood structure better.

CBOW vs Skip-gram Differences:
  - CBOW tends to produce smoother, more averaged embeddings. Related words
    may form broader, less distinct clusters.
  - Skip-gram captures finer-grained relationships and may produce tighter
    clusters with more separation between groups.
  - Skip-gram generally performs better on rare words, which may be visible
    as better placement of infrequent domain-specific terms.

PCA vs t-SNE Differences:
  - PCA is a linear projection that preserves global variance directions.
    The axes have interpretable meaning (directions of maximum variance).
  - t-SNE is non-linear and focuses on preserving local neighborhoods.
    It typically reveals clusters more clearly but distances between
    clusters are not necessarily meaningful.
    """)
    
    print("[DONE] All visualizations saved to visualizations/")


if __name__ == "__main__":
    main()
