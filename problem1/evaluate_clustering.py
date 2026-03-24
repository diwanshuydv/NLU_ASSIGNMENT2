import os
import glob
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics import silhouette_score
import warnings

# Suppress sklearn warnings about small number of samples if any
warnings.filterwarnings("ignore")

# Configuration
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# We use the same semantic groups as in visualize.py
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

def load_model(filepath: str) -> Word2Vec:
    return Word2Vec.load(filepath)

def get_word_vectors_and_labels(model: Word2Vec):
    vectors = []
    labels = []
    
    for group_idx, (group_name, word_list) in enumerate(WORD_GROUPS.items()):
        for word in word_list:
            if word in model.wv:
                vectors.append(model.wv[word])
                labels.append(group_idx)  # Numeric labels for silhouette score
                
    return np.array(vectors), np.array(labels)

def evaluate_clustering_quality(model: Word2Vec):
    """
    Evaluates how well the model clusters semantically similar words together.
    We use the Silhouette Score:
    - +1 indicates that the word is far away from the neighboring clusters.
    - 0 indicates that the word is on or very close to the decision boundary between two neighboring clusters.
    - -1 indicates that those words might have been assigned to the wrong cluster.
    """
    vectors, labels = get_word_vectors_and_labels(model)
    
    # We need at least 2 clusters and more than 1 sample per cluster to calculate a meaningful silhouette score
    if len(np.unique(labels)) < 2 or len(vectors) < 5:
        return -1.0, len(vectors)
        
    try:
        # Calculate cosine distance-based silhouette score on the raw embeddings
        score = silhouette_score(vectors, labels, metric='cosine')
        return score, len(vectors)
    except Exception as e:
        return -1.0, len(vectors)

def main():
    if not os.path.exists(MODELS_DIR):
        print("[ERROR] Models directory not found.")
        return
        
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.model"))
    if not model_files:
        print("[ERROR] No .model files found in the models directory.")
        return
        
    print(f"Found {len(model_files)} models to evaluate. Calculating clustering quality (Silhouette score)...")
    
    results = []
    
    for idx, filepath in enumerate(model_files, 1):
        filename = os.path.basename(filepath)
        
        # Parse hyperparameters
        parts = filename.replace(".model", "").split("_")
        try:
            m_type = parts[0]
            dim = int(parts[1][1:])
            win = int(parts[2][1:])
            neg = int(parts[3][1:])
        except:
            m_type, dim, win, neg = "unknown", -1, -1, -1
            
        try:
            model = load_model(filepath)
            score, valid_words = evaluate_clustering_quality(model)
            
            results.append({
                "Filename": filename,
                "Model_Type": m_type,
                "Dim": dim,
                "Window": win,
                "Neg": neg,
                "Clustering_Score": round(score, 4),
                "Valid_Words": valid_words
            })
        except Exception as e:
            pass
            
    df = pd.DataFrame(results)
    if not df.empty:
        # Sort by Clustering Score (descending)
        df = df.sort_values(by=["Clustering_Score", "Valid_Words"], ascending=[False, False])
        
        print(f"\n{'=' * 80}")
        print("CLUSTERING EVALUATION RANKING (Top 15)")
        print("Metric: Cosine Silhouette Score (Higher is better, max 1.0)")
        print(f"{'=' * 80}")
        cols_to_print = ["Filename", "Clustering_Score", "Valid_Words"]
        print(df.head(15)[cols_to_print].to_string(index=False))
        print(f"{'=' * 80}")
        
        # Save to CSV
        out_csv = os.path.join(MODELS_DIR, "clustering_evaluation_results.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n[SAVED] Full clustering evaluation results saved to: {out_csv}")
        
        best_model = df.iloc[0]["Filename"]
        print(f"\n[WINNER] The best model based on cluster mapping is: {best_model}")
        print("You can update `visualize.py` to use this model to see the best visually distinct clusters.")
    else:
        print("\n[WARN] No results gathered.")

if __name__ == "__main__":
    main()
