import os
import glob
import pandas as pd
from gensim.models import Word2Vec

# Configuration
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Analogy experiments from analysis.py: (positive1, negative1, positive2) -> expected answer
ANALOGIES = [
    {
        "description": "UG : B.Tech :: PG : ?",
        "positive": ["b.tech", "pg"],
        "negative": ["ug"],
        "expected": "m.tech"
    },
    {
        "description": "professor : teaching :: student : ?",
        "positive": ["teaching", "student"],
        "negative": ["professor"],
        "expected": "learning"
    },
    {
        "description": "department : head :: institute : ?",
        "positive": ["head", "institute"],
        "negative": ["department"],
        "expected": "director"
    },
    {
        "description": "semester : examination :: degree : ?",
        "positive": ["examination", "degree"],
        "negative": ["semester"],
        "expected": "convocation" # or "award", etc.
    },
    {
        "description": "bachelor : science :: master : ?",
        "positive": ["science", "master"],
        "negative": ["bachelor"],
        "expected": "technology"
    },
]

def load_model(filepath: str) -> Word2Vec:
    return Word2Vec.load(filepath)

def score_model_on_analogies(model: Word2Vec) -> dict:
    score = 0
    total = 0
    analogy_results = {}
    
    for i, analogy in enumerate(ANALOGIES):
        desc = analogy["description"]
        expected = analogy["expected"]
        
        # Check vocab
        skip = False
        for word in analogy["positive"] + analogy["negative"]:
            if word not in model.wv:
                skip = True
                break
                
        if skip:
            analogy_results[f"A{i+1}_Solved"] = False
            continue
            
        total += 1
        results = model.wv.most_similar(
            positive=analogy["positive"], 
            negative=analogy["negative"], 
            topn=10
        )
        
        # Check if expected word is in top 10
        words_only = [w for w, _ in results]
        # We use substring match to be a bit forgiving (e.g. m.tech vs mtech)
        solved = any(expected in w or w in expected for w in words_only)
        if solved:
            score += 1
            
        analogy_results[f"A{i+1}_Solved"] = solved
        
    return {
        "Analogy_Score": score,
        "Analogy_Total": total,
        **analogy_results
    }

def main():
    if not os.path.exists(MODELS_DIR):
        print("[ERROR] Models directory not found.")
        return
        
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.model"))
    if not model_files:
        print("[ERROR] No .model files found in the models directory.")
        return
        
    print(f"Found {len(model_files)} models to evaluate. This might take a minute...\n")
    
    results = []
    
    for idx, filepath in enumerate(model_files, 1):
        filename = os.path.basename(filepath)
        print(f"[{idx}/{len(model_files)}] Evaluating {filename}...")
        
        # Parse hyperparameters from filename
        # Expected format: cbow_d100_w5_n5.model
        parts = filename.replace(".model", "").split("_")
        try:
            m_type = parts[0]
            dim = int(parts[1][1:])
            win = int(parts[2][1:])
            neg = int(parts[3][1:])
        except:
            # If filename structure is different, skip parsing
            m_type, dim, win, neg = "unknown", -1, -1, -1
            
        # Evaluate
        try:
            model = load_model(filepath)
            vocab_size = len(model.wv)
            eval_metrics = score_model_on_analogies(model)
            
            results.append({
                "Filename": filename,
                "Model_Type": m_type,
                "Dim": dim,
                "Window": win,
                "Neg": neg,
                "Vocab_Size": vocab_size,
                **eval_metrics
            })
        except Exception as e:
            print(f"  -> Error evaluating {filename}: {e}")
            
    # Sort and display results
    df = pd.DataFrame(results)
    if not df.empty:
        # Sort by analogy score (descending) 
        df = df.sort_values(by=["Analogy_Score", "Vocab_Size"], ascending=[False, False])
        
        print(f"\n{'=' * 80}")
        print("MODEL EVALUATION RANKING (Top 10)")
        print(f"{'=' * 80}")
        cols_to_print = ["Filename", "Analogy_Score", "Analogy_Total"]
        print(df.head(10)[cols_to_print].to_string(index=False))
        print(f"{'=' * 80}")
        
        # Save to CSV
        out_csv = os.path.join(MODELS_DIR, "evaluation_results.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n[SAVED] Full evaluation results saved to: {out_csv}")
    else:
        print("\n[WARN] No results gathered.")

if __name__ == "__main__":
    main()
