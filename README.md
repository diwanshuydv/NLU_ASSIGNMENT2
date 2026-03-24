# NLU Assignment 2

This repository contains two tasks for NLU Assignment 2:
1. **Word Embeddings (Problem 1):** Scrapes academic corpus data, trains Word2Vec models (CBOW, Skip-Gram) with varied hyperparameters, computes analogy similarities, and plots 2D TSNE/PCA clusters for domain-specific embeddings.
2. **Character-Level Name Generation (Problem 2):** Generates sequence embeddings mapping English-Indo names using PyTorch character-level variants (Vanilla RNN, BLSTM, RNN+Attention), logging complexity metrics and validation losses.

## Prerequisites
- macOS/Linux environment
- Python 3.8+ 

It is recommended to run this project inside the initialized virtual environment.

```bash
# Activate the existing virtual environment:
source venv/bin/activate

# Ensure you have your dependencies installed:
pip install -r requirements.txt
```

---

## Executing the Scripts

If you want to re-run the evaluations and get the final output logic described in the report without running the extensive training stages, use the following execution flow:


### Problem 1: Eval & Plots
Moves through all 158 Word2Vec `.model` binaries and outputs the Analogy CSV and the Clustering Output CSV, checking against analogies. Then run the visualize script for the TSNE and PCA images.
```bash
python problem1/evaluate_all_models.py
python problem1/evaluate_clustering.py
python problem1/visualize.py
```

### Problem 2: Name Generating & Validation
Checks generated samples against the 1,000 name input dictionary dataset. Ranks by percent-novelty and identifies potential gibberish generation to detect failures modes.
```bash
python problem2/evaluate.py
```

*(Note: These evaluation scripts skip the actual training loops which take considerably longer to execute. By default, they test the existing `models/` and `checkpoints/` directories already provided within the workspace.)*
