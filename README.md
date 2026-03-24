# NLU Assignment 2: Word Embeddings & Character-Level Name Generation 🧠📝

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)
![Gensim](https://img.shields.io/badge/Gensim-Word2Vec-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

This repository contains the complete implementation, evaluation scripts, and formal report for **Natural Language Understanding (NLU) Assignment 2**. The assignment is divided into two distinct NLP problems: Training Word2Vec embeddings on a custom scraped academic corpus, and generative character-level modeling of Indian names using custom-built Recurrent Neural Networks (Vanilla RNN, BLSTM, and RNN+Attention).

---

## 📑 Table of Contents
- [Project Overview](#project-overview)
  - [Problem 1: Word Embeddings](#problem-1-word-embeddings)
  - [Problem 2: Character-Level Name Generation](#problem-2-character-level-name-generation)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Execution Instructions](#execution-instructions)
  - [Quick Answers Extraction](#quick-answers-extraction)
  - [Problem 1 Evaluation](#problem-1-evaluation)
  - [Problem 2 Evaluation](#problem-2-evaluation)
- [Training New Models (Optional)](#training-new-models-optional)
- [Results Snapshot](#results-snapshot)

---

## 🎯 Project Overview

### Problem 1: Word Embeddings
In this task, we built a domain-specific Word2Vec embedding space from scratch.
1. **Corpus Generation:** Scraped text data from the IIT Jodhpur website (including departments, academics, and research pages) using `BeautifulSoup`.
2. **Preprocessing:** Cleaned, tokenized, and normalized the text into a unified vocabulary.
3. **Word2Vec Training:** Trained both **CBOW (Continuous Bag of Words)** and **Skip-gram** models with Negative Sampling using the `gensim` library. We performed an extensive hyperparameter search across 158 setups (varying dimensions, window sizes, and negative samples).
4. **Semantic Evaluation:** Evaluated models quantitatively using Analogy solving (e.g., *semester : examination :: degree : ?*) and Silhouette Clustering Scores.
5. **Visualization:** Projected high-dimensional word clusters (Students, Academic, Research, Programs, Administration) into 2D spaces using PCA and t-SNE.

### Problem 2: Character-Level Name Generation
In this task, we implemented character-level Sequence-to-Sequence generation models **entirely from scratch** using PyTorch linear operations (no `nn.RNN` or `nn.LSTM` shortcuts).
1. **Dataset:** Filtered and curated a high-quality subset of 1,000 Indian First Names from a larger CSV constraint.
2. **Architectures:**
   - **Vanilla RNN:** Basic recurrent dynamics suffering from long-term dependency limits.
   - **Bidirectional LSTM (BLSTM):** Advanced gating mechanism processing context forwards and backwards.
   - **RNN + Basic Attention:** An attention mechanism dynamically weighing sequential hidden states (Bahdanau-style) concatenated for sequence decoding.
3. **Generation:** Starting from a `<SOS>` (Start of Sequence) token, the models autogressively predict characters until `<EOS>`.
4. **Quantitative Metrics:** Measured generalizability via **Novelty Rate** (generating names not in the training set) and **Diversity** (ratio of unique generated names). 

---

## 📂 Repository Structure

```text
NLU_ASSIGNMENT2/
├── extract_answers.py            # Extracts targeted assignment answers
├── requirements.txt              # PyPI dependencies
├── Report.md                     # Comprehensive Formal Analysis Report
├── report.pdf                    # PDF version of the formal report
├── problem1/
│   ├── scraper.py                # IITJ Web Scraper
│   ├── preprocess.py             # Data cleaning pipeline
│   ├── train.py                  # Trains 158 Word2Vec variations
│   ├── analysis.py               # Semantic and Nearest Neighbors analysis
│   ├── evaluate_all_models.py    # Generates analogy CSV across all models
│   ├── evaluate_clustering.py    # Generates Silhouette scores for clustering
│   ├── visualize.py              # t-SNE and PCA visualization pipeline
│   ├── data/                     # Raw txt formats and combined corpus
│   ├── models/                   # Saved Word2Vec .model binaries & CSV outputs
│   └── visualizations/           # Output graphs and word clouds
└── problem2/
    ├── dataset.py                # Character dataloader and sequence padding
    ├── models.py                 # Core Architectures built from scratch!
    ├── train.py                  # PyTorch training loop with checkpointing
    ├── evaluate.py               # Evaluates generation Novelty & Diversity
    ├── generate.py               # Script testing custom inference
    ├── checkpoints/              # Stored .pt model weights and JSON logs
    ├── generated/                # Outputs of the generative inference (.txt)
    └── visualizations/           # Loss curves plot
```

---

## ⚙️ Installation & Setup

It is highly recommended to run this repository inside an activated Python virtual environment.

```bash
# Clone the repository
git clone https://github.com/diwanshuydv/NLU_ASSIGNMENT2.git
cd NLU_ASSIGNMENT2

# Ensure your virtual environment is activated
# macOS/Linux:
python -m venv venv
source venv/bin/activate

# Install all dependencies (PyTorch, Gensim, Scikit-Learn, Pandas, Matplotlib, BeautifulSoup4, NLTK)
pip install -r requirements.txt
```

---

## 🚀 Execution Instructions

*Note: The actual training files computationally take hours to finish. You do not need to retrain the models. Pre-trained checkpoints and models exist respectively in the `/models/` and `/checkpoints/` directories. Running the evaluation instructions below immediately outputs the results logged in `Report.pdf`.*


### Problem 1 Evaluation
Analyze all Word2Vec `.model` binaries against analogy tasks and clustering silhouettes. Post-evaluation, plot the t-SNE and PCA dimensionalities for the best-performing models to visually verify semantic grouping.
```bash
python problem1/evaluate_all_models.py
python problem1/evaluate_clustering.py
python problem1/analysis.py
python problem1/visualize.py
```

### Problem 2 Evaluation
Evaluate the previously generated names output by testing it against the original 1,000 training names payload to compute Novelty, Diversity, and detect gibberish strings.
```bash
python problem2/evaluate.py
```

---

## 🏃‍♂️ Training New Models (Optional)

If you'd like to completely blow away the existing files and train both Word2Vec setups and Character-Level sequences from scratch:

**Retrain Problem 1 (Word2Vec):**
```bash
python problem1/scraper.py
python problem1/preprocess.py
python problem1/train.py
```

**Retrain Problem 2 (RNNs):**
```bash
python problem2/train.py
```

---

## 📊 Results Snapshot

*For a highly detailed technical breakdown and formal tables reviewing hyperparameter influence, please see [Report.md](./Report.md).*

- **Word2Vec Best Model:** Skip-gram models consistently captured richer structural semantics (`skipgram_d200_w7_n10` yielding the highest structured clustering).
- **Novel Generation Architecture:** All custom RNN architectures generated names hitting high generalizability bounds:
  - **Vanilla RNN:** Novelty **89.0%** | Diversity **0.960**
  - **BLSTM:** Novelty **92.0%** | Diversity **0.975**
  - **RNN + Attention:** Novelty **91.5%** | Diversity **0.955**

Sample Indian-origin names synthetically produced by the custom Attention Model:
> *Nitir, Ratin, Deekyakt, Jeesha, Vavi, Pradhana, Sanal*

---
*Created for NLU Assignment 2.*
