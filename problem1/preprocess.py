"""
===============================================================================
preprocess.py — Text Preprocessing and Dataset Statistics for Word2Vec
===============================================================================
This script takes raw text files from data/raw/, applies a comprehensive
preprocessing pipeline, and produces a clean corpus file for Word2Vec training.

Preprocessing steps (as required by the assignment):
  1. Removal of boilerplate text and formatting artifacts
  2. Tokenization using NLTK
  3. Lowercasing
  4. Removal of excessive punctuation and non-textual content
  5. Removal of non-English text (Hindi/Devanagari characters)

It also computes and prints dataset statistics:
  - Total number of documents
  - Total number of tokens
  - Vocabulary size
  - Word cloud visualization of most frequent words
===============================================================================
"""

import os
import re
import string
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download required NLTK data (punkt tokenizer)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
CORPUS_FILE = os.path.join(os.path.dirname(__file__), "data", "corpus.txt")
VIS_DIR = os.path.join(os.path.dirname(__file__), "visualizations")

# Common boilerplate/navigational terms to remove from scraped web pages
BOILERPLATE_PATTERNS = [
    r"click here",
    r"read more",
    r"copyright\s*©?\s*\d{4}",
    r"all rights reserved",
    r"privacy policy",
    r"terms and conditions",
    r"cookie policy",
    r"skip to content",
    r"toggle navigation",
    r"home\s*\|",
    r"follow us on",
    r"powered by",
]


def remove_non_english(text: str) -> str:
    """
    Remove non-English text — specifically Devanagari (Hindi) and other
    non-Latin scripts. Only keep ASCII + basic Latin characters.
    
    This is needed because IIT Jodhpur pages may contain Hindi text.
    
    Args:
        text: Input text possibly containing Hindi/Devanagari characters.
    
    Returns:
        Text with only English (Latin) characters, numbers, and punctuation.
    """
    # Remove Devanagari Unicode range (U+0900 to U+097F) and other non-Latin
    text = re.sub(r"[\u0900-\u097F]+", " ", text)  # Hindi/Devanagari
    text = re.sub(r"[\u0980-\u0AFF]+", " ", text)  # Bengali, Gujarati
    text = re.sub(r"[\u0B00-\u0DFF]+", " ", text)  # Oriya, Tamil, Telugu, etc.
    return text


def clean_text(text: str) -> str:
    """
    Applies the full cleaning pipeline to a raw text string:
      1. Remove non-English characters
      2. Remove URLs, email addresses
      3. Remove boilerplate patterns
      4. Remove HTML entities and formatting artifacts
      5. Normalize whitespace
    
    Args:
        text: Raw text from a scraped page or PDF.
    
    Returns:
        Cleaned text ready for tokenization.
    """
    # Step 1: Remove non-English script characters
    text = remove_non_english(text)
    
    # Step 2: Remove URLs and email addresses
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    
    # Step 3: Remove HTML entities (e.g., &nbsp; &amp;)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    
    # Step 4: Remove boilerplate patterns
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    
    # Step 5: Remove special characters but keep basic punctuation
    # Keep periods, commas, hyphens for sentence structure
    text = re.sub(r"[^\w\s.,;:!?'\-()]", " ", text)
    
    # Step 6: Remove excessive punctuation (3+ repeated punctuation chars)
    text = re.sub(r"([.,;:!?])\1{2,}", r"\1", text)
    
    # Step 7: Remove standalone numbers (but keep numbers in words like "2019")
    text = re.sub(r"\b\d+\b", " ", text)
    
    # Step 8: Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def tokenize_and_lower(text: str) -> list:
    """
    Tokenizes text into words using NLTK and converts to lowercase.
    Removes purely punctuation tokens and very short tokens (length 1).
    
    Args:
        text: Cleaned text string.
    
    Returns:
        List of lowercase word tokens.
    """
    tokens = word_tokenize(text.lower())
    
    # Filter out pure punctuation tokens and single-character tokens
    # (except meaningful ones like 'a' and 'i')
    filtered = []
    for token in tokens:
        # Skip pure punctuation
        if all(c in string.punctuation for c in token):
            continue
        # Skip very short meaningless tokens
        if len(token) < 2:
            continue
        # Skip tokens that are just numbers
        if token.isdigit():
            continue
        filtered.append(token)
    
    return filtered


def create_sentence_corpus(text: str) -> list:
    """
    Splits cleaned text into sentences, then tokenizes each sentence.
    This format is needed by gensim Word2Vec (list of lists of words).
    
    Args:
        text: Cleaned text string.
    
    Returns:
        List of sentences, where each sentence is a list of lowercase tokens.
    """
    sentences = sent_tokenize(text)
    tokenized_sentences = []
    for sent in sentences:
        tokens = tokenize_and_lower(sent)
        if len(tokens) >= 3:  # Only keep sentences with at least 3 tokens
            tokenized_sentences.append(tokens)
    return tokenized_sentences


def generate_wordcloud(word_freq: dict, output_path: str) -> None:
    """
    Generates and saves a Word Cloud image from word frequencies.
    
    Args:
        word_freq: Dictionary mapping words to their frequencies.
        output_path: File path to save the word cloud image.
    """
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        max_words=150,
        colormap="viridis",
        contour_width=2,
        contour_color="steelblue",
    )
    wc.generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud — IIT Jodhpur Corpus", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIS] Word cloud saved to: {output_path}")


def main():
    """
    Main pipeline:
      1. Read all raw text files from data/raw/
      2. Clean and preprocess each document
      3. Tokenize and create a sentence-level corpus (for Word2Vec)
      4. Save the corpus to data/corpus.txt
      5. Report dataset statistics
      6. Generate word cloud visualization
    """
    os.makedirs(os.path.dirname(CORPUS_FILE), exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)
    
    # ---- Step 1: Read all raw text files ----
    raw_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".txt")])
    if not raw_files:
        print("[ERROR] No raw text files found in data/raw/. Run scraper.py first.")
        return
    
    print(f"[INFO] Found {len(raw_files)} raw text files")
    
    # ---- Step 2 & 3: Clean of each document and tokenize ----
    all_sentences = []  # List of lists of tokens (for Word2Vec)
    all_tokens = []     # Flat list of all tokens (for statistics)
    doc_count = 0
    
    for filename in raw_files:
        filepath = os.path.join(RAW_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
        
        # Apply cleaning pipeline
        cleaned = clean_text(raw_text)
        if len(cleaned) < 50:  # Skip nearly empty files
            print(f"  [SKIP] {filename} — too short after cleaning")
            continue
        
        # Tokenize into sentences of words
        sentences = create_sentence_corpus(cleaned)
        all_sentences.extend(sentences)
        
        # Flat token list for statistics
        doc_tokens = [token for sent in sentences for token in sent]
        all_tokens.extend(doc_tokens)
        doc_count += 1
        
        print(f"  [OK] {filename}: {len(sentences)} sentences, {len(doc_tokens)} tokens")
    
    # ---- Step 4: Save corpus as one-sentence-per-line format ----
    # Deduplicate sentences while preserving order
    seen_sentences = set()
    unique_sentences = []
    
    for sentence in all_sentences:
        sent_str = " ".join(sentence)
        if sent_str not in seen_sentences:
            seen_sentences.add(sent_str)
            unique_sentences.append(sentence)
            
    print(f"\n[INFO] Found {len(all_sentences) - len(unique_sentences)} exact duplicate sentences. Removing them...")
    all_sentences = unique_sentences
    all_tokens = [token for sent in all_sentences for token in sent] # Recompute all_tokens based on unique sentences
    
    # Each line is a space-separated list of tokens (one sentence)
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        for sentence in all_sentences:
            f.write(" ".join(sentence) + "\n")
    print(f"[SAVED] Corpus saved to: {CORPUS_FILE}")
    print(f"  Total unique sentences: {len(all_sentences)}")
    
    # ---- Step 5: Dataset Statistics ----
    vocab = set(all_tokens)
    word_freq = Counter(all_tokens)
    
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"  Total documents:       {doc_count}")
    print(f"  Total sentences:       {len(all_sentences)}")
    print(f"  Total tokens:          {len(all_tokens)}")
    print(f"  Vocabulary size:       {len(vocab)}")
    print(f"  Avg tokens/sentence:   {len(all_tokens) / max(len(all_sentences), 1):.1f}")
    print(f"  Top 20 most frequent words:")
    for word, count in word_freq.most_common(20):
        print(f"    {word:20s}  {count:6d}")
    print("=" * 60)
    
    # ---- Step 6: Generate Word Cloud ----
    # Remove very common stopwords for a more informative cloud
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))
    filtered_freq = {w: c for w, c in word_freq.items() if w not in stop_words}
    
    wordcloud_path = os.path.join(VIS_DIR, "wordcloud.png")
    generate_wordcloud(filtered_freq, wordcloud_path)
    
    print("\n[DONE] Preprocessing complete!")


if __name__ == "__main__":
    main()
