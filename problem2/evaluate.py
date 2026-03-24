"""
===============================================================================
evaluate.py — Quantitative and Qualitative Evaluation of Generated Names
===============================================================================
Evaluates the generated names from each model using two metrics:

1. Novelty Rate: Percentage of generated names that do NOT appear in the
   training set. A higher novelty rate means the model is generating new
   names rather than memorizing the training data.

2. Diversity: Ratio of unique generated names to total generated names.
   Higher diversity means less repetition in the outputs.

Also provides qualitative analysis:
  - Representative samples from each model
  - Common failure modes (too short, too long, gibberish, etc.)
===============================================================================
"""

import os
from collections import Counter

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
TRAINING_NAMES_FILE = os.path.join(os.path.dirname(__file__), "TrainingNames.txt")
GENERATED_DIR = os.path.join(os.path.dirname(__file__), "generated")

# Models to evaluate
MODEL_NAMES = ["VanillaRNN", "BLSTM", "RNN+Attention"]


def load_names(filepath: str) -> list:
    """
    Loads names from a text file (one name per line).
    
    Args:
        filepath: Path to the names file.
    
    Returns:
        List of name strings.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def compute_novelty_rate(generated: list, training: list) -> float:
    """
    Computes the Novelty Rate: percentage of generated names not in training.
    
    Novelty = |generated \\ training| / |generated| * 100
    
    A high novelty rate indicates the model is creative and not just
    memorizing training examples. However, too high novelty with low
    quality names might indicate the model hasn't learned well.
    
    Args:
        generated: List of generated name strings.
        training: List of training name strings.
    
    Returns:
        Novelty rate as a percentage (0-100).
    """
    training_set = set(name.lower() for name in training)
    novel_count = sum(1 for name in generated if name.lower() not in training_set)
    return (novel_count / max(len(generated), 1)) * 100


def compute_diversity(generated: list) -> float:
    """
    Computes the Diversity: ratio of unique names to total generated names.
    
    Diversity = |unique(generated)| / |generated|
    
    A diversity of 1.0 means every generated name is unique.
    Low diversity indicates the model keeps generating the same names.
    
    Args:
        generated: List of generated name strings.
    
    Returns:
        Diversity ratio (0.0 to 1.0).
    """
    unique_names = set(name.lower() for name in generated)
    return len(unique_names) / max(len(generated), 1)


def analyze_quality(generated: list) -> dict:
    """
    Performs qualitative analysis of generated names to identify
    common failure modes.
    
    Failure modes checked:
      - Too short (< 2 chars): Likely incomplete generation
      - Too long (> 15 chars): Model failed to generate EOS
      - Contains digits: Model generated non-alphabetic characters
      - Gibberish (too many consonant clusters): Not phonetically valid
    
    Args:
        generated: List of generated name strings.
    
    Returns:
        Dictionary with quality analysis results.
    """
    vowels = set("aeiou")
    
    too_short = [n for n in generated if len(n) < 2]
    too_long = [n for n in generated if len(n) > 15]
    has_digits = [n for n in generated if any(c.isdigit() for c in n)]
    
    # Detect gibberish: names with 4+ consecutive consonants
    gibberish = []
    for name in generated:
        max_consonants = 0
        current = 0
        for c in name.lower():
            if c.isalpha() and c not in vowels:
                current += 1
                max_consonants = max(max_consonants, current)
            else:
                current = 0
        if max_consonants >= 4:
            gibberish.append(name)
    
    # Name length distribution
    lengths = [len(n) for n in generated]
    avg_length = sum(lengths) / max(len(lengths), 1)
    
    return {
        "too_short": too_short,
        "too_long": too_long,
        "has_digits": has_digits,
        "gibberish": gibberish,
        "avg_length": avg_length,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
    }


def main():
    """
    Main evaluation pipeline:
      1. Load training names and generated names for each model
      2. Compute Novelty Rate and Diversity metrics
      3. Perform qualitative analysis
      4. Print comparison table and analysis
    """
    if not os.path.exists(TRAINING_NAMES_FILE):
        print("[ERROR] TrainingNames.txt not found.")
        return
    
    # Load training names
    training_names = load_names(TRAINING_NAMES_FILE)
    print(f"[INFO] Training set: {len(training_names)} names")
    
    # ---- Evaluate each model ----
    results = {}
    
    for model_name in MODEL_NAMES:
        gen_file = os.path.join(GENERATED_DIR, f"{model_name}_generated.txt")
        
        if not os.path.exists(gen_file):
            print(f"[WARN] Generated file not found for {model_name}: {gen_file}")
            continue
        
        generated = load_names(gen_file)
        print(f"\n{'=' * 60}")
        print(f"EVALUATION: {model_name}")
        print(f"{'=' * 60}")
        print(f"  Total generated names: {len(generated)}")
        
        # ---- Quantitative metrics ----
        novelty = compute_novelty_rate(generated, training_names)
        diversity = compute_diversity(generated)
        
        print(f"\n  QUANTITATIVE METRICS:")
        print(f"    Novelty Rate: {novelty:.1f}%")
        print(f"    Diversity:    {diversity:.3f}")
        
        # ---- Qualitative analysis ----
        quality = analyze_quality(generated)
        
        print(f"\n  QUALITATIVE ANALYSIS:")
        print(f"    Average name length: {quality['avg_length']:.1f} chars")
        print(f"    Length range: {quality['min_length']} - {quality['max_length']} chars")
        print(f"    Too short (< 2 chars): {len(quality['too_short'])}")
        print(f"    Too long (> 15 chars): {len(quality['too_long'])}")
        print(f"    Contains digits: {len(quality['has_digits'])}")
        print(f"    Gibberish (consonant clusters): {len(quality['gibberish'])}")
        
        # Print representative samples
        print(f"\n  REPRESENTATIVE SAMPLES (first 20):")
        for i, name in enumerate(generated[:20]):
            print(f"    {i+1:3d}. {name}")
        
        # Print failure examples if any
        if quality["too_short"]:
            print(f"\n  FAILURE MODE — Too Short: {quality['too_short'][:5]}")
        if quality["too_long"]:
            print(f"  FAILURE MODE — Too Long: {quality['too_long'][:5]}")
        if quality["gibberish"]:
            print(f"  FAILURE MODE — Gibberish: {quality['gibberish'][:5]}")
        
        results[model_name] = {
            "novelty": novelty,
            "diversity": diversity,
            "quality": quality,
            "total": len(generated),
        }
    
    # ---- Comparison table ----
    if results:
        print(f"\n{'=' * 70}")
        print("COMPARISON TABLE")
        print(f"{'=' * 70}")
        print(f"{'Model':<20} {'Generated':>10} {'Novelty%':>10} {'Diversity':>10} "
              f"{'AvgLen':>8} {'Gibberish':>10}")
        print(f"{'-' * 70}")
        for name, res in results.items():
            print(f"{name:<20} {res['total']:>10} {res['novelty']:>9.1f}% "
                  f"{res['diversity']:>10.3f} {res['quality']['avg_length']:>8.1f} "
                  f"{len(res['quality']['gibberish']):>10}")
        print(f"{'=' * 70}")
        
        # ---- Discussion ----
        print(f"\nDISCUSSION:")
        print("""
  Novelty Rate Analysis:
    - Higher novelty indicates the model generates new names rather than
      memorizing training data. Values > 50% suggest good generalization.
    - Very high novelty (> 95%) might mean the model generates gibberish.

  Diversity Analysis:
    - Diversity close to 1.0 means the model rarely repeats names.
    - Low diversity could indicate mode collapse (stuck generating few names).

  Common Failure Modes in Character-Level Name Generation:
    1. Repetitive characters: e.g., "Aaaaaa" when the model gets stuck
    2. Too-short names: Model generates EOS too early
    3. Unpronounceable consonant clusters: e.g., "Brxkt" 
    4. Mixing name patterns: e.g., starting masculine and ending feminine

  Expected differences between models:
    - Vanilla RNN may produce more repetitive or shorter names due to 
      vanishing gradient issues limiting long-range dependencies.
    - BLSTM should capture bidirectional context better during training,
      but generation is still sequential (forward-only).
    - RNN+Attention should produce more coherent longer names by 
      explicitly attending to earlier characters during generation.
        """)
    
    print("[DONE] Evaluation complete!")


if __name__ == "__main__":
    main()
