"""
===============================================================================
generate.py — Name Generation from Trained Models
===============================================================================
Loads trained model checkpoints and generates batches of names from each
model (Vanilla RNN, BLSTM, RNN+Attention). Saves generated names to
text files for evaluation and qualitative analysis.

Supports configurable temperature for controlling generation diversity:
  - temperature < 1.0: More conservative/common names
  - temperature = 1.0: Standard sampling
  - temperature > 1.0: More creative/diverse names
===============================================================================
"""

import os
import torch

from dataset import NameDataset
from models import VanillaRNN, BidirectionalLSTM, RNNWithAttention

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
TRAINING_NAMES_FILE = os.path.join(os.path.dirname(__file__), "TrainingNames.txt")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated")

NUM_GENERATE = 200      # Number of names to generate per model
TEMPERATURE = 0.8       # Sampling temperature
MAX_LEN = 20            # Maximum generated name length

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")

# Model configurations: (class, checkpoint_name)
MODELS = [
    (VanillaRNN, "VanillaRNN"),
    (BidirectionalLSTM, "BLSTM"),
    (RNNWithAttention, "RNN+Attention"),
]


def load_trained_model(model_class, model_name: str,
                       vocab_size: int) -> torch.nn.Module:
    """
    Loads a trained model from its best checkpoint.
    
    Args:
        model_class: The model class to instantiate.
        model_name: Name used when saving the checkpoint.
        vocab_size: Vocabulary size (must match training).
    
    Returns:
        The model loaded with trained weights.
    """
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return None
    
    # Instantiate model with same architecture as training
    model = model_class(
        vocab_size=vocab_size,
        embed_size=64,
        hidden_size=128,
        num_layers=2,
        dropout=0.0,  # No dropout during generation
    ).to(DEVICE)
    
    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"[LOADED] {model_name} from epoch {checkpoint['epoch']} "
          f"(val_loss: {checkpoint['val_loss']:.4f})")
    
    return model


def main():
    """
    Main generation pipeline:
      1. Load the dataset (for vocabulary mappings)
      2. Load each trained model
      3. Generate names from each model
      4. Save generated names to files
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(TRAINING_NAMES_FILE):
        print("[ERROR] TrainingNames.txt not found. Run generate_names.py first.")
        return
    
    # ---- Load dataset for vocabulary ----
    dataset = NameDataset(TRAINING_NAMES_FILE)
    
    print(f"\n[INFO] Generating {NUM_GENERATE} names per model")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Max length: {MAX_LEN}")
    print(f"  Device: {DEVICE}")
    
    # ---- Generate names from each model ----
    for model_class, model_name in MODELS:
        print(f"\n{'=' * 50}")
        print(f"Generating from: {model_name}")
        print(f"{'=' * 50}")
        
        model = load_trained_model(model_class, model_name, dataset.vocab_size)
        if model is None:
            continue
        
        # Generate names using the model's generate method
        names = model.generate(
            dataset,
            max_len=MAX_LEN,
            temperature=TEMPERATURE,
            num_names=NUM_GENERATE,
        )
        
        # Save to file
        output_file = os.path.join(OUTPUT_DIR, f"{model_name}_generated.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            for name in names:
                f.write(name + "\n")
        
        print(f"\n  Generated {len(names)} names")
        print(f"  Saved to: {output_file}")
        print(f"  Sample names: {names[:15]}")
        
        # Quick stats
        avg_len = sum(len(n) for n in names) / max(len(names), 1)
        unique_count = len(set(n.lower() for n in names))
        print(f"  Avg length: {avg_len:.1f} chars")
        print(f"  Unique: {unique_count}/{len(names)}")
    
    print(f"\n[DONE] All generated names saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
