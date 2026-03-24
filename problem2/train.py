"""
===============================================================================
train.py — Training Loop for Character-Level Name Generation Models
===============================================================================
Trains the three RNN models (Vanilla RNN, BLSTM, RNN+Attention) on the
Indian names dataset. For each model:
  - Uses cross-entropy loss for next-character prediction
  - Implements learning rate scheduling and gradient clipping
  - Saves model checkpoints and training loss curves
  - Reports model architecture and trainable parameter counts

Hyperparameters are configurable via command-line arguments or the
CONFIGS dictionary below.
===============================================================================
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Import our custom modules
from dataset import get_dataloaders, NameDataset
from models import (VanillaRNN, BidirectionalLSTM, RNNWithAttention,
                    count_parameters, print_model_summary)

# --------------------------------------------------------------------------
# Configuration: Hyperparameters for each model
# --------------------------------------------------------------------------
TRAINING_NAMES_FILE = os.path.join(os.path.dirname(__file__), "TrainingNames.txt")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
VIS_DIR = os.path.join(os.path.dirname(__file__), "visualizations")

# Shared training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.003
NUM_EPOCHS = 40
GRAD_CLIP = 5.0          # Max gradient norm for clipping (prevents exploding gradients)
EMBED_SIZE = 64           # Character embedding dimensionality
HIDDEN_SIZE = 128         # Hidden state dimensionality
NUM_LAYERS = 2            # Number of stacked recurrent layers
DROPOUT = 0.2             # Dropout probability for regularization

# Device selection: use GPU if available (much faster for training)
DEVICE = torch.device("cuda" if torch.cuda.is_available() 
                       else "mps" if torch.backends.mps.is_available() 
                       else "cpu")


def train_one_epoch(model: nn.Module, dataloader, criterion, optimizer,
                    device: torch.device) -> float:
    """
    Train the model for one epoch over the full training dataset.
    
    For each batch:
      1. Forward pass: compute next-character prediction logits
      2. Compute cross-entropy loss (ignoring PAD tokens)
      3. Backward pass: compute gradients
      4. Clip gradients to prevent exploding gradient problem
      5. Update weights
    
    Args:
        model: The RNN model to train.
        dataloader: Training data loader.
        criterion: Loss function (CrossEntropyLoss).
        optimizer: Optimizer (Adam).
        device: Device to train on (cpu/cuda/mps).
    
    Returns:
        Average loss over the epoch.
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device)   # (batch, seq_len)
        targets = targets.to(device) # (batch, seq_len)
        
        # Forward pass: get logits for each position
        logits = model(inputs)  # (batch, seq_len, vocab_size)
        
        # Reshape for cross-entropy: (batch * seq_len, vocab_size) vs (batch * seq_len)
        loss = criterion(
            logits.reshape(-1, model.vocab_size),
            targets.reshape(-1)
        )
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        # This is crucial for RNNs which can have unstable gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model: nn.Module, dataloader, criterion,
             device: torch.device) -> float:
    """
    Evaluate the model on the validation set.
    
    Same as training but without gradient computation or weight updates.
    
    Args:
        model: The RNN model to evaluate.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Device.
    
    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            logits = model(inputs)
            loss = criterion(
                logits.reshape(-1, model.vocab_size),
                targets.reshape(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


def plot_training_curves(histories: dict, output_path: str) -> None:
    """
    Plots training and validation loss curves for all models on one figure.
    
    This allows visual comparison of convergence speed and final loss
    across the three architectures.
    
    Args:
        histories: Dict mapping model_name -> {train_losses, val_losses}.
        output_path: Path to save the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {"VanillaRNN": "#e74c3c", "BLSTM": "#3498db", "RNN+Attention": "#2ecc71"}
    
    # Training loss subplot
    for name, history in histories.items():
        axes[0].plot(history["train_losses"], label=name,
                     color=colors.get(name, "gray"), linewidth=2)
    axes[0].set_title("Training Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Cross-Entropy Loss", fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Validation loss subplot
    for name, history in histories.items():
        axes[1].plot(history["val_losses"], label=name,
                     color=colors.get(name, "gray"), linewidth=2)
    axes[1].set_title("Validation Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Cross-Entropy Loss", fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle("Training Curves — Character-Level Name Generation Models",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIS] Training curves saved to: {output_path}")


def train_model(model_class, model_name: str, vocab_size: int,
                train_loader, val_loader, dataset) -> dict:
    """
    Complete training pipeline for a single model:
      1. Initialize model and optimizer
      2. Train for NUM_EPOCHS epochs
      3. Save best checkpoint (lowest validation loss)
      4. Return training history
    
    Args:
        model_class: Model class to instantiate (VanillaRNN, etc.).
        model_name: Display name for this model.
        vocab_size: Vocabulary size from the dataset.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        dataset: NameDataset instance.
    
    Returns:
        Dictionary with train_losses, val_losses, and best model path.
    """
    print(f"\n{'#' * 60}")
    print(f"# Training: {model_name}")
    print(f"{'#' * 60}")
    
    # ---- Initialize model ----
    model = model_class(
        vocab_size=vocab_size,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)
    
    # Print model architecture and parameter count
    print_model_summary(model)
    
    # ---- Loss function and optimizer ----
    # ignore_index=0 tells CrossEntropyLoss to ignore PAD tokens
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler: reduce LR by 0.5x when val loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    
    # ---- Training loop ----
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt")
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        # Train one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        # Evaluate on validation set
        val_loss = evaluate(model, val_loader, criterion, DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Step the learning rate scheduler
        scheduler.step(val_loss)
        
        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "vocab_size": vocab_size,
                "model_name": model_name,
            }, best_model_path)
        
        # Print progress every 2 epochs
        if (epoch + 1) % 2 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Best Val: {best_val_loss:.4f} | "
                  f"Time: {elapsed:.0f}s")
    
    total_time = time.time() - start_time
    print(f"\n  Training complete in {total_time:.1f}s")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Checkpoint saved to: {best_model_path}")
    
    # ---- Generate sample names with the trained model ----
    # Load the best checkpoint for generation
    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    sample_names = model.generate(dataset, max_len=20, temperature=0.8, num_names=10)
    print(f"\n  Sample generated names: {sample_names}")
    
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "total_time": total_time,
        "best_model_path": best_model_path,
        "num_params": count_parameters(model),
    }


def main():
    """
    Main entry point:
      1. Load dataset and create data loaders
      2. Train all three models (Vanilla RNN, BLSTM, RNN+Attention)
      3. Save training history and plot comparison curves
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)
    
    if not os.path.exists(TRAINING_NAMES_FILE):
        print("[ERROR] TrainingNames.txt not found. Run generate_names.py first.")
        return
    
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Hyperparameters:")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs:        {NUM_EPOCHS}")
    print(f"  Hidden size:   {HIDDEN_SIZE}")
    print(f"  Embed size:    {EMBED_SIZE}")
    print(f"  Num layers:    {NUM_LAYERS}")
    print(f"  Dropout:       {DROPOUT}")
    print(f"  Grad clip:     {GRAD_CLIP}")
    
    # ---- Load dataset ----
    train_loader, val_loader, dataset = get_dataloaders(
        TRAINING_NAMES_FILE, batch_size=BATCH_SIZE
    )
    vocab_size = dataset.vocab_size
    
    # ---- Train all three models ----
    all_histories = {}
    model_configs = [
        (VanillaRNN, "VanillaRNN"),
        (BidirectionalLSTM, "BLSTM"),
        (RNNWithAttention, "RNN+Attention"),
    ]
    
    for model_class, model_name in model_configs:
        history = train_model(
            model_class, model_name, vocab_size,
            train_loader, val_loader, dataset
        )
        all_histories[model_name] = history
    
    # ---- Plot comparison curves ----
    curves_path = os.path.join(VIS_DIR, "training_curves.png")
    plot_training_curves(all_histories, curves_path)
    
    # ---- Print summary comparison table ----
    print(f"\n{'=' * 70}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':<20} {'Params':>10} {'Best Val Loss':>15} {'Train Time':>12}")
    print(f"{'-' * 60}")
    for name, hist in all_histories.items():
        print(f"{name:<20} {hist['num_params']:>10,} {hist['best_val_loss']:>15.4f} "
              f"{hist['total_time']:>10.1f}s")
    print(f"{'=' * 70}")
    
    # Save training summary as JSON
    summary = {name: {k: v for k, v in hist.items() if k != "train_losses" and k != "val_losses"}
               for name, hist in all_histories.items()}
    summary_path = os.path.join(CHECKPOINT_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[SAVED] Training summary: {summary_path}")
    
    print("\n[DONE] All models trained successfully!")


if __name__ == "__main__":
    main()
