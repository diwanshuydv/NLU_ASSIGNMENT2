"""
===============================================================================
dataset.py — Character-Level Dataset for Name Generation
===============================================================================
Provides a PyTorch Dataset class for character-level name generation.
Handles:
  - Character-to-index and index-to-character mappings
  - Special tokens: <PAD>, <SOS> (start-of-sequence), <EOS> (end-of-sequence)
  - Converting names to padded integer sequences
  - Train/validation splitting
===============================================================================
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# --------------------------------------------------------------------------
# Special token definitions
# --------------------------------------------------------------------------
PAD_TOKEN = "<PAD>"   # Padding for shorter sequences
SOS_TOKEN = "<SOS>"   # Start-of-sequence marker
EOS_TOKEN = "<EOS>"   # End-of-sequence marker


class NameDataset(Dataset):
    """
    A character-level dataset for Indian name generation.
    
    Each name is converted to a sequence of integer indices:
      [SOS_idx, char1_idx, char2_idx, ..., charN_idx, EOS_idx]
    
    Sequences are padded to max_len for batching.
    
    Attributes:
        names: List of raw name strings.
        char2idx: Dictionary mapping characters to integer indices.
        idx2char: Dictionary mapping integer indices back to characters.
        vocab_size: Total number of unique characters + special tokens.
        max_len: Maximum sequence length (including SOS + EOS tokens).
        data: List of padded integer tensors for each name.
    """
    
    def __init__(self, filepath: str, max_len: int = None):
        """
        Initialize the dataset from a file of names (one per line).
        
        Args:
            filepath: Path to TrainingNames.txt (one name per line).
            max_len: Maximum sequence length. If None, computed from data.
        """
        # ---- Step 1: Load names from file ----
        with open(filepath, "r", encoding="utf-8") as f:
            self.names = [line.strip().lower() for line in f if line.strip()]
        
        print(f"[DATASET] Loaded {len(self.names)} names")
        
        # ---- Step 2: Build character vocabulary ----
        # Collect all unique characters across all names
        all_chars = set()
        for name in self.names:
            all_chars.update(name)
        all_chars = sorted(all_chars)  # Sort for reproducibility
        
        # Create mappings with special tokens at the beginning
        self.char2idx = {
            PAD_TOKEN: 0,
            SOS_TOKEN: 1,
            EOS_TOKEN: 2,
        }
        for i, char in enumerate(all_chars, start=3):
            self.char2idx[char] = i
        
        # Reverse mapping for decoding generated sequences back to strings
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Characters: {''.join(all_chars)}")
        
        # ---- Step 3: Encode names as padded integer sequences ----
        # Each name becomes: [SOS, c1, c2, ..., cN, EOS, PAD, PAD, ...]
        if max_len is None:
            # +2 for SOS and EOS tokens
            self.max_len = max(len(name) for name in self.names) + 2
        else:
            self.max_len = max_len
        
        print(f"  Max sequence length: {self.max_len}")
        
        self.data = []
        for name in self.names:
            encoded = self._encode_name(name)
            self.data.append(encoded)
    
    def _encode_name(self, name: str) -> torch.Tensor:
        """
        Encodes a single name as a padded integer tensor.
        
        Format: [SOS, char1, char2, ..., charN, EOS, PAD, PAD, ...]
        
        Args:
            name: Raw name string (already lowercased).
        
        Returns:
            LongTensor of shape (max_len,) with encoded character indices.
        """
        indices = [self.char2idx[SOS_TOKEN]]
        for char in name:
            if char in self.char2idx:
                indices.append(self.char2idx[char])
        indices.append(self.char2idx[EOS_TOKEN])
        
        # Pad to max_len
        while len(indices) < self.max_len:
            indices.append(self.char2idx[PAD_TOKEN])
        
        # Truncate if somehow longer (shouldn't happen with proper max_len)
        indices = indices[:self.max_len]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def decode_indices(self, indices: list) -> str:
        """
        Decodes a list of integer indices back to a name string.
        Stops at the first EOS token and strips special tokens.
        
        Args:
            indices: List of integer indices from model output.
        
        Returns:
            Decoded name string.
        """
        chars = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            char = self.idx2char.get(idx, "")
            if char == EOS_TOKEN:
                break
            if char not in (PAD_TOKEN, SOS_TOKEN):
                chars.append(char)
        return "".join(chars)
    
    def __len__(self):
        """Returns the total number of names in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns a single encoded name tensor.
        
        For training, the input is all tokens except the last,
        and the target is all tokens except the first (shifted by one).
        This is standard for next-character prediction.
        
        Returns:
            Tuple of (input_tensor, target_tensor), each of shape (max_len-1,).
        """
        sequence = self.data[idx]
        # Input:  [SOS, c1, c2, ..., cN, EOS, PAD, ...] without last token
        # Target: [c1, c2, ..., cN, EOS, PAD, PAD, ...] without first token
        input_seq = sequence[:-1]
        target_seq = sequence[1:]
        return input_seq, target_seq


def get_dataloaders(filepath: str, batch_size: int = 64,
                    val_split: float = 0.1, max_len: int = None) -> tuple:
    """
    Creates train and validation DataLoaders from the name file.
    
    Args:
        filepath: Path to TrainingNames.txt.
        batch_size: Number of names per batch.
        val_split: Fraction of data to use for validation.
        max_len: Maximum sequence length (auto-computed if None).
    
    Returns:
        Tuple of (train_loader, val_loader, dataset) where dataset
        contains the vocabulary mappings needed for generation.
    """
    dataset = NameDataset(filepath, max_len=max_len)
    
    # Split into train and validation sets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Train size: {train_size}, Val size: {val_size}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader, dataset
