"""
===============================================================================
models.py — Character-Level Name Generation Models (From Scratch)
===============================================================================
Implements three recurrent neural network architectures for character-level
name generation, as required by the assignment:

1. Vanilla RNN — Basic recurrent neural network with tanh activation
2. Bidirectional LSTM (BLSTM) — LSTM processing sequences in both directions
3. RNN with Basic Attention — RNN augmented with attention over hidden states

All RNN cells are implemented FROM SCRATCH using basic PyTorch operations
(linear layers, activations, etc.), NOT using nn.RNN or nn.LSTM directly.
Only nn.Linear, nn.Embedding, activations, and nn.Module are used.

Each model follows the same interface:
  - forward(x) -> logits of shape (batch, seq_len, vocab_size)
  - generate(dataset, max_len, temperature) -> list of generated name strings
===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================================================
# MODEL 1: VANILLA RNN (from scratch)
# ==========================================================================
class VanillaRNNCell(nn.Module):
    """
    A single vanilla RNN cell implemented from scratch.
    
    The vanilla RNN computes:
        h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
    
    This is the simplest form of recurrence — the hidden state at each
    time step is a function of the current input and previous hidden state.
    
    Known limitation: Vanilla RNNs suffer from vanishing/exploding gradients
    for long sequences, but character-level names are short enough that this
    is acceptable.
    
    Args:
        input_size: Dimensionality of input features.
        hidden_size: Number of hidden units.
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Input-to-hidden weights: transforms input x_t
        self.W_ih = nn.Linear(input_size, hidden_size)
        # Hidden-to-hidden weights: transforms previous hidden state h_{t-1}
        self.W_hh = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Single RNN step:  h_t = tanh(W_ih * x_t + W_hh * h_{t-1})
        
        Args:
            x: Input at current time step, shape (batch, input_size).
            h_prev: Previous hidden state, shape (batch, hidden_size).
        
        Returns:
            h_new: New hidden state, shape (batch, hidden_size).
        """
        # Combine input and hidden transformations, then apply tanh
        h_new = torch.tanh(self.W_ih(x) + self.W_hh(h_prev))
        return h_new


class VanillaRNN(nn.Module):
    """
    Vanilla RNN for character-level name generation.
    
    Architecture:
        Embedding -> RNN layers -> Fully-connected output -> Logits
    
    The embedding layer converts character indices to dense vectors.
    Multiple RNN layers can be stacked for greater representational capacity.
    The output layer projects hidden states to vocabulary-sized logits.
    
    Args:
        vocab_size: Number of unique characters + special tokens.
        embed_size: Dimensionality of character embeddings.
        hidden_size: Number of hidden units in each RNN layer.
        num_layers: Number of stacked RNN layers.
        dropout: Dropout probability between layers (applied if num_layers > 1).
    """
    
    def __init__(self, vocab_size: int, embed_size: int = 64,
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_name = "VanillaRNN"
        
        # Character embedding layer: converts indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # Stack of RNN cells — first layer takes embedding input,
        # subsequent layers take previous layer's hidden state as input
        self.rnn_cells = nn.ModuleList()
        for i in range(num_layers):
            input_dim = embed_size if i == 0 else hidden_size
            self.rnn_cells.append(VanillaRNNCell(input_dim, hidden_size))
        
        # Dropout for regularization between layers
        self.dropout = nn.Dropout(dropout)
        
        # Output projection: hidden_state -> vocabulary logits
        self.fc_out = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Vanilla RNN.
        
        Processes the input sequence one time step at a time through all
        RNN layers, collecting output logits at each step.
        
        Args:
            x: Input character indices, shape (batch, seq_len).
        
        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Embed input characters
        embedded = self.embedding(x)  # (batch, seq_len, embed_size)
        
        # Initialize hidden states to zeros for each layer
        h = [torch.zeros(batch_size, self.hidden_size, device=device)
             for _ in range(self.num_layers)]
        
        # Process each time step sequentially
        outputs = []
        for t in range(seq_len):
            # Input to the first layer is the embedded character
            layer_input = embedded[:, t, :]  # (batch, embed_size)
            
            # Pass through each RNN layer
            for i in range(self.num_layers):
                h[i] = self.rnn_cells[i](layer_input, h[i])
                layer_input = self.dropout(h[i]) if i < self.num_layers - 1 else h[i]
            
            # Project the final layer's hidden state to vocabulary logits
            output = self.fc_out(h[-1])  # (batch, vocab_size)
            outputs.append(output)
        
        # Stack outputs along time dimension
        logits = torch.stack(outputs, dim=1)  # (batch, seq_len, vocab_size)
        return logits
    
    def generate(self, dataset, max_len: int = 20, temperature: float = 0.8,
                 num_names: int = 100) -> list:
        """
        Generate names by sampling from the model's output distribution.
        
        Uses temperature-controlled sampling: lower temperature produces
        more conservative (common) names, higher temperature produces
        more diverse (creative) names.
        
        Args:
            dataset: NameDataset with char2idx/idx2char mappings.
            max_len: Maximum length of generated names.
            temperature: Sampling temperature (0.1 = conservative, 1.5 = creative).
            num_names: Number of names to generate.
        
        Returns:
            List of generated name strings.
        """
        self.eval()
        device = next(self.parameters()).device
        names = []
        
        with torch.no_grad():
            for _ in range(num_names):
                # Start with SOS token
                current = torch.tensor([[dataset.char2idx["<SOS>"]]], device=device)
                
                # Initialize hidden states
                h = [torch.zeros(1, self.hidden_size, device=device)
                     for _ in range(self.num_layers)]
                
                generated_indices = []
                
                for _ in range(max_len):
                    embedded = self.embedding(current[:, -1])  # (1, embed_size)
                    
                    layer_input = embedded
                    for i in range(self.num_layers):
                        h[i] = self.rnn_cells[i](layer_input, h[i])
                        layer_input = h[i]
                    
                    logits = self.fc_out(h[-1])  # (1, vocab_size)
                    
                    # Apply temperature scaling and sample
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_char = torch.multinomial(probs, 1)  # (1, 1)
                    
                    char_idx = next_char.item()
                    if char_idx == dataset.char2idx["<EOS>"]:
                        break
                    if char_idx == dataset.char2idx["<PAD>"]:
                        break
                    
                    generated_indices.append(char_idx)
                    current = next_char
                
                name = dataset.decode_indices(generated_indices)
                if name:  # Only keep non-empty names
                    names.append(name.capitalize())
        
        self.train()
        return names


# ==========================================================================
# MODEL 2: BIDIRECTIONAL LSTM (from scratch)
# ==========================================================================
class LSTMCell(nn.Module):
    """
    A single LSTM cell implemented from scratch.
    
    The LSTM uses gating mechanisms to control information flow:
        f_t = sigmoid(W_if * x_t + W_hf * h_{t-1} + b_f)     # Forget gate
        i_t = sigmoid(W_ii * x_t + W_hi * h_{t-1} + b_i)     # Input gate
        g_t = tanh(W_ig * x_t + W_hg * h_{t-1} + b_g)        # Cell candidate
        o_t = sigmoid(W_io * x_t + W_ho * h_{t-1} + b_o)     # Output gate
        c_t = f_t * c_{t-1} + i_t * g_t                       # Cell state
        h_t = o_t * tanh(c_t)                                  # Hidden state
    
    The forget gate controls what information to discard from the cell state.
    The input gate controls what new information to store.
    The output gate controls what parts of the cell state to expose.
    
    This design solves the vanishing gradient problem that affects Vanilla RNNs.
    
    Args:
        input_size: Dimensionality of input features.
        hidden_size: Number of hidden units.
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Combined weight matrices for all four gates (efficient computation)
        # Processes input: [forget, input, cell_candidate, output]
        self.W_i = nn.Linear(input_size, 4 * hidden_size)
        # Processes hidden state: [forget, input, cell_candidate, output]
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size)
    
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor,
                c_prev: torch.Tensor) -> tuple:
        """
        Single LSTM step with all four gates computed simultaneously.
        
        Args:
            x: Input at current time step, shape (batch, input_size).
            h_prev: Previous hidden state, shape (batch, hidden_size).
            c_prev: Previous cell state, shape (batch, hidden_size).
        
        Returns:
            Tuple of (h_new, c_new), each of shape (batch, hidden_size).
        """
        # Compute all four gate pre-activations simultaneously
        gates = self.W_i(x) + self.W_h(h_prev)  # (batch, 4 * hidden_size)
        
        # Split into individual gates
        f_gate, i_gate, g_gate, o_gate = gates.chunk(4, dim=-1)
        
        # Apply gate activations
        f_gate = torch.sigmoid(f_gate)  # Forget gate:  what to discard
        i_gate = torch.sigmoid(i_gate)  # Input gate:   what to store
        g_gate = torch.tanh(g_gate)     # Cell candidate: new information
        o_gate = torch.sigmoid(o_gate)  # Output gate:  what to output
        
        # Update cell state: forget old + add new
        c_new = f_gate * c_prev + i_gate * g_gate
        
        # Compute new hidden state
        h_new = o_gate * torch.tanh(c_new)
        
        return h_new, c_new


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM for character-level name generation.
    
    Architecture:
        Embedding -> Forward LSTM + Backward LSTM -> Concatenate -> FC -> Logits
    
    The forward LSTM processes the sequence left-to-right (natural order),
    while the backward LSTM processes right-to-left. Their hidden states
    are concatenated, giving each position context from both directions.
    
    For generation, only the forward direction is used (since we can't
    look into the future when generating sequentially).
    
    Args:
        vocab_size: Number of unique characters + special tokens.
        embed_size: Dimensionality of character embeddings.
        hidden_size: Number of hidden units per direction.
        num_layers: Number of stacked BLSTM layers.
        dropout: Dropout probability between layers.
    """
    
    def __init__(self, vocab_size: int, embed_size: int = 64,
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_name = "BLSTM"
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # Forward LSTM cells (left-to-right processing)
        self.forward_cells = nn.ModuleList()
        # Backward LSTM cells (right-to-left processing)
        self.backward_cells = nn.ModuleList()
        
        for i in range(num_layers):
            # Keep forward and backward connections independent to avoid target leakage!
            input_dim = embed_size if i == 0 else hidden_size
            self.forward_cells.append(LSTMCell(input_dim, hidden_size))
            self.backward_cells.append(LSTMCell(input_dim, hidden_size))
        
        self.dropout = nn.Dropout(dropout)
        
        # Output projection: concatenated bidirectional hidden -> vocab logits
        # Hidden size * 2 because we concatenate forward and backward states
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)
        
        # For generation mode (unidirectional), we need a separate projection
        # that only uses the forward hidden state
        self.fc_gen = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bidirectional forward pass for training.
        
        Processes the sequence in both directions through all layers,
        concatenating forward and backward hidden states at each position.
        
        Args:
            x: Input character indices, shape (batch, seq_len).
        
        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Embed input characters
        layer_input_f = self.embedding(x)
        layer_input_b = layer_input_f.clone()
        
        # Process through each BLSTM layer
        for layer_idx in range(self.num_layers):
            # ---- Forward direction (left to right) ----
            h_f = torch.zeros(batch_size, self.hidden_size, device=device)
            c_f = torch.zeros(batch_size, self.hidden_size, device=device)
            forward_outputs = []
            
            for t in range(seq_len):
                h_f, c_f = self.forward_cells[layer_idx](
                    layer_input_f[:, t, :], h_f, c_f
                )
                forward_outputs.append(h_f)
            
            # ---- Backward direction (right to left) ----
            h_b = torch.zeros(batch_size, self.hidden_size, device=device)
            c_b = torch.zeros(batch_size, self.hidden_size, device=device)
            backward_outputs = [None] * seq_len
            
            for t in range(seq_len - 1, -1, -1):
                h_b, c_b = self.backward_cells[layer_idx](
                    layer_input_b[:, t, :], h_b, c_b
                )
                backward_outputs[t] = h_b
            
            forward_stack = torch.stack(forward_outputs, dim=1)   # (batch, seq_len, hidden)
            backward_stack = torch.stack(backward_outputs, dim=1) # (batch, seq_len, hidden)
            
            # Keep inputs to next layer strictly separate
            layer_input_f = forward_stack
            layer_input_b = backward_stack
            
            # Apply dropout between layers (not on the last layer)
            if layer_idx < self.num_layers - 1:
                layer_input_f = self.dropout(layer_input_f)
                layer_input_b = self.dropout(layer_input_b)
        
        # Project to vocabulary logits using ONLY the forward direction
        # Using backward direction causes target leakage in autoregressive modeling!
        logits = self.fc_gen(forward_stack)  # (batch, seq_len, vocab_size)
        return logits
    
    def generate(self, dataset, max_len: int = 20, temperature: float = 0.8,
                 num_names: int = 100) -> list:
        """
        Generate names using only the forward direction of the BLSTM.
        
        During generation, we can't use backward context (future characters
        don't exist yet), so we only use the forward LSTM cells.
        
        Args:
            dataset: NameDataset with vocabulary mappings.
            max_len: Maximum name length.
            temperature: Sampling temperature.
            num_names: Number of names to generate.
        
        Returns:
            List of generated name strings.
        """
        self.eval()
        device = next(self.parameters()).device
        names = []
        
        with torch.no_grad():
            for _ in range(num_names):
                # Start with SOS token
                current_idx = dataset.char2idx["<SOS>"]
                
                # Initialize hidden/cell states for all forward layers
                h = [torch.zeros(1, self.hidden_size, device=device)
                     for _ in range(self.num_layers)]
                c = [torch.zeros(1, self.hidden_size, device=device)
                     for _ in range(self.num_layers)]
                
                generated_indices = []
                
                for _ in range(max_len):
                    x = torch.tensor([[current_idx]], device=device)
                    embedded = self.embedding(x[:, 0])  # (1, embed_size)
                    
                    layer_input = embedded
                    for i in range(self.num_layers):
                        h[i], c[i] = self.forward_cells[i](layer_input, h[i], c[i])
                        # Since forward and backward were separated, we just pass h as-is
                        layer_input = h[i]
                    
                    logits = self.fc_gen(h[-1])  # (1, vocab_size)
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_char = torch.multinomial(probs, 1)
                    
                    char_idx = next_char.item()
                    if char_idx == dataset.char2idx["<EOS>"]:
                        break
                    if char_idx == dataset.char2idx["<PAD>"]:
                        break
                    
                    generated_indices.append(char_idx)
                    current_idx = char_idx
                
                name = dataset.decode_indices(generated_indices)
                if name:
                    names.append(name.capitalize())
        
        self.train()
        return names


# ==========================================================================
# MODEL 3: RNN WITH BASIC ATTENTION (from scratch)
# ==========================================================================
class AttentionMechanism(nn.Module):
    """
    Basic additive (Bahdanau-style) attention mechanism.
    
    Attention allows the model to focus on different parts of the input
    sequence when generating each output character. At each time step,
    the model computes attention weights over all previous hidden states,
    creating a weighted sum (context vector) that supplements the current
    hidden state.
    
    Score function (additive attention):
        score(h_t, h_s) = V^T * tanh(W_1 * h_t + W_2 * h_s)
    
    where:
        h_t = current hidden state (query)
        h_s = source hidden states (keys/values)
        W_1, W_2, V = learnable parameters
    
    Args:
        hidden_size: Dimensionality of hidden states.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Additive attention score parameters
        self.W_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, query: torch.Tensor,
                keys: torch.Tensor) -> tuple:
        """
        Compute attention weights and context vector.
        
        Args:
            query: Current hidden state, shape (batch, hidden_size).
            keys: All previous hidden states, shape (batch, num_steps, hidden_size).
        
        Returns:
            Tuple of:
              - context: Weighted sum of values, shape (batch, hidden_size).
              - weights: Attention weights, shape (batch, num_steps).
        """
        if keys.size(1) == 0:
            # No previous states to attend to — return zeros
            return torch.zeros_like(query), torch.zeros(query.size(0), 0, device=query.device)
        
        # Expand query to match keys dimensions for broadcasting
        # query: (batch, 1, hidden_size), keys: (batch, num_steps, hidden_size)
        query_expanded = query.unsqueeze(1)
        
        # Compute additive attention scores
        # score = V^T * tanh(W_query * query + W_key * keys)
        scores = self.V(torch.tanh(
            self.W_query(query_expanded) + self.W_key(keys)
        ))  # (batch, num_steps, 1)
        
        scores = scores.squeeze(-1)  # (batch, num_steps)
        
        # Normalize scores to get attention weights (sum to 1)
        weights = F.softmax(scores, dim=-1)  # (batch, num_steps)
        
        # Compute context vector as weighted sum of keys (= values here)
        context = torch.bmm(weights.unsqueeze(1), keys)  # (batch, 1, hidden_size)
        context = context.squeeze(1)  # (batch, hidden_size)
        
        return context, weights


class RNNWithAttention(nn.Module):
    """
    RNN with Basic Attention for character-level name generation.
    
    Architecture:
        Embedding -> RNN layers -> Attention over hidden states ->
        Combine (hidden + context) -> FC -> Logits
    
    At each time step, the attention mechanism computes a weighted sum
    of all previous hidden states (context vector). This context vector
    is concatenated with the current hidden state before the output
    projection, giving the model explicit access to earlier parts of
    the sequence.
    
    This helps with modeling long-range character dependencies in names.
    
    Args:
        vocab_size: Number of unique characters + special tokens.
        embed_size: Dimensionality of character embeddings.
        hidden_size: Number of hidden units in RNN.
        num_layers: Number of stacked RNN layers.
        dropout: Dropout probability.
    """
    
    def __init__(self, vocab_size: int, embed_size: int = 64,
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_name = "RNN+Attention"
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # RNN cells (same as Vanilla RNN) — using LSTM cells here for
        # better gradient flow, with attention on top
        self.rnn_cells = nn.ModuleList()
        for i in range(num_layers):
            input_dim = embed_size if i == 0 else hidden_size
            self.rnn_cells.append(LSTMCell(input_dim, hidden_size))
        
        # Attention mechanism operates on the top-layer hidden states
        self.attention = AttentionMechanism(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Combine hidden state and attention context before output
        # Input: hidden_size (current hidden) + hidden_size (attention context)
        self.fc_combine = nn.Linear(hidden_size * 2, hidden_size)
        
        # Output projection to vocabulary logits
        self.fc_out = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention over accumulated hidden states.
        
        At each time step t:
          1. Run input through RNN layers
          2. Compute attention context over hidden states h_1, ..., h_{t-1}
          3. Combine current hidden state with attention context
          4. Project to vocabulary logits
        
        Args:
            x: Input character indices, shape (batch, seq_len).
        
        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Embed input characters
        embedded = self.embedding(x)  # (batch, seq_len, embed_size)
        
        # Initialize hidden and cell states for each layer
        h = [torch.zeros(batch_size, self.hidden_size, device=device)
             for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=device)
             for _ in range(self.num_layers)]
        
        # Collect top-layer hidden states for attention
        all_hidden = []
        outputs = []
        
        for t in range(seq_len):
            layer_input = embedded[:, t, :]  # (batch, embed_size)
            
            # Pass through RNN layers
            for i in range(self.num_layers):
                h[i], c[i] = self.rnn_cells[i](layer_input, h[i], c[i])
                layer_input = self.dropout(h[i]) if i < self.num_layers - 1 else h[i]
            
            # ---- Attention mechanism ----
            # Create tensor of all previous hidden states for attention
            if all_hidden:
                hidden_stack = torch.stack(all_hidden, dim=1)  # (batch, t, hidden)
                context, attn_weights = self.attention(h[-1], hidden_stack)
            else:
                # First time step: no previous states, context is zeros
                context = torch.zeros(batch_size, self.hidden_size, device=device)
            
            # Store current hidden state for future attention
            all_hidden.append(h[-1].clone())
            
            # Combine current hidden state with attention context
            combined = torch.cat([h[-1], context], dim=-1)  # (batch, hidden*2)
            combined = torch.tanh(self.fc_combine(combined))  # (batch, hidden)
            
            # Project to vocabulary logits
            output = self.fc_out(combined)  # (batch, vocab_size)
            outputs.append(output)
        
        logits = torch.stack(outputs, dim=1)  # (batch, seq_len, vocab_size)
        return logits
    
    def generate(self, dataset, max_len: int = 20, temperature: float = 0.8,
                 num_names: int = 100) -> list:
        """
        Generate names using RNN with attention over generated hidden states.
        
        During generation, the attention mechanism attends to all previously
        generated hidden states, helping maintain coherence in longer names.
        
        Args:
            dataset: NameDataset with vocabulary mappings.
            max_len: Maximum name length.
            temperature: Sampling temperature.
            num_names: Number of names to generate.
        
        Returns:
            List of generated name strings.
        """
        self.eval()
        device = next(self.parameters()).device
        names = []
        
        with torch.no_grad():
            for _ in range(num_names):
                current_idx = dataset.char2idx["<SOS>"]
                
                h = [torch.zeros(1, self.hidden_size, device=device)
                     for _ in range(self.num_layers)]
                c = [torch.zeros(1, self.hidden_size, device=device)
                     for _ in range(self.num_layers)]
                
                all_hidden = []
                generated_indices = []
                
                for _ in range(max_len):
                    x = torch.tensor([[current_idx]], device=device)
                    embedded = self.embedding(x[:, 0])  # (1, embed_size)
                    
                    layer_input = embedded
                    for i in range(self.num_layers):
                        h[i], c[i] = self.rnn_cells[i](layer_input, h[i], c[i])
                        layer_input = h[i]
                    
                    # Attention over previous hidden states
                    if all_hidden:
                        hidden_stack = torch.stack(all_hidden, dim=1)
                        context, _ = self.attention(h[-1], hidden_stack)
                    else:
                        context = torch.zeros(1, self.hidden_size, device=device)
                    
                    all_hidden.append(h[-1].clone())
                    
                    combined = torch.cat([h[-1], context], dim=-1)
                    combined = torch.tanh(self.fc_combine(combined))
                    logits = self.fc_out(combined)
                    
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_char = torch.multinomial(probs, 1)
                    
                    char_idx = next_char.item()
                    if char_idx == dataset.char2idx["<EOS>"]:
                        break
                    if char_idx == dataset.char2idx["<PAD>"]:
                        break
                    
                    generated_indices.append(char_idx)
                    current_idx = char_idx
                
                name = dataset.decode_indices(generated_indices)
                if name:
                    names.append(name.capitalize())
        
        self.train()
        return names


def count_parameters(model: nn.Module) -> int:
    """
    Counts the total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
    
    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module) -> None:
    """
    Prints a summary of the model architecture and parameter count.
    
    Args:
        model: PyTorch model with a model_name attribute.
    """
    total_params = count_parameters(model)
    print(f"\n{'=' * 50}")
    print(f"Model: {model.model_name}")
    print(f"{'=' * 50}")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"\nLayer-by-layer breakdown:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name:40s}  {str(list(param.shape)):20s}  ({param.numel():,})")
    print(f"{'=' * 50}")
