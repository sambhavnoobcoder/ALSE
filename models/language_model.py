"""
Language Models for ALSE

- SegmentPriorLM: Small prior LM for perplexity measurement
- LargeScaleLM: 50M parameter LM for fair comparison with BPE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentPriorLM(nn.Module):
    """
    Prior language model for measuring P(token_k | token_{<k})

    Used to evaluate the quality of learned discrete tokens

    Args:
        num_codes: Vocabulary size
        d_model: Model dimension (default: 64)
        n_layers: Number of transformer layers (default: 2)
    """
    def __init__(self, num_codes=128, d_model=64, n_layers=2):
        super().__init__()
        self.code_embedding = nn.Embedding(num_codes, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=2,
            dim_feedforward=128,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_head = nn.Linear(d_model, num_codes)

    def forward(self, code_ids, segment_mask=None):
        """
        Args:
            code_ids: Discrete token IDs [B, K]
            segment_mask: Optional mask for valid segments [B, K]

        Returns:
            logits: Next token prediction logits [B, K, V]
        """
        B, K = code_ids.shape
        x = self.code_embedding(code_ids)

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(K, K, device=code_ids.device), diagonal=1
        ).bool()

        h = self.transformer(x, mask=causal_mask)
        logits = self.output_head(h)

        return logits


class LargeScaleLM(nn.Module):
    """
    Large-scale language model for fair comparison (PATH C)

    50M parameter transformer LM for evaluating modeling capacity
    Same architecture used for both BPE and ALSE tokens

    Args:
        vocab_size: Vocabulary size (128 for ALSE, 512 for BPE)
        d_model: Model dimension (default: 512)
        n_layers: Number of layers (default: 6)
        n_heads: Number of attention heads (default: 8)
    """
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8):
        super().__init__()
        self.vocab_size = vocab_size

        # Token embedding + position embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(1024, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head
        self.output_head = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small random values"""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, token_ids, mask=None):
        """
        Args:
            token_ids: Input token IDs [B, L]
            mask: Optional attention mask [B, L]

        Returns:
            logits: Next token prediction logits [B, L, V]
        """
        B, L = token_ids.shape

        # Embed tokens and positions
        token_emb = self.embedding(token_ids)
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(positions)
        x = token_emb + pos_emb

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(L, L, device=token_ids.device), diagonal=1
        ).bool()

        # Apply transformer
        h = self.transformer(x, mask=causal_mask)

        # Predict next tokens
        logits = self.output_head(h)

        return logits

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class StudentLM(nn.Module):
    """
    Student language model for distillation experiments (PATH B)

    Smaller model that learns from a larger teacher model

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension (default: 64)
        n_layers: Number of layers (default: 2)
    """
    def __init__(self, vocab_size, d_model=64, n_layers=2):
        super().__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=2,
            dim_feedforward=d_model * 2,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        """
        Args:
            token_ids: Input token IDs [B, L]

        Returns:
            logits: Next token prediction logits [B, L, V]
        """
        B, L = token_ids.shape
        x = self.embedding(token_ids)

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(L, L, device=token_ids.device), diagonal=1
        ).bool()

        h = self.transformer(x, mask=causal_mask)
        logits = self.output_head(h)

        return logits


class ClassificationHead(nn.Module):
    """
    Classification head for downstream tasks (GLUE, etc.)

    Args:
        d_model: Input dimension
        num_classes: Number of output classes
    """
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: Encoder output [B, L, D]

        Returns:
            logits: Classification logits [B, num_classes]
        """
        # Pool across sequence (mean pooling)
        pooled = hidden_states.mean(dim=1)
        logits = self.classifier(pooled)
        return logits
