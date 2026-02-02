"""
Deterministic Amortizer for Production Inference

Fast, deterministic tokenizer that directly predicts boundaries and segments
No curriculum learning, no soft segmentation - optimized for speed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .alse_v3 import VectorQuantizer


class DeterministicAmortizer(nn.Module):
    """
    Deterministic tokenizer for production inference

    Given bytes, directly predicts:
    1. Hard boundaries (threshold at 0.5)
    2. Segment embeddings
    3. Discrete token IDs via VQ

    No curriculum, no soft segmentation - fast and deterministic

    Args:
        num_codes: Vocabulary size (default: 128)
        code_dim: Code vector dimension (default: 12)
    """
    def __init__(self, num_codes=128, code_dim=12):
        super().__init__()

        # Fast boundary predictor
        self.byte_embedding = nn.Embedding(256, 32)
        self.byte_proj = nn.Linear(32, 64)

        self.boundary_net = nn.Sequential(
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=2,
                dim_feedforward=128,
                batch_first=True,
                norm_first=True
            ),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Fast segment encoder
        self.segment_encoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, code_dim)
        )

        # Vector quantizer
        self.vq = VectorQuantizer(num_codes=num_codes, code_dim=code_dim)
        self.num_codes = num_codes

    def forward(self, byte_input):
        """
        Fast deterministic tokenization

        Args:
            byte_input: Byte sequences [B, T]

        Returns:
            token_ids: Discrete token IDs [B, K]
            K: Number of tokens
            boundaries: Predicted boundaries [B, T]
        """
        batch_size, T = byte_input.shape

        # Embed bytes
        byte_emb = self.byte_embedding(byte_input)
        byte_emb_proj = self.byte_proj(byte_emb)

        # Predict boundaries (no curriculum)
        boundaries = self.boundary_net[0](byte_emb_proj)
        for layer in self.boundary_net[1:]:
            boundaries = layer(boundaries)
        boundaries = boundaries.squeeze(-1)

        # Hard segmentation (deterministic)
        # Threshold at 0.5, compute segments
        hard_boundaries = (boundaries > 0.5).float()
        cumsum = torch.cumsum(hard_boundaries, dim=1)
        K = int(cumsum[:, -1].max().item()) + 1
        K = max(1, K)

        # Pool bytes into segments using hard assignment
        segment_indices = torch.arange(
            K, device=byte_input.device
        ).float().unsqueeze(0).expand(batch_size, -1)

        distances = torch.abs(cumsum.unsqueeze(-1) - segment_indices.unsqueeze(1))

        # Sharp weights for near-hard assignment
        weights = F.softmax(-distances * 10.0, dim=-1)

        # Pool bytes
        segments = torch.einsum('btk,btd->bkd', weights, byte_emb_proj)

        # Encode and quantize
        u_k = self.segment_encoder(segments)
        z_q, code_ids, _ = self.vq(u_k)

        return code_ids, K, boundaries

    @torch.no_grad()
    def tokenize(self, byte_input):
        """
        Tokenize bytes to discrete IDs (inference only)

        Args:
            byte_input: Byte sequences [B, T]

        Returns:
            token_ids: Discrete token IDs [B, K]
        """
        token_ids, K, _ = self.forward(byte_input)
        return token_ids

    def inference_speed_test(self, batch_size=8, seq_len=512, num_iterations=100):
        """
        Measure inference speed

        Args:
            batch_size: Batch size for testing
            seq_len: Sequence length
            num_iterations: Number of test iterations

        Returns:
            avg_time_ms: Average time per sequence in milliseconds
        """
        import time

        device = next(self.parameters()).device
        self.eval()

        # Warmup
        dummy_input = torch.randint(0, 256, (batch_size, seq_len), device=device)
        for _ in range(10):
            _ = self.tokenize(dummy_input)

        # Measure
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()

        for _ in range(num_iterations):
            _ = self.tokenize(dummy_input)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        end = time.time()

        total_time = end - start
        avg_time_per_batch = total_time / num_iterations
        avg_time_per_seq = avg_time_per_batch / batch_size * 1000  # Convert to ms

        return avg_time_per_seq

    def load_from_alse(self, alse_model):
        """
        Initialize amortizer weights from trained ALSE model

        Args:
            alse_model: Trained ALSEV3 model
        """
        # Copy byte embeddings
        self.byte_embedding.load_state_dict(alse_model.byte_embedding.state_dict())
        self.byte_proj.load_state_dict(alse_model.byte_proj.state_dict())

        # Copy VQ codebook
        self.vq.load_state_dict(alse_model.vq.state_dict())

        # Copy segment encoder
        self.segment_encoder.load_state_dict(alse_model.segment_encoder.state_dict())

        print("✓ Amortizer initialized from ALSE model")
