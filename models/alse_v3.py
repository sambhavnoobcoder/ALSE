"""
ALSE V3.3 - Core Architecture Components

Implements the main components of the Adaptive Learned Segmentation Encoder:
- BoundaryPredictor: Learns where to segment byte sequences
- AdaptiveSoftSegmentation: Soft segmentation with curriculum learning
- VectorQuantizer: Maps continuous representations to discrete codebook
- LossySegmentDecoder: Reconstructs bytes from discrete tokens
- ALSEV3: Main model integrating all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryPredictor(nn.Module):
    """
    Predicts segmentation boundaries in byte sequences

    Args:
        d_model: Dimension of internal representations (default: 64)
    """
    def __init__(self, d_model=64):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=2,
            dim_feedforward=128,
            batch_first=True,
            norm_first=True
        )
        self.boundary_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: Input embeddings [B, T, D]
            mask: Optional attention mask [B, T]

        Returns:
            boundaries: Boundary probabilities [B, T]
        """
        h = self.encoder(x, src_key_padding_mask=~mask if mask is not None else None)
        return self.boundary_head(h).squeeze(-1)


class AdaptiveSoftSegmentation(nn.Module):
    """
    Soft segmentation with curriculum learning

    Gradually transitions from forced equidistant boundaries to learned boundaries

    Args:
        temperature: Softmax temperature for segment assignment (default: 1.0)
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, boundaries, curriculum_strength=0.0):
        """
        Args:
            embeddings: Byte embeddings [B, T, D]
            boundaries: Predicted boundaries [B, T]
            curriculum_strength: Strength of forced boundaries (0-1)

        Returns:
            segments: Soft-pooled segments [B, K, D]
            segment_mask: Valid segment mask [B, K]
            weights: Assignment weights [B, T, K]
            K: Number of segments
        """
        batch_size, seq_len, d_model = embeddings.shape

        # Apply curriculum learning: blend forced and learned boundaries
        if curriculum_strength > 0.0:
            forced_K = max(1, int(seq_len // 4))
            forced_boundaries = torch.zeros_like(boundaries)
            positions = torch.linspace(
                0, seq_len-1, forced_K+1, device=boundaries.device
            ).long()[1:]
            for b in range(batch_size):
                forced_boundaries[b, positions] = 1.0
            boundaries = (curriculum_strength * forced_boundaries +
                         (1 - curriculum_strength) * boundaries)

        # Compute segment indices via cumulative sum
        cumsum = torch.cumsum(boundaries, dim=1)
        K = torch.ceil(cumsum[:, -1]).long().max().item()
        K = max(1, K)

        # Soft assignment: each byte belongs to all segments with different weights
        segment_indices = torch.arange(
            K, device=embeddings.device
        ).float().unsqueeze(0).expand(batch_size, -1)
        distances = torch.abs(cumsum.unsqueeze(-1) - segment_indices.unsqueeze(1))
        weights = F.softmax(-distances / self.temperature, dim=-1)

        # Pool embeddings into segments
        segments = torch.einsum('btk,btd->bkd', weights, embeddings)
        segment_mask = (segment_indices < torch.ceil(cumsum[:, -1]).unsqueeze(-1))

        return segments, segment_mask, weights, K


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer with straight-through estimator

    Args:
        num_codes: Size of discrete codebook (vocab size)
        code_dim: Dimension of each code vector
        beta: Commitment loss weight (default: 0.25)
    """
    def __init__(self, num_codes=128, code_dim=12, beta=0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta

        # Initialize codebook
        self.embedding = nn.Embedding(num_codes, code_dim)
        self.embedding.weight.data.uniform_(-1/num_codes, 1/num_codes)

    def forward(self, z):
        """
        Args:
            z: Continuous segment embeddings [B, K, D]

        Returns:
            z_q: Quantized embeddings [B, K, D]
            code_idx: Discrete codes [B, K]
            vq_loss: VQ loss (commitment + codebook)
        """
        B, K, D = z.shape
        z_flat = z.reshape(-1, D)

        # Compute distances to all codebook entries
        distances = (torch.sum(z_flat**2, dim=1, keepdim=True) +
                    torch.sum(self.embedding.weight**2, dim=1) -
                    2 * torch.matmul(z_flat, self.embedding.weight.t()))

        # Find nearest codes
        code_idx = torch.argmin(distances, dim=1)
        code_idx = code_idx.reshape(B, K)

        # Lookup quantized embeddings
        z_q = self.embedding(code_idx)

        # Compute VQ loss
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, code_idx, vq_loss


class LossySegmentDecoder(nn.Module):
    """
    Reconstructs bytes from quantized segment embeddings

    Args:
        code_dim: Dimension of input codes
        max_seg_len: Maximum segment length to decode
    """
    def __init__(self, code_dim=12, max_seg_len=16):
        super().__init__()
        self.max_seg_len = max_seg_len

        self.decoder = nn.Sequential(
            nn.Linear(code_dim, 32),
            nn.GELU(),
            nn.Linear(32, max_seg_len * 256)
        )

    def forward(self, z_k):
        """
        Args:
            z_k: Quantized segments [B, K, D]

        Returns:
            logits: Byte reconstruction logits [B, K, max_seg_len, 256]
        """
        B, K, D = z_k.shape
        logits = self.decoder(z_k)
        logits = logits.reshape(B, K, self.max_seg_len, 256)
        return logits


class ALSEV3(nn.Module):
    """
    ALSE V3.3 - Full architecture with all components

    Args:
        target_compression: Target bytes per token (default: 4.0)
        num_codes: Vocabulary size (default: 128)
        code_dim: Code vector dimension (default: 12)
    """
    def __init__(self, target_compression=4.0, num_codes=128, code_dim=12):
        super().__init__()

        # Byte embedding
        self.byte_embedding = nn.Embedding(256, 32)
        self.byte_proj = nn.Linear(32, 64)

        # Segmentation components
        self.boundary_net = BoundaryPredictor(d_model=64)
        self.segmentation = AdaptiveSoftSegmentation(temperature=1.0)

        # Segment encoding
        self.segment_dim = code_dim
        self.segment_encoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, self.segment_dim)
        )

        # Regularization
        self.bottleneck_dropout = nn.Dropout(0.3)
        self.recon_dropout = nn.Dropout(0.3)

        # Vector quantization
        self.vq = VectorQuantizer(num_codes=num_codes, code_dim=self.segment_dim, beta=0.25)

        # Decoder (imported separately for language modeling)
        self.segment_decoder = LossySegmentDecoder(code_dim=self.segment_dim, max_seg_len=16)

        # Configuration
        self.target_compression = target_compression
        self.num_codes = num_codes

        # Training curriculum
        self.current_step = 0
        self.curriculum_warmup_steps = 6000
        self.min_curriculum_strength = 0.35
        self.lambda_max = 8.0
        self.lambda_min = 2.0
        self.lambda_warmup_steps = 7000

    def get_curriculum_strength(self):
        """Compute current curriculum strength (1.0 -> min)"""
        if self.current_step >= self.curriculum_warmup_steps:
            return self.min_curriculum_strength
        progress = self.current_step / self.curriculum_warmup_steps
        return 1.0 - progress * (1.0 - self.min_curriculum_strength)

    def get_lambda_curriculum(self):
        """Compute current compression penalty weight"""
        if self.current_step >= self.lambda_warmup_steps:
            return self.lambda_min
        progress = self.current_step / self.lambda_warmup_steps
        return self.lambda_max - progress * (self.lambda_max - self.lambda_min)

    def forward(self, byte_input, byte_mask=None):
        """
        Forward pass through ALSE

        Args:
            byte_input: Byte sequences [B, T]
            byte_mask: Optional mask [B, T]

        Returns:
            Dictionary with outputs and losses
        """
        batch_size, T = byte_input.shape

        # Embed bytes
        byte_emb = self.byte_embedding(byte_input)
        byte_emb_proj = self.byte_proj(byte_emb)

        # Predict boundaries
        boundaries = self.boundary_net(byte_emb_proj, byte_mask)

        # Soft segmentation with curriculum
        curriculum_strength = self.get_curriculum_strength()
        s_k, segment_mask, weights, K = self.segmentation(
            byte_emb_proj, boundaries, curriculum_strength
        )

        # Encode segments
        u_k = self.segment_encoder(s_k)
        u_k = self.bottleneck_dropout(u_k)

        # Vector quantization
        z_q, code_idx, vq_loss = self.vq(u_k)

        # Decode
        z_q_recon = self.recon_dropout(z_q)
        recon_logits = self.segment_decoder(z_q_recon)

        # Compute reconstruction loss
        recon_loss = self._compute_reconstruction_loss(
            recon_logits, byte_input, weights, segment_mask
        )

        # Compression penalty
        actual_compression = T / max(K, 1)
        lambda_curr = self.get_lambda_curriculum()
        compression_penalty = lambda_curr * F.relu(
            self.target_compression - actual_compression
        )

        # Total loss
        total_loss = recon_loss + vq_loss + compression_penalty

        return {
            'loss': total_loss,
            'recon_loss': recon_loss.item(),
            'vq_loss': vq_loss.item(),
            'compression_penalty': compression_penalty.item(),
            'code_idx': code_idx,
            'K': K,
            'actual_compression': actual_compression,
            'boundaries': boundaries,
        }

    def _compute_reconstruction_loss(self, recon_logits, byte_input, weights, segment_mask):
        """Compute weighted reconstruction loss"""
        B, K, max_seg_len, vocab_size = recon_logits.shape
        T = byte_input.shape[1]

        # Distribute bytes to segments using soft weights
        byte_targets = byte_input.unsqueeze(1).expand(-1, K, -1)

        # Compute loss per segment
        losses = []
        for k in range(K):
            seg_logits = recon_logits[:, k, :, :]
            seg_weights = weights[:, :, k]

            for t in range(min(T, max_seg_len)):
                if t < T:
                    target = byte_targets[:, k, t]
                    logit = seg_logits[:, t, :]
                    weight = seg_weights[:, t]
                    loss_t = F.cross_entropy(logit, target, reduction='none') * weight
                    losses.append(loss_t.mean())

        return torch.stack(losses).mean() if losses else torch.tensor(0.0)

    def encode(self, byte_input):
        """Encode bytes to discrete tokens (inference mode)"""
        with torch.no_grad():
            outputs = self.forward(byte_input)
            return outputs['code_idx']

    def get_vocab_usage(self, code_indices):
        """Compute vocabulary usage statistics"""
        unique_codes = torch.unique(code_indices)
        usage = len(unique_codes) / self.num_codes
        return usage, unique_codes
