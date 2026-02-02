# ALSE Architecture Documentation

## Overview

ALSE (Adaptive Learned Segmentation Encoder) learns discrete tokenization end-to-end through a VQ-VAE architecture with soft segmentation.

## Components

### 1. Boundary Predictor

Learns where to segment byte sequences into tokens.

```
Input: Byte embeddings [B, T, D]
Output: Boundary probabilities [B, T]
```

Architecture:
- Transformer encoder layer
- MLP head with sigmoid activation

### 2. Soft Segmentation

Converts byte sequences to variable-length segments using soft pooling.

Key features:
- Curriculum learning (forced → learned boundaries)
- Temperature-controlled softmax for assignment
- Differentiable throughout training

### 3. Segment Encoder

Encodes variable-length segments to fixed-size vectors.

```
Input: Segments [B, K, D]
Output: Continuous embeddings [B, K, code_dim]
```

### 4. Vector Quantizer

Maps continuous embeddings to discrete codebook entries.

Components:
- Learnable codebook [vocab_size, code_dim]
- Nearest-neighbor assignment
- Straight-through estimator for gradients

Loss:
```
VQ_loss = ||sg[z] - e||² + β||z - sg[e]||²
```

where:
- z = encoder output
- e = quantized embedding
- sg = stop-gradient
- β = commitment weight (0.25)

### 5. Lossy Decoder

Reconstructs bytes from discrete tokens.

```
Input: Quantized embeddings [B, K, code_dim]
Output: Byte logits [B, K, max_seg_len, 256]
```

### 6. Prior Language Model

Learns P(token_k | token_{<k}) for perplexity evaluation.

Architecture:
- Token embeddings
- Causal transformer
- Next-token prediction head

## Training

### Curriculum Learning

```python
curriculum_strength = 1.0 → min_strength (over warmup_steps)
lambda = lambda_max → lambda_min (over lambda_warmup_steps)
```

Gradually transitions from forced equidistant boundaries to fully learned boundaries.

### Loss Function

```
Total_loss = Reconstruction_loss + VQ_loss + Compression_penalty

Reconstruction_loss = Σ CE(predicted_bytes, target_bytes) * soft_weights
VQ_loss = codebook_loss + β * commitment_loss
Compression_penalty = λ * ReLU(target_compression - actual_compression)
```

### Optimization

- Optimizer: AdamW
- Learning rate: 5e-4
- Batch size: 8
- Gradient clipping: 1.0

## Inference

### Training Mode
Uses soft segmentation with curriculum.

### Inference Mode (Amortizer)
Fast deterministic tokenization:
1. Predict boundaries (hard threshold at 0.5)
2. Pool bytes into segments
3. Encode and quantize
4. Return discrete token IDs

Speed: ~1.42ms per sequence (440 bytes)

## Key Design Decisions

### Why Soft Segmentation?

Hard segmentation is non-differentiable. Soft segmentation allows:
- Gradient flow through segment boundaries
- Smooth optimization landscape
- Curriculum learning from forced to learned

### Why Curriculum Learning?

Starting with fully learned boundaries leads to:
- Poor local minima (no segmentation)
- Training instability

Curriculum provides:
- Strong initialization signal
- Gradual complexity increase
- Stable training

### Why VQ-VAE?

Discrete codes enable:
- Standard language model evaluation
- Fair comparison with BPE
- Integration with existing LM frameworks

## Comparison with BPE

| Aspect | BPE | ALSE |
|--------|-----|------|
| Learning | Heuristic frequency-based | End-to-end gradient-based |
| Segmentation | Fixed (greedy merge) | Learned (soft boundaries) |
| Optimization | N/A | Jointly with model |
| Flexibility | Fixed after training | Adapts during training |

## Implementation Notes

- Use `batch_first=True` for all transformers
- Normalize embeddings before VQ
- Apply dropout after encoding/before decoding
- Clip gradients to prevent instability
- Monitor compression ratio during training

## References

- VQ-VAE: van den Oord et al., 2017
- Soft Segmentation: Inspired by soft attention mechanisms
- Curriculum Learning: Bengio et al., 2009
