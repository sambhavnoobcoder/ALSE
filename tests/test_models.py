"""
Tests for ALSE models
"""

import torch
import pytest
from alse.models import (
    ALSEV3,
    BoundaryPredictor,
    VectorQuantizer,
    SegmentPriorLM,
    DeterministicAmortizer
)


def test_boundary_predictor():
    """Test boundary predictor forward pass"""
    model = BoundaryPredictor(d_model=64)
    x = torch.randn(2, 100, 64)
    boundaries = model(x)
    assert boundaries.shape == (2, 100)
    assert (boundaries >= 0).all() and (boundaries <= 1).all()


def test_vector_quantizer():
    """Test vector quantization"""
    vq = VectorQuantizer(num_codes=128, code_dim=12)
    z = torch.randn(2, 10, 12)
    z_q, code_idx, vq_loss = vq(z)

    assert z_q.shape == z.shape
    assert code_idx.shape == (2, 10)
    assert code_idx.min() >= 0 and code_idx.max() < 128
    assert vq_loss.item() >= 0


def test_alse_forward():
    """Test ALSE forward pass"""
    model = ALSEV3(target_compression=4.0, num_codes=128, code_dim=12)
    byte_input = torch.randint(0, 256, (2, 512))

    outputs = model(byte_input)

    assert 'loss' in outputs
    assert 'code_idx' in outputs
    assert 'K' in outputs
    assert outputs['loss'].requires_grad


def test_alse_encode():
    """Test ALSE encoding"""
    model = ALSEV3(target_compression=4.0, num_codes=128, code_dim=12)
    model.eval()

    byte_input = torch.randint(0, 256, (2, 512))
    code_idx = model.encode(byte_input)

    assert code_idx.shape[0] == 2
    assert code_idx.min() >= 0 and code_idx.max() < 128


def test_prior_lm():
    """Test prior language model"""
    lm = SegmentPriorLM(num_codes=128, d_model=64, n_layers=2)
    code_ids = torch.randint(0, 128, (2, 20))

    logits = lm(code_ids)

    assert logits.shape == (2, 20, 128)


def test_amortizer():
    """Test deterministic amortizer"""
    amortizer = DeterministicAmortizer(num_codes=128, code_dim=12)
    byte_input = torch.randint(0, 256, (2, 512))

    token_ids, K, boundaries = amortizer(byte_input)

    assert token_ids.shape[0] == 2
    assert token_ids.min() >= 0 and token_ids.max() < 128
    assert K > 0


if __name__ == '__main__':
    pytest.main([__file__])
