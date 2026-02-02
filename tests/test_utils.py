"""
Tests for ALSE utilities
"""

import torch
import pytest
from alse.utils import (
    prepare_byte_sequences,
    compute_bpb,
    compute_perplexity,
    evaluate_vocab_usage
)


def test_prepare_byte_sequences():
    """Test byte sequence preparation"""
    texts = ["Hello world", "Test text"]
    byte_seqs, masks = prepare_byte_sequences(texts, max_length=128)

    assert byte_seqs.shape == (2, 128)
    assert masks.shape == (2, 128)
    assert byte_seqs.min() >= 0 and byte_seqs.max() <= 255


def test_compute_bpb():
    """Test BPB computation"""
    perplexity = 10.0
    bytes_per_token = 2.0

    bpb = compute_bpb(perplexity, bytes_per_token)

    assert bpb > 0
    assert isinstance(bpb, float)


def test_evaluate_vocab_usage():
    """Test vocabulary usage evaluation"""
    token_ids = torch.randint(0, 128, (10, 50))
    vocab_size = 128

    usage_ratio, unique_count, unique_tokens = evaluate_vocab_usage(
        token_ids, vocab_size
    )

    assert 0 <= usage_ratio <= 1
    assert unique_count <= vocab_size
    assert len(unique_tokens) == unique_count


if __name__ == '__main__':
    pytest.main([__file__])
