"""
Evaluation Metrics

Functions for computing BPB, perplexity, and other evaluation metrics
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_perplexity(model, token_ids, mask=None):
    """
    Compute perplexity on token sequences

    Args:
        model: Language model (must output logits)
        token_ids: Token ID sequences [B, L]
        mask: Optional mask for valid positions [B, L]

    Returns:
        perplexity: Perplexity score
        loss: Cross-entropy loss
    """
    model.eval()

    with torch.no_grad():
        logits = model(token_ids)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = token_ids[:, 1:].contiguous()

        # Compute loss
        if mask is not None:
            shift_mask = mask[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction='none'
            )
            loss = (loss * shift_mask.reshape(-1).float()).sum() / shift_mask.sum()
        else:
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1)
            )

        perplexity = torch.exp(loss).item()

    return perplexity, loss.item()


def compute_bpb(perplexity, bytes_per_token):
    """
    Compute Bits Per Byte (BPB)

    BPB is the ONLY fair comparison metric between tokenization systems
    with different token granularities.

    Formula: BPB = log2(perplexity) / bytes_per_token

    Args:
        perplexity: Language model perplexity
        bytes_per_token: Average bytes per token (compression ratio)

    Returns:
        bpb: Bits per byte
    """
    cross_entropy_bits = np.log2(perplexity)
    bpb = cross_entropy_bits / bytes_per_token
    return bpb


def evaluate_vocab_usage(token_ids, vocab_size):
    """
    Evaluate vocabulary usage

    Args:
        token_ids: Token ID tensor [B, L] or list of token lists
        vocab_size: Total vocabulary size

    Returns:
        usage_ratio: Fraction of vocabulary used
        unique_count: Number of unique tokens
        unique_tokens: Set of unique token IDs
    """
    if isinstance(token_ids, torch.Tensor):
        unique_tokens = torch.unique(token_ids).cpu().numpy().tolist()
    else:
        # Flatten list of lists
        flat_tokens = [token for seq in token_ids for token in seq]
        unique_tokens = list(set(flat_tokens))

    unique_count = len(unique_tokens)
    usage_ratio = unique_count / vocab_size

    return usage_ratio, unique_count, unique_tokens


def compute_compression_ratio(texts, token_ids_list):
    """
    Compute compression ratio (bytes per token)

    Args:
        texts: List of text strings
        token_ids_list: List of token ID lists

    Returns:
        compression_ratio: Average bytes per token
        total_bytes: Total byte count
        total_tokens: Total token count
    """
    total_bytes = 0
    total_tokens = 0

    for text, token_ids in zip(texts, token_ids_list):
        byte_count = len(text.encode('utf-8', errors='ignore'))
        token_count = len(token_ids)

        total_bytes += byte_count
        total_tokens += token_count

    compression_ratio = total_bytes / max(total_tokens, 1)

    return compression_ratio, total_bytes, total_tokens


def evaluate_model_comprehensive(model, texts, tokenizer=None, device='cpu'):
    """
    Comprehensive model evaluation

    Args:
        model: Language model
        texts: List of text strings
        tokenizer: Tokenizer (BPE wrapper or None for byte-level)
        device: Compute device

    Returns:
        metrics: Dictionary with all evaluation metrics
    """
    model.eval()

    # Tokenize texts
    if tokenizer is not None:
        token_ids_list = tokenizer.encode_batch(texts)
        compression_ratio, total_bytes, total_tokens = compute_compression_ratio(
            texts, token_ids_list
        )
        vocab_size = tokenizer.vocab_size

        # Convert to tensor
        max_len = max(len(ids) for ids in token_ids_list)
        token_tensor = torch.zeros((len(texts), max_len), dtype=torch.long, device=device)
        for i, ids in enumerate(token_ids_list):
            token_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    else:
        # Byte-level
        from .data import prepare_byte_sequences
        token_tensor, _ = prepare_byte_sequences(texts, max_length=512, device=device)
        compression_ratio = 1.0  # 1 byte per token
        total_bytes = sum(len(t.encode('utf-8')) for t in texts)
        total_tokens = total_bytes
        vocab_size = 256

    # Compute perplexity
    perplexity, loss = compute_perplexity(model, token_tensor)

    # Compute BPB
    bpb = compute_bpb(perplexity, compression_ratio)

    # Compute vocab usage
    usage_ratio, unique_count, _ = evaluate_vocab_usage(token_tensor, vocab_size)

    metrics = {
        'perplexity': perplexity,
        'loss': loss,
        'bpb': bpb,
        'compression_ratio': compression_ratio,
        'bytes_per_token': compression_ratio,
        'vocab_size': vocab_size,
        'vocab_usage': usage_ratio,
        'unique_tokens': unique_count,
        'total_bytes': total_bytes,
        'total_tokens': total_tokens,
    }

    return metrics


def compare_systems(bpe_metrics, alse_metrics):
    """
    Compare BPE and ALSE systems

    Args:
        bpe_metrics: Metrics dictionary for BPE
        alse_metrics: Metrics dictionary for ALSE

    Returns:
        comparison: Dictionary with comparison results
    """
    bpb_improvement = (bpe_metrics['bpb'] - alse_metrics['bpb']) / bpe_metrics['bpb'] * 100

    comparison = {
        'bpe_bpb': bpe_metrics['bpb'],
        'alse_bpb': alse_metrics['bpb'],
        'bpb_improvement_pct': bpb_improvement,
        'bpe_perplexity': bpe_metrics['perplexity'],
        'alse_perplexity': alse_metrics['perplexity'],
        'bpe_compression': bpe_metrics['compression_ratio'],
        'alse_compression': alse_metrics['compression_ratio'],
        'bpe_vocab_usage': bpe_metrics['vocab_usage'],
        'alse_vocab_usage': alse_metrics['vocab_usage'],
    }

    return comparison


def print_evaluation_summary(metrics, system_name="System"):
    """
    Print formatted evaluation summary

    Args:
        metrics: Metrics dictionary
        system_name: Name of the system being evaluated
    """
    print(f"\n{'='*60}")
    print(f"{system_name} Evaluation Summary")
    print(f"{'='*60}")
    print(f"Perplexity:        {metrics['perplexity']:.4f}")
    print(f"BPB:               {metrics['bpb']:.4f} ← FAIR COMPARISON METRIC")
    print(f"Compression Ratio: {metrics['compression_ratio']:.2f} bytes/token")
    print(f"Vocab Size:        {metrics['vocab_size']}")
    print(f"Vocab Usage:       {metrics['vocab_usage']*100:.1f}% ({metrics['unique_tokens']}/{metrics['vocab_size']})")
    print(f"Total Bytes:       {metrics['total_bytes']:,}")
    print(f"Total Tokens:      {metrics['total_tokens']:,}")
    print(f"{'='*60}\n")
