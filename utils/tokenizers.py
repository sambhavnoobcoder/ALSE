"""
Tokenizer Utilities

BPE baseline tokenizer and wrapper for comparison with ALSE
"""

import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers


def train_bpe_tokenizer(texts, vocab_size=512, show_progress=False):
    """
    Train BPE tokenizer on text corpus

    Args:
        texts: List of text strings
        vocab_size: Target vocabulary size (default: 512)
        show_progress: Show training progress (default: False)

    Returns:
        tokenizer: Trained BPE tokenizer
    """
    # Initialize BPE model
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Train
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[],
        show_progress=show_progress
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    return tokenizer


class BPETokenizerWrapper:
    """
    Wrapper for BPE tokenizer with PyTorch integration

    Provides encode/decode and metrics computation
    """
    def __init__(self, tokenizer, vocab_size):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

    def encode(self, text):
        """
        Encode text to token IDs

        Args:
            text: Input text string

        Returns:
            token_ids: List of token IDs
        """
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def encode_batch(self, texts):
        """
        Encode batch of texts

        Args:
            texts: List of text strings

        Returns:
            token_ids_list: List of token ID lists
        """
        encodings = self.tokenizer.encode_batch(texts)
        return [enc.ids for enc in encodings]

    def decode(self, token_ids):
        """
        Decode token IDs to text

        Args:
            token_ids: List of token IDs

        Returns:
            text: Decoded text string
        """
        return self.tokenizer.decode(token_ids)

    def get_vocab_size(self):
        """Get vocabulary size"""
        return self.vocab_size

    def compute_compression_ratio(self, texts):
        """
        Compute average bytes per token

        Args:
            texts: List of text strings

        Returns:
            compression_ratio: Average bytes per token
        """
        total_bytes = 0
        total_tokens = 0

        for text in texts:
            byte_count = len(text.encode('utf-8', errors='ignore'))
            token_ids = self.encode(text)
            token_count = len(token_ids)

            total_bytes += byte_count
            total_tokens += token_count

        compression_ratio = total_bytes / max(total_tokens, 1)
        return compression_ratio

    def compute_vocab_usage(self, texts):
        """
        Compute vocabulary usage statistics

        Args:
            texts: List of text strings

        Returns:
            usage_ratio: Fraction of vocab used
            unique_tokens: Set of unique token IDs
        """
        unique_tokens = set()

        for text in texts:
            token_ids = self.encode(text)
            unique_tokens.update(token_ids)

        usage_ratio = len(unique_tokens) / self.vocab_size

        return usage_ratio, unique_tokens

    def tokenize_to_tensor(self, texts, max_length=512, device='cpu'):
        """
        Tokenize texts to PyTorch tensor

        Args:
            texts: List of text strings
            max_length: Maximum sequence length (default: 512)
            device: Target device

        Returns:
            token_tensor: Tensor of token IDs [N, max_length]
            mask_tensor: Tensor of valid positions [N, max_length]
        """
        token_sequences = []
        masks = []

        for text in texts:
            token_ids = self.encode(text)

            # Truncate or pad
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            else:
                token_ids = token_ids + [0] * (max_length - len(token_ids))

            # Create mask
            mask = [1] * min(len(token_ids), max_length)
            if len(mask) < max_length:
                mask = mask + [0] * (max_length - len(mask))

            token_sequences.append(token_ids)
            masks.append(mask)

        # Convert to tensors
        token_tensor = torch.tensor(token_sequences, dtype=torch.long, device=device)
        mask_tensor = torch.tensor(masks, dtype=torch.bool, device=device)

        return token_tensor, mask_tensor


def train_bpe_tokenizers(texts, vocab_sizes=[128, 512, 2048], show_progress=False):
    """
    Train multiple BPE tokenizers with different vocab sizes

    Args:
        texts: List of text strings
        vocab_sizes: List of vocabulary sizes to train (default: [128, 512, 2048])
        show_progress: Show training progress (default: False)

    Returns:
        tokenizers: Dictionary mapping vocab_size -> BPETokenizerWrapper
    """
    tokenizers = {}

    for vocab_size in vocab_sizes:
        print(f"Training BPE tokenizer (vocab={vocab_size})...")
        tokenizer = train_bpe_tokenizer(texts, vocab_size, show_progress)
        tokenizers[vocab_size] = BPETokenizerWrapper(tokenizer, vocab_size)

    return tokenizers


def compare_tokenizations(text, bpe_tokenizer, alse_model, device='cpu'):
    """
    Compare BPE and ALSE tokenizations side-by-side

    Args:
        text: Input text string
        bpe_tokenizer: BPE tokenizer wrapper
        alse_model: ALSE model
        device: Compute device

    Returns:
        comparison: Dictionary with tokenization comparison
    """
    import numpy as np

    # BPE tokenization
    bpe_tokens = bpe_tokenizer.encode(text)
    bpe_count = len(bpe_tokens)

    # ALSE tokenization
    byte_ids = list(text.encode('utf-8', errors='ignore'))
    byte_tensor = torch.tensor([byte_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        alse_tokens = alse_model.encode(byte_tensor)[0].cpu().numpy()
        alse_count = len(alse_tokens)

    # Compute compression
    byte_count = len(byte_ids)
    bpe_compression = byte_count / max(bpe_count, 1)
    alse_compression = byte_count / max(alse_count, 1)

    comparison = {
        'text_preview': text[:100] + '...' if len(text) > 100 else text,
        'byte_count': byte_count,
        'bpe_token_count': bpe_count,
        'alse_token_count': alse_count,
        'bpe_compression': bpe_compression,
        'alse_compression': alse_compression,
        'bpe_tokens': bpe_tokens[:20],  # First 20 tokens
        'alse_tokens': alse_tokens[:20].tolist(),
    }

    return comparison
