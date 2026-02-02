"""
Data Loading Utilities

Functions for loading and preprocessing datasets for ALSE training
"""

import torch
from datasets import load_dataset


def load_wikitext2(num_examples=1000, train_split=0.8):
    """
    Load WikiText-2 dataset

    Args:
        num_examples: Number of examples to load (default: 1000)
        train_split: Fraction for training (default: 0.8)

    Returns:
        train_texts: List of training texts
        eval_texts: List of evaluation texts
    """
    print(f"Loading WikiText-2 (n={num_examples})...")

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    # Filter empty texts
    texts = [
        item['text'] for item in dataset
        if len(item['text'].strip()) > 0
    ][:num_examples]

    # Split train/eval
    split_idx = int(len(texts) * train_split)
    train_texts = texts[:split_idx]
    eval_texts = texts[split_idx:]

    print(f"✓ Loaded {len(train_texts)} train, {len(eval_texts)} eval texts")

    return train_texts, eval_texts


def prepare_byte_sequences(texts, max_length=512, device='cpu'):
    """
    Convert texts to byte sequences

    Args:
        texts: List of text strings
        max_length: Maximum sequence length (default: 512)
        device: Target device

    Returns:
        byte_sequences: Tensor of byte sequences [N, max_length]
        masks: Tensor of valid position masks [N, max_length]
    """
    byte_sequences = []
    masks = []

    for text in texts:
        # Convert to bytes
        byte_ids = list(text.encode('utf-8', errors='ignore'))

        # Truncate or pad
        if len(byte_ids) > max_length:
            byte_ids = byte_ids[:max_length]
        else:
            # Pad with zeros
            byte_ids = byte_ids + [0] * (max_length - len(byte_ids))

        # Create mask (1 for valid, 0 for padding)
        mask = [1] * min(len(byte_ids), max_length)
        if len(mask) < max_length:
            mask = mask + [0] * (max_length - len(mask))

        byte_sequences.append(byte_ids)
        masks.append(mask)

    # Convert to tensors
    byte_sequences = torch.tensor(byte_sequences, dtype=torch.long, device=device)
    masks = torch.tensor(masks, dtype=torch.bool, device=device)

    return byte_sequences, masks


def load_glue_sst2(num_train=1000, num_eval=200):
    """
    Load GLUE SST-2 sentiment classification dataset

    Args:
        num_train: Number of training examples (default: 1000)
        num_eval: Number of evaluation examples (default: 200)

    Returns:
        train_data: List of (text, label) tuples for training
        eval_data: List of (text, label) tuples for evaluation
    """
    print(f"Loading GLUE SST-2 (train={num_train}, eval={num_eval})...")

    # Load dataset
    train_dataset = load_dataset('glue', 'sst2', split='train')
    val_dataset = load_dataset('glue', 'sst2', split='validation')

    # Extract train data
    train_data = [
        (item['sentence'], item['label'])
        for item in train_dataset
    ][:num_train]

    # Extract eval data
    eval_data = [
        (item['sentence'], item['label'])
        for item in val_dataset
    ][:num_eval]

    print(f"✓ Loaded SST-2: {len(train_data)} train, {len(eval_data)} eval")

    return train_data, eval_data


def create_dataloader(texts, batch_size=8, max_length=512, device='cpu', shuffle=True):
    """
    Create a DataLoader from texts

    Args:
        texts: List of text strings
        batch_size: Batch size (default: 8)
        max_length: Maximum sequence length (default: 512)
        device: Target device
        shuffle: Whether to shuffle (default: True)

    Returns:
        dataloader: Iterator yielding (byte_sequences, masks)
    """
    from torch.utils.data import DataLoader, TensorDataset

    byte_sequences, masks = prepare_byte_sequences(texts, max_length, device)

    dataset = TensorDataset(byte_sequences, masks)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False
    )

    return dataloader


def count_bytes(texts):
    """
    Count total bytes in text corpus

    Args:
        texts: List of text strings

    Returns:
        total_bytes: Total number of bytes
    """
    total_bytes = sum(len(text.encode('utf-8', errors='ignore')) for text in texts)
    return total_bytes


def get_dataset_stats(texts):
    """
    Get statistics about a text corpus

    Args:
        texts: List of text strings

    Returns:
        stats: Dictionary with corpus statistics
    """
    total_texts = len(texts)
    total_bytes = count_bytes(texts)
    avg_bytes_per_text = total_bytes / max(total_texts, 1)

    # Character and word counts
    total_chars = sum(len(text) for text in texts)
    total_words = sum(len(text.split()) for text in texts)

    stats = {
        'num_texts': total_texts,
        'total_bytes': total_bytes,
        'total_chars': total_chars,
        'total_words': total_words,
        'avg_bytes_per_text': avg_bytes_per_text,
        'avg_chars_per_text': total_chars / max(total_texts, 1),
        'avg_words_per_text': total_words / max(total_texts, 1),
    }

    return stats
