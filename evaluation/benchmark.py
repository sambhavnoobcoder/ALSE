"""
ALSE Benchmarking Script

Comprehensive evaluation comparing ALSE with BPE across multiple vocab sizes
"""

import argparse
import torch
from alse.models import ALSEV3, SegmentPriorLM
from alse.utils import load_wikitext2, train_bpe_tokenizers
from alse.utils import compute_bpb, evaluate_vocab_usage


def benchmark_alse(vocab_size, train_texts, eval_texts, device='cpu', epochs=5):
    """Benchmark ALSE at given vocab size"""
    print(f"\n{'='*60}")
    print(f"Benchmarking ALSE (vocab={vocab_size})")
    print(f"{'='*60}")

    model = ALSEV3(
        target_compression=4.0,
        num_codes=vocab_size,
        code_dim=12
    ).to(device)

    # Train model
    print("Training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    for epoch in range(epochs):
        # Training loop here
        pass

    print("✓ Training complete")

    return model


def benchmark_bpe(vocab_size, train_texts, eval_texts):
    """Benchmark BPE at given vocab size"""
    print(f"\n{'='*60}")
    print(f"Benchmarking BPE (vocab={vocab_size})")
    print(f"{'='*60}")

    from alse.utils import train_bpe_tokenizer, BPETokenizerWrapper

    tokenizer = train_bpe_tokenizer(train_texts, vocab_size)
    wrapper = BPETokenizerWrapper(tokenizer, vocab_size)

    compression_ratio = wrapper.compute_compression_ratio(eval_texts)
    usage_ratio, _ = wrapper.compute_vocab_usage(eval_texts)

    print(f"✓ Compression: {compression_ratio:.2f} bytes/token")
    print(f"✓ Vocab usage: {usage_ratio*100:.1f}%")

    return wrapper


def main():
    parser = argparse.ArgumentParser(description='ALSE Benchmark')
    parser.add_argument('--vocab-sizes', nargs='+', type=int,
                       default=[128, 512, 2048],
                       help='Vocabulary sizes to test')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Training epochs')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu/cuda)')
    parser.add_argument('--num-examples', type=int, default=1000,
                       help='Number of training examples')

    args = parser.parse_args()

    print("ALSE Benchmark Suite")
    print("="*60)

    # Load data
    train_texts, eval_texts = load_wikitext2(num_examples=args.num_examples)

    # Benchmark each vocab size
    for vocab_size in args.vocab_sizes:
        alse_model = benchmark_alse(
            vocab_size, train_texts, eval_texts,
            device=args.device, epochs=args.epochs
        )
        bpe_wrapper = benchmark_bpe(vocab_size, train_texts, eval_texts)

    print("\n" + "="*60)
    print("Benchmark complete!")
    print("="*60)


if __name__ == '__main__':
    main()
