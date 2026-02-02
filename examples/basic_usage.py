"""
ALSE Basic Usage Example

Demonstrates how to:
1. Load data
2. Train ALSE model
3. Evaluate on test data
4. Compare with BPE baseline
"""

import torch
from alse.models import ALSEV3
from alse.utils import load_wikitext2, prepare_byte_sequences
from alse.utils import train_bpe_tokenizer, BPETokenizerWrapper
from alse.utils import compute_bpb, compute_perplexity

def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = 128
    num_epochs = 5
    batch_size = 8

    print("="*60)
    print("ALSE Basic Usage Example")
    print("="*60)

    # 1. Load data
    print("\n[1/5] Loading WikiText-2...")
    train_texts, eval_texts = load_wikitext2(num_examples=1000)

    # 2. Initialize ALSE
    print("\n[2/5] Initializing ALSE model...")
    model = ALSEV3(
        target_compression=4.0,
        num_codes=vocab_size,
        code_dim=12
    ).to(device)

    print(f"  ✓ Vocab size: {vocab_size}")
    print(f"  ✓ Device: {device}")

    # 3. Train ALSE
    print(f"\n[3/5] Training ALSE ({num_epochs} epochs)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        # Prepare batches
        for i in range(0, len(train_texts), batch_size):
            batch_texts = train_texts[i:i+batch_size]
            byte_sequences, masks = prepare_byte_sequences(
                batch_texts, max_length=512, device=device
            )

            # Forward pass
            outputs = model(byte_sequences, masks)
            loss = outputs['loss']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.current_step += 1
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    # 4. Evaluate ALSE
    print("\n[4/5] Evaluating ALSE...")
    model.eval()

    # Collect tokens
    all_tokens = []
    for text in eval_texts[:100]:
        byte_seq = torch.tensor(
            [list(text.encode('utf-8', errors='ignore'))],
            device=device
        )
        with torch.no_grad():
            tokens = model.encode(byte_seq)
            all_tokens.extend(tokens[0].cpu().tolist())

    # Compute metrics
    from alse.models import SegmentPriorLM
    prior_lm = SegmentPriorLM(num_codes=vocab_size).to(device)

    # (In practice, you'd train the prior LM here)
    # For demo, we just compute compression stats

    total_bytes = sum(len(t.encode('utf-8')) for t in eval_texts[:100])
    total_tokens = len(all_tokens)
    compression_ratio = total_bytes / max(total_tokens, 1)

    print(f"  ✓ Bytes per token: {compression_ratio:.2f}")
    print(f"  ✓ Total tokens: {total_tokens:,}")

    # 5. Compare with BPE
    print("\n[5/5] Comparing with BPE baseline...")
    bpe_tokenizer = train_bpe_tokenizer(train_texts, vocab_size=vocab_size)
    bpe_wrapper = BPETokenizerWrapper(bpe_tokenizer, vocab_size)

    bpe_compression = bpe_wrapper.compute_compression_ratio(eval_texts[:100])
    print(f"  ✓ BPE bytes per token: {bpe_compression:.2f}")
    print(f"  ✓ ALSE bytes per token: {compression_ratio:.2f}")

    if compression_ratio > bpe_compression:
        improvement = (compression_ratio - bpe_compression) / bpe_compression * 100
        print(f"\n  🎉 ALSE achieves {improvement:.1f}% better compression!")

    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)


if __name__ == '__main__':
    main()
