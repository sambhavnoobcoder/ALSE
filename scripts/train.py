"""
ALSE Training Script

Train ALSE models with configurable hyperparameters
"""

import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm

from alse.models import ALSEV3
from alse.utils import load_wikitext2, prepare_byte_sequences


def train_alse(vocab_size, epochs, learning_rate, batch_size, device='cpu'):
    """
    Train ALSE model

    Args:
        vocab_size: Vocabulary size
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        device: Device to train on

    Returns:
        model: Trained model
        history: Training history
    """
    print(f"\n{'='*60}")
    print(f"Training ALSE (vocab={vocab_size})")
    print(f"{'='*60}")

    # Load data
    train_texts, eval_texts = load_wikitext2(num_examples=1000)

    # Initialize model
    model = ALSEV3(
        target_compression=4.0,
        num_codes=vocab_size,
        code_dim=12
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [],
        'recon_loss': [],
        'vq_loss': [],
        'compression': [],
    }

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        epoch_recon = []
        epoch_vq = []
        epoch_comp = []

        # Progress bar
        pbar = tqdm(range(0, len(train_texts), batch_size),
                   desc=f"Epoch {epoch+1}/{epochs}")

        for i in pbar:
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            model.current_step += 1

            # Track metrics
            epoch_losses.append(loss.item())
            epoch_recon.append(outputs['recon_loss'])
            epoch_vq.append(outputs['vq_loss'])
            epoch_comp.append(outputs['actual_compression'])

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'comp': f"{outputs['actual_compression']:.2f}"
            })

        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_comp = sum(epoch_comp) / len(epoch_comp)

        history['train_loss'].append(avg_loss)
        history['recon_loss'].append(sum(epoch_recon) / len(epoch_recon))
        history['vq_loss'].append(sum(epoch_vq) / len(epoch_vq))
        history['compression'].append(avg_comp)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Compression={avg_comp:.2f}")

    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train ALSE')
    parser.add_argument('--vocab-size', type=int, default=128,
                       help='Vocabulary size')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu/cuda)')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                       help='Output directory for checkpoints')

    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    # Train model
    model, history = train_alse(
        vocab_size=args.vocab_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        device=device
    )

    # Save checkpoint
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    checkpoint_path = output_dir / f"alse_v{args.vocab_size}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': args.vocab_size,
        'history': history,
    }, checkpoint_path)

    print(f"\n✓ Model saved to {checkpoint_path}")

    # Save history
    history_path = output_dir / f"history_v{args.vocab_size}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"✓ History saved to {history_path}")


if __name__ == '__main__':
    main()
