"""
Training Script for Custom CRNN OCR Model
==========================================

TRAINING LOOP EXPLAINED:

    For each epoch:
        For each batch of (images, labels):
            1. Forward pass  — feed images through CRNN, get predictions
            2. Compute loss  — CTC loss measures prediction vs ground truth
            3. Backward pass — compute gradients (how to adjust weights)
            4. Update weights — optimizer steps in the direction that reduces loss

CTC LOSS (Connectionist Temporal Classification):
    The key insight: we don't know which character maps to which image column.
    CTC considers ALL possible alignments between the prediction sequence and
    the target text, sums their probabilities, and optimizes that total.

    Example for "cat":
      Valid alignments:  c-a-t--  or  -cc-at-  or  cca-a-t  etc.
      (where '-' is the blank token)
    CTC says: "I don't care which alignment you use, just make sure the total
    probability of all valid alignments for 'cat' is high."

USAGE:
    python -m pytorch_ocr.train                    # train with defaults
    python -m pytorch_ocr.train --epochs 50        # more epochs
    python -m pytorch_ocr.train --lr 0.0005        # lower learning rate
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import CRNN, CharsetEncoder
from .dataset import SyntheticOCRDataset, collate_fn


# Default character set: lowercase + digits + common punctuation
DEFAULT_CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789 .-,"


def train(
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 0.001,
    num_samples: int = 5000,
    charset: str = DEFAULT_CHARSET,
    save_path: str = "ocr_model.pt",
):
    """Train the CRNN model on synthetic data."""

    # ── Setup ─────────────────────────────────────────────────
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    encoder = CharsetEncoder(charset)
    print(f"Charset: {len(charset)} characters → {encoder.num_classes} classes (with blank)")

    # ── Dataset ───────────────────────────────────────────────
    dataset = SyntheticOCRDataset(
        charset=charset,
        num_samples=num_samples,
        image_height=32,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # 0 for macOS compatibility
    )
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")

    # ── Model ─────────────────────────────────────────────────
    model = CRNN(num_classes=encoder.num_classes, image_height=32).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters")

    # ── Loss & Optimizer ──────────────────────────────────────
    # CTC Loss:
    #   - blank=0         → index 0 is the CTC blank token
    #   - zero_infinity   → prevents NaN when a prediction is too short
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    # Adam optimizer: adaptive learning rate per parameter
    # WHY Adam?  It adjusts the learning rate for each weight individually
    #   based on the history of gradients. Works well out of the box.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler: reduce LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # ── Training Loop ─────────────────────────────────────────
    print(f"\nTraining for {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        for batch_idx, (images, labels, label_lengths) in enumerate(dataloader):
            images = images.to(device)

            # Forward pass: images → log probabilities at each time step
            log_probs = model(images)  # (seq_len, batch, num_classes)

            # Input lengths: how many time steps the model produced
            # (same for all images in batch since they're padded to same width)
            seq_len = log_probs.shape[0]
            input_lengths = torch.full(
                (images.shape[0],), seq_len, dtype=torch.int32
            )

            # CTC loss needs: log_probs, labels, input_lengths, label_lengths
            # Note: CTC loss is not yet supported on MPS, so we compute it on CPU
            loss = ctc_loss(
                log_probs.cpu(), labels.cpu(),
                input_lengths.cpu(), label_lengths.cpu(),
            )

            # Backward pass & weight update
            optimizer.zero_grad()   # clear old gradients
            loss.backward()         # compute new gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()        # update weights

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  "
              f"lr={current_lr:.6f}  time={elapsed:.1f}s")

        scheduler.step(avg_loss)

        # Quick accuracy check every 5 epochs
        if epoch % 5 == 0:
            _eval_sample(model, dataset, encoder, device)

    # ── Save Model ────────────────────────────────────────────
    save_path = str(Path(save_path))
    torch.save({
        "model_state_dict": model.state_dict(),
        "charset": charset,
        "num_classes": encoder.num_classes,
    }, save_path)
    print(f"\nModel saved to: {save_path}")

    return model


def _eval_sample(model, dataset, encoder, device, n=5):
    """Evaluate on a few samples to show training progress."""
    model.eval()
    print("  Sample predictions:")

    with torch.no_grad():
        for i in range(n):
            image, label, _ = dataset[i]
            image = image.unsqueeze(0).to(device)  # add batch dim

            log_probs = model(image)  # (seq_len, 1, num_classes)
            pred_indices = log_probs.argmax(dim=2).squeeze(1).cpu().tolist()
            pred_text = encoder.decode(pred_indices)

            true_text = dataset.words[i]
            match = "ok" if pred_text == true_text else "  "
            print(f"    [{match}]  true='{true_text}'  pred='{pred_text}'")

    model.train()
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CRNN OCR model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--save", default="ocr_model.pt")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_samples=args.samples,
        save_path=args.save,
    )
