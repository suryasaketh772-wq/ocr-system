"""
Train CRNN on REAL cropped word images from the input photo.
=============================================================

The previous approach rendered words synthetically with Helvetica font.
That doesn't match the actual pixels in a screenshot.

This script:
  1. Uses Tesseract to detect word bounding boxes + labels
  2. Crops each word region from the real image
  3. Augments the real crops (slight rotation, noise, contrast)
  4. Trains the CRNN on these real pixel patterns

This is how real-world OCR fine-tuning works:
  "Show the model what the text ACTUALLY looks like in your domain."
"""

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import pytesseract
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_ocr.model import CRNN, CharsetEncoder


CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?()-+/'"
TARGET_HEIGHT = 32


def extract_word_crops(image_path: str) -> list[tuple[np.ndarray, str]]:
    """
    Extract (cropped_image, label) pairs from the photo using Tesseract.

    This is semi-supervised: Tesseract provides the labels (which may have
    some errors), but the images are the REAL pixel crops from the photo.
    """
    raw = cv2.imread(image_path)
    pil_img = Image.open(image_path)
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)

    pairs = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])

        # Only use high-confidence detections as labels
        if not text or conf < 60 or len(text) < 2:
            continue

        # Filter to charset
        clean = "".join(c for c in text if c in CHARSET)
        if len(clean) < 2:
            continue

        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        pad = 3
        y1 = max(0, y - pad)
        y2 = min(raw.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(raw.shape[1], x + w + pad)

        crop = raw[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Convert to grayscale + resize to target height
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ch, cw = gray.shape
        new_w = max(int(cw * (TARGET_HEIGHT / ch)), 16)
        resized = cv2.resize(gray, (new_w, TARGET_HEIGHT))

        pairs.append((resized, clean))

    return pairs


class RealCropDataset(Dataset):
    """
    Dataset of real word crops with data augmentation.

    Each crop is augmented multiple times to increase training diversity:
      - Random brightness/contrast shifts
      - Slight Gaussian noise
      - Small random erosion/dilation (thickens/thins text strokes)
    """

    def __init__(self, crops: list[tuple[np.ndarray, str]], charset: str,
                 augment_factor: int = 30):
        self.encoder = CharsetEncoder(charset)
        self.samples = []

        for img, text in crops:
            label = self.encoder.encode(text)
            if len(label) == 0:
                continue
            for _ in range(augment_factor):
                self.samples.append((img.copy(), label, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label, text = self.samples[idx]
        img = self._augment(img)
        img_tensor = torch.FloatTensor(img).unsqueeze(0)  # (1, H, W)
        label_tensor = torch.IntTensor(label)
        return img_tensor, label_tensor, len(label)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a grayscale image."""
        arr = img.astype(np.float32) / 255.0

        # Random brightness shift
        if random.random() > 0.3:
            shift = random.uniform(-0.1, 0.1)
            arr = np.clip(arr + shift, 0, 1)

        # Random contrast adjustment
        if random.random() > 0.3:
            factor = random.uniform(0.8, 1.3)
            mean = arr.mean()
            arr = np.clip((arr - mean) * factor + mean, 0, 1)

        # Gaussian noise
        if random.random() > 0.4:
            noise = np.random.normal(0, 0.02, arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0, 1)

        return arr


def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    max_w = max(img.shape[2] for img in images)
    h = images[0].shape[1]

    padded = []
    for img in images:
        w = img.shape[2]
        if w < max_w:
            pad = torch.ones(1, h, max_w - w)
            img = torch.cat([img, pad], dim=2)
        padded.append(img)

    return (
        torch.stack(padded, 0),
        torch.cat(labels, 0),
        torch.IntTensor(label_lengths),
    )


def train(image_path: str, epochs: int = 30, batch_size: int = 16, lr: float = 0.001):
    print(f"{'='*60}")
    print(f"  Training CRNN on real crops from: {Path(image_path).name}")
    print(f"{'='*60}\n")

    # Step 1: Extract real word crops
    print("[1/4] Extracting word crops from image...")
    crops = extract_word_crops(image_path)
    print(f"      Extracted {len(crops)} high-confidence word crops")
    print(f"      Sample words: {', '.join(t for _, t in crops[:15])}")
    print()

    # Step 2: Build dataset
    print("[2/4] Building augmented dataset...")
    encoder = CharsetEncoder(CHARSET)
    dataset = RealCropDataset(crops, CHARSET, augment_factor=30)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)
    print(f"      {len(dataset)} training samples ({len(crops)} crops x 30 augmentations)")
    print(f"      {len(loader)} batches per epoch\n")

    # Step 3: Train
    print("[3/4] Training...\n")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"      Device: {device}")

    model = CRNN(num_classes=encoder.num_classes, image_height=TARGET_HEIGHT).to(device)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        for images, labels, label_lengths in loader:
            images = images.to(device)
            log_probs = model(images)

            seq_len = log_probs.shape[0]
            input_lengths = torch.full((images.shape[0],), seq_len, dtype=torch.int32)

            loss = ctc_loss(
                log_probs.cpu(), labels.cpu(),
                input_lengths.cpu(), label_lengths.cpu(),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - start
        print(f"      Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  time={elapsed:.1f}s")
        scheduler.step(avg_loss)

        if epoch % 5 == 0:
            _evaluate(model, crops, encoder, device)

    # Step 4: Save
    save_path = str(Path(image_path).stem) + "_real_ocr_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "charset": CHARSET,
        "num_classes": encoder.num_classes,
    }, save_path)

    print(f"\n[4/4] Model saved to: {save_path}")
    print(f"\n{'='*60}")
    print("  Final Evaluation on Real Crops")
    print(f"{'='*60}")
    _evaluate(model, crops, encoder, device, n=25)

    return model


def _evaluate(model, crops, encoder, device, n=10):
    """Test the model on the actual crops from the image."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for img, label_text in crops[:n]:
            arr = img.astype(np.float32) / 255.0
            tensor = torch.FloatTensor(arr).unsqueeze(0).unsqueeze(0).to(device)

            log_probs = model(tensor)
            pred_indices = log_probs.argmax(dim=2).squeeze(1).cpu().tolist()
            pred_text = encoder.decode(pred_indices)

            total += 1
            match = pred_text == label_text
            if match:
                correct += 1

            tag = "OK" if match else "  "
            print(f"      [{tag}]  expected='{label_text}' → predicted='{pred_text}'")

    acc = (correct / total * 100) if total > 0 else 0
    print(f"      Accuracy: {correct}/{total} ({acc:.0f}%)\n")
    model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train(args.image, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
