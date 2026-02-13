"""
Train the CRNN model using text extracted from a real image.

This script:
  1. Extracts text from your image using Tesseract
  2. Uses those words as the vocabulary for synthetic training data
  3. Trains the CRNN to recognize those words
  4. Tests the trained model on crops from the original image
"""

import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_engine.preprocessor import ImagePreprocessor
from ocr_engine.recognizer import OCRRecognizer
from pytorch_ocr.model import CRNN, CharsetEncoder
from pytorch_ocr.dataset import collate_fn


CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?()-+/'"


class ImageTextDataset(Dataset):
    """Generate synthetic images from words extracted from a real image."""

    def __init__(self, words: list[str], charset: str, samples_per_word: int = 50,
                 image_height: int = 32):
        self.encoder = CharsetEncoder(charset)
        self.image_height = image_height
        self.charset = charset

        # Repeat words to build the dataset
        self.words = []
        for word in words:
            # Filter to charset-safe characters
            clean = "".join(c for c in word if c in charset)
            if len(clean) >= 2:
                self.words.extend([clean] * samples_per_word)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        text = self.words[idx]
        image = self._render(text)
        label = self.encoder.encode(text)
        return (
            torch.FloatTensor(image).unsqueeze(0),
            torch.IntTensor(label),
            len(label),
        )

    def _render(self, text: str) -> np.ndarray:
        import random
        font_size = random.randint(18, 28)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except OSError:
            font = ImageFont.load_default()

        width = max(len(text) * font_size, 128)
        img = Image.new("L", (width, self.image_height), color=255)
        draw = ImageDraw.Draw(img)
        y = max(0, (self.image_height - font_size) // 2 - 2)
        draw.text((4, y), text, fill=0, font=font)

        arr = np.array(img, dtype=np.float32) / 255.0

        # Random augmentation: slight noise
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.03, arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0, 1)

        return arr


def extract_words(image_path: str) -> list[str]:
    """Extract words from the image using Tesseract."""
    pp = ImagePreprocessor()
    ocr = OCRRecognizer(engine="tesseract")
    processed = pp.preprocess(image_path)
    text = ocr.recognize(processed)

    # Split into individual words, clean up
    words = re.findall(r'[A-Za-z0-9]+(?:[\'.-][A-Za-z0-9]+)*', text)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for w in words:
        if w.lower() not in seen and len(w) >= 2:
            seen.add(w.lower())
            unique.append(w)
    return unique


def train_on_image(image_path: str, epochs: int = 20, batch_size: int = 16,
                   lr: float = 0.001):
    """Full pipeline: extract → train → evaluate."""

    print(f"{'='*60}")
    print(f"  Training CRNN on text from: {image_path}")
    print(f"{'='*60}\n")

    # Step 1: Extract words
    print("[1/4] Extracting text from image...")
    words = extract_words(image_path)
    print(f"      Found {len(words)} unique words:")
    for i in range(0, len(words), 8):
        chunk = words[i:i+8]
        print(f"        {', '.join(chunk)}")
    print()

    # Step 2: Build dataset
    print("[2/4] Building synthetic training dataset...")
    encoder = CharsetEncoder(CHARSET)
    dataset = ImageTextDataset(words, CHARSET, samples_per_word=40)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=0)
    print(f"      {len(dataset)} training samples, {len(dataloader)} batches/epoch\n")

    # Step 3: Train
    print("[3/4] Training CRNN model...\n")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"      Device: {device}")

    model = CRNN(num_classes=encoder.num_classes, image_height=32).to(device)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        for images, labels, label_lengths in dataloader:
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

        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start
        print(f"      Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  time={elapsed:.1f}s")
        scheduler.step(avg_loss)

        if epoch % 5 == 0:
            _show_predictions(model, dataset, encoder, device)

    # Step 4: Save
    save_path = str(Path(image_path).stem) + "_ocr_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "charset": CHARSET,
        "num_classes": encoder.num_classes,
    }, save_path)
    print(f"\n[4/4] Model saved to: {save_path}")

    # Final evaluation
    print(f"\n{'='*60}")
    print("  Final Predictions")
    print(f"{'='*60}")
    _show_predictions(model, dataset, encoder, device, n=15)

    return model


def _show_predictions(model, dataset, encoder, device, n=8):
    model.eval()
    shown = set()
    count = 0

    with torch.no_grad():
        for i in range(len(dataset)):
            if count >= n:
                break
            word = dataset.words[i]
            if word in shown:
                continue
            shown.add(word)

            image, label, _ = dataset[i]
            image = image.unsqueeze(0).to(device)
            log_probs = model(image)
            pred_indices = log_probs.argmax(dim=2).squeeze(1).cpu().tolist()
            pred_text = encoder.decode(pred_indices)

            match = "OK" if pred_text == word else "  "
            print(f"      [{match}]  expected='{word}'  predicted='{pred_text}'")
            count += 1

    model.train()
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train_on_image(args.image, epochs=args.epochs, batch_size=args.batch_size,
                   lr=args.lr)
