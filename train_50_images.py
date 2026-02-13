"""
Train CRNN on real word crops from all 50 training images.

Combines crops from every image into one large dataset, augments them,
and trains the model to recognize diverse text styles.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import pytesseract
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from pytorch_ocr.model import CRNN, CharsetEncoder

import random

CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?()-+/'"
TARGET_HEIGHT = 32


def extract_crops_from_image(image_path: str) -> list[tuple[np.ndarray, str]]:
    """Extract (crop, label) pairs from a single image using Tesseract."""
    raw = cv2.imread(image_path)
    if raw is None:
        return []

    pil_img = Image.open(image_path)
    # Convert to RGB if grayscale (Tesseract handles both)
    if pil_img.mode == "L":
        pil_img = pil_img.convert("RGB")

    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)

    pairs = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        if not text or conf < 50 or len(text) < 2:
            continue

        clean = "".join(c for c in text if c in CHARSET)
        if len(clean) < 2:
            continue

        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        pad = 3
        y1, y2 = max(0, y - pad), min(raw.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(raw.shape[1], x + w + pad)

        crop = raw[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        if len(crop.shape) == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        ch, cw = crop.shape
        new_w = max(int(cw * (TARGET_HEIGHT / ch)), 16)
        resized = cv2.resize(crop, (new_w, TARGET_HEIGHT))
        pairs.append((resized, clean))

    return pairs


class MultiImageDataset(Dataset):
    def __init__(self, all_crops, charset, augment_factor=20):
        self.encoder = CharsetEncoder(charset)
        self.samples = []
        for img, text in all_crops:
            label = self.encoder.encode(text)
            if len(label) == 0:
                continue
            for _ in range(augment_factor):
                self.samples.append((img.copy(), label, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label, _ = self.samples[idx]
        img = self._augment(img)
        return torch.FloatTensor(img).unsqueeze(0), torch.IntTensor(label), len(label)

    def _augment(self, img):
        arr = img.astype(np.float32) / 255.0
        if random.random() > 0.3:
            arr = np.clip(arr + random.uniform(-0.1, 0.1), 0, 1)
        if random.random() > 0.3:
            factor = random.uniform(0.8, 1.3)
            mean = arr.mean()
            arr = np.clip((arr - mean) * factor + mean, 0, 1)
        if random.random() > 0.4:
            arr = np.clip(arr + np.random.normal(0, 0.02, arr.shape).astype(np.float32), 0, 1)
        return arr


def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    max_w = max(img.shape[2] for img in images)
    h = images[0].shape[1]
    padded = []
    for img in images:
        w = img.shape[2]
        if w < max_w:
            img = torch.cat([img, torch.ones(1, h, max_w - w)], dim=2)
        padded.append(img)
    return torch.stack(padded), torch.cat(labels), torch.IntTensor(lengths)


def main():
    print("=" * 60)
    print("  Training CRNN on 50 images")
    print("=" * 60)

    # Step 1: Extract crops from all images
    print("\n[1/4] Extracting word crops from 50 images...")
    img_dir = Path("training_images")
    all_crops = []
    unique_words = set()

    for img_path in sorted(img_dir.glob("*.png")):
        crops = extract_crops_from_image(str(img_path))
        all_crops.extend(crops)
        for _, text in crops:
            unique_words.add(text.lower())
        print(f"      {img_path.name}: {len(crops)} words")

    print(f"\n      Total: {len(all_crops)} word crops, {len(unique_words)} unique words")

    # Step 2: Build dataset
    print("\n[2/4] Building augmented dataset...")
    encoder = CharsetEncoder(CHARSET)
    dataset = MultiImageDataset(all_crops, CHARSET, augment_factor=20)
    loader = DataLoader(dataset, batch_size=32, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)
    print(f"      {len(dataset)} samples, {len(loader)} batches/epoch")

    # Step 3: Train
    print("\n[3/4] Training CRNN...\n")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"      Device: {device}")

    model = CRNN(num_classes=encoder.num_classes, image_height=TARGET_HEIGHT).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"      Parameters: {params:,}\n")

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    epochs = 30
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

        if epoch % 10 == 0:
            _evaluate(model, all_crops, encoder, device)

    # Step 4: Save
    save_path = "ocr_50img_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "charset": CHARSET,
        "num_classes": encoder.num_classes,
    }, save_path)

    print(f"\n[4/4] Model saved to: {save_path}")
    print(f"\n{'='*60}")
    print("  Final Evaluation")
    print(f"{'='*60}")
    _evaluate(model, all_crops, encoder, device, n=30)


def _evaluate(model, crops, encoder, device, n=15):
    model.eval()
    correct = 0
    total = 0
    seen = set()

    with torch.no_grad():
        for img, label_text in crops:
            if label_text in seen:
                continue
            seen.add(label_text)
            if total >= n:
                break

            arr = img.astype(np.float32) / 255.0
            tensor = torch.FloatTensor(arr).unsqueeze(0).unsqueeze(0).to(device)
            log_probs = model(tensor)
            pred = encoder.decode(log_probs.argmax(dim=2).squeeze(1).cpu().tolist())

            total += 1
            match = pred == label_text
            if match:
                correct += 1
            tag = "OK" if match else "  "
            print(f"      [{tag}]  '{label_text}' -> '{pred}'")

    acc = (correct / total * 100) if total > 0 else 0
    print(f"      Accuracy: {correct}/{total} ({acc:.0f}%)\n")
    model.train()


if __name__ == "__main__":
    main()
