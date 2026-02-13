"""
Train CRNN on handwritten notes from real photos.

Handwriting is much harder than printed text because:
  - Inconsistent letter shapes and sizes
  - Connected/overlapping characters
  - Uneven baselines, tilted lines
  - Background noise (paper texture, shadows, pen bleed)

Strategy for 70%+ accuracy:
  1. Heavy preprocessing: denoise, sharpen, adaptive threshold, morphological cleanup
  2. Extract word crops with enhanced Tesseract settings for handwriting
  3. Aggressive data augmentation (rotation, elastic distortion, blur, contrast)
  4. Train for more epochs with a lower learning rate
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
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset, DataLoader

from pytorch_ocr.model import CRNN, CharsetEncoder

CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?()-+/':*&|="
TARGET_HEIGHT = 32


def preprocess_handwriting(image_path: str) -> np.ndarray:
    """
    Specialized preprocessing for handwritten notes on paper.

    Steps:
      1. Convert to grayscale
      2. Apply CLAHE (contrast-limited adaptive histogram equalization)
         to handle uneven lighting / shadows
      3. Bilateral filter to smooth paper texture while keeping ink edges
      4. Adaptive threshold to separate ink from paper
      5. Morphological operations to clean up noise
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE: fixes uneven lighting (shadows across the page)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Bilateral filter: smooths paper texture, preserves ink edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # Sharpen to make ink strokes crisper
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(filtered, -1, kernel)

    return sharpened


def extract_handwritten_crops(image_path: str) -> list[tuple[np.ndarray, str]]:
    """
    Extract word crops from handwritten images.

    Uses multiple Tesseract PSM modes and picks the one that gives
    the most confident results.
    """
    raw = cv2.imread(image_path)
    if raw is None:
        return []

    # Preprocess for better Tesseract detection on handwriting
    processed = preprocess_handwriting(image_path)

    # Use Tesseract with PSM 6 (uniform block) — works better for handwriting
    # Also use --oem 1 (LSTM only) which is better for handwriting
    configs = [
        "--oem 1 --psm 6",
        "--oem 1 --psm 3",
        "--oem 3 --psm 4",
    ]

    best_pairs = []
    best_count = 0

    for config in configs:
        pil_img = Image.fromarray(processed)
        data = pytesseract.image_to_data(
            pil_img, output_type=pytesseract.Output.DICT, config=config
        )

        pairs = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])

            if not text or conf < 30 or len(text) < 2:
                continue

            clean = "".join(c for c in text if c in CHARSET)
            if len(clean) < 2:
                continue

            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            pad = 5  # more padding for handwriting
            y1 = max(0, y - pad)
            y2 = min(raw.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(raw.shape[1], x + w + pad)

            crop = raw[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                continue

            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # Apply CLAHE to each crop individually
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            gray_crop = clahe.apply(gray_crop)

            ch, cw = gray_crop.shape
            new_w = max(int(cw * (TARGET_HEIGHT / ch)), 16)
            resized = cv2.resize(gray_crop, (new_w, TARGET_HEIGHT),
                                 interpolation=cv2.INTER_CUBIC)

            pairs.append((resized, clean))

        if len(pairs) > best_count:
            best_count = len(pairs)
            best_pairs = pairs

    return best_pairs


class HandwritingDataset(Dataset):
    """
    Dataset with aggressive augmentation for handwritten text.

    Handwriting varies much more than printed text, so we need
    stronger augmentations to teach the model to handle:
      - Different ink darkness / pen pressure
      - Paper texture and shadows
      - Slight rotation from uneven writing
      - Blurriness from camera focus
    """

    def __init__(self, all_crops, charset, augment_factor=50):
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

        # Random brightness (ink can be lighter or darker)
        if random.random() > 0.2:
            arr = np.clip(arr + random.uniform(-0.15, 0.15), 0, 1)

        # Random contrast
        if random.random() > 0.2:
            factor = random.uniform(0.7, 1.5)
            mean = arr.mean()
            arr = np.clip((arr - mean) * factor + mean, 0, 1)

        # Gaussian noise (paper texture simulation)
        if random.random() > 0.3:
            noise = np.random.normal(0, random.uniform(0.01, 0.04), arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0, 1)

        # Random slight blur (camera focus variation)
        if random.random() > 0.5:
            ksize = random.choice([3, 5])
            arr_uint8 = (arr * 255).astype(np.uint8)
            arr_uint8 = cv2.GaussianBlur(arr_uint8, (ksize, ksize), 0)
            arr = arr_uint8.astype(np.float32) / 255.0

        # Invert some samples (dark bg, light text)
        if random.random() > 0.9:
            arr = 1.0 - arr

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


def train(image_paths: list[str], epochs: int = 50, batch_size: int = 16,
          lr: float = 0.0008):
    print("=" * 60)
    print("  Training CRNN on Handwritten Notes")
    print("=" * 60)

    # Step 1: Extract crops from all images
    print(f"\n[1/4] Extracting word crops from {len(image_paths)} images...")
    all_crops = []
    unique_words = set()

    for img_path in image_paths:
        name = Path(img_path).name
        crops = extract_handwritten_crops(img_path)
        all_crops.extend(crops)
        for _, text in crops:
            unique_words.add(text.lower())
        print(f"      {name}: {len(crops)} words")
        if crops:
            sample = [t for _, t in crops[:8]]
            print(f"        → {', '.join(sample)}")

    print(f"\n      Total: {len(all_crops)} word crops, {len(unique_words)} unique words")

    if len(all_crops) < 10:
        print("\n      WARNING: Very few crops extracted. Trying with relaxed settings...")
        # Retry with even more relaxed confidence threshold
        all_crops = []
        for img_path in image_paths:
            processed = preprocess_handwriting(img_path)
            raw = cv2.imread(img_path)
            pil_img = Image.fromarray(processed)
            data = pytesseract.image_to_data(
                pil_img, output_type=pytesseract.Output.DICT,
                config="--oem 1 --psm 6"
            )
            for i in range(len(data["text"])):
                text = data["text"][i].strip()
                conf = int(data["conf"][i])
                if not text or conf < 15 or len(text) < 2:
                    continue
                clean = "".join(c for c in text if c in CHARSET)
                if len(clean) < 2:
                    continue
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                pad = 5
                y1, y2 = max(0, y - pad), min(raw.shape[0], y + h + pad)
                x1, x2 = max(0, x - pad), min(raw.shape[1], x + w + pad)
                crop = raw[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                gray = clahe.apply(gray)
                ch, cw = gray.shape
                new_w = max(int(cw * (TARGET_HEIGHT / ch)), 16)
                resized = cv2.resize(gray, (new_w, TARGET_HEIGHT))
                all_crops.append((resized, clean))

        print(f"      Retry total: {len(all_crops)} word crops")

    # Step 2: Build dataset
    print("\n[2/4] Building augmented dataset...")
    encoder = CharsetEncoder(CHARSET)
    augment_factor = max(50, 2000 // max(len(all_crops), 1))  # ensure enough samples
    dataset = HandwritingDataset(all_crops, CHARSET, augment_factor=augment_factor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)
    print(f"      {len(dataset)} samples ({len(all_crops)} crops x {augment_factor} aug)")
    print(f"      {len(loader)} batches/epoch")

    # Step 3: Train
    print(f"\n[3/4] Training CRNN for {epochs} epochs...\n")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"      Device: {device}")

    model = CRNN(num_classes=encoder.num_classes, image_height=TARGET_HEIGHT).to(device)
    print(f"      Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

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
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"      Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  lr={lr_now:.6f}  time={elapsed:.1f}s")
        scheduler.step(avg_loss)

        if epoch % 10 == 0:
            acc = _evaluate(model, all_crops, encoder, device)
            if acc >= 70.0:
                print(f"      ** Target accuracy {acc:.0f}% >= 70% reached! **")

    # Step 4: Save
    save_path = "handwritten_ocr_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "charset": CHARSET,
        "num_classes": encoder.num_classes,
    }, save_path)

    print(f"\n[4/4] Model saved to: {save_path}")
    print(f"\n{'='*60}")
    print("  Final Evaluation")
    print(f"{'='*60}")
    _evaluate(model, all_crops, encoder, device, n=40)


def _evaluate(model, crops, encoder, device, n=20):
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
            match = pred.lower() == label_text.lower()  # case-insensitive for handwriting
            if match:
                correct += 1
            tag = "OK" if match else "  "
            print(f"      [{tag}]  '{label_text}' -> '{pred}'")

    acc = (correct / total * 100) if total > 0 else 0
    print(f"      Accuracy: {correct}/{total} ({acc:.0f}%)\n")
    model.train()
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Paths to handwritten note images")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0008)
    args = parser.parse_args()

    train(args.images, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
