"""
Full-page prediction with the trained CRNN model.
==================================================

For a full document/screenshot, we need a two-stage pipeline:

  Stage 1: TEXT DETECTION  — Find where each word is (bounding boxes)
           We use Tesseract for this since it's fast and reliable.

  Stage 2: TEXT RECOGNITION — Crop each word region and feed it to our CRNN.

This is exactly how production OCR works:
  EasyOCR  = CRAFT (detector)  + CRNN (recognizer)
  Our pipe = Tesseract (detector) + our trained CRNN (recognizer)

USAGE:
    python -m pytorch_ocr.predict_full <model.pt> <image>
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import pytesseract
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_ocr.model import CRNN, CharsetEncoder


def load_model(model_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    charset = checkpoint["charset"]
    num_classes = checkpoint["num_classes"]
    encoder = CharsetEncoder(charset)
    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, encoder


def detect_words(image_path: str) -> list[dict]:
    """
    Use Tesseract to detect word-level bounding boxes.

    Returns list of {"text": ..., "x": ..., "y": ..., "w": ..., "h": ..., "conf": ...}
    """
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        if text and conf > 30 and len(text) >= 2:
            words.append({
                "text": text,
                "x": data["left"][i],
                "y": data["top"][i],
                "w": data["width"][i],
                "h": data["height"][i],
                "conf": conf,
                "line_num": data["line_num"][i],
                "block_num": data["block_num"][i],
            })
    return words


def crop_and_preprocess(image: np.ndarray, bbox: dict, target_height: int = 32,
                        padding: int = 4) -> np.ndarray:
    """
    Crop a word region from the image and prepare it for the CRNN.

    Steps:
      1. Crop the bounding box (with some padding)
      2. Convert to grayscale
      3. Resize to target_height keeping aspect ratio
      4. Normalize to [0, 1]
    """
    h_img, w_img = image.shape[:2]

    x1 = max(0, bbox["x"] - padding)
    y1 = max(0, bbox["y"] - padding)
    x2 = min(w_img, bbox["x"] + bbox["w"] + padding)
    y2 = min(h_img, bbox["y"] + bbox["h"] + padding)

    crop = image[y1:y2, x1:x2]

    # To grayscale if needed
    if len(crop.shape) == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Resize to target height, keep aspect ratio
    ch, cw = crop.shape
    if ch == 0 or cw == 0:
        return None

    new_w = max(int(cw * (target_height / ch)), 16)
    crop = cv2.resize(crop, (new_w, target_height))

    return crop.astype(np.float32) / 255.0


def predict_word(model: CRNN, encoder: CharsetEncoder, image: np.ndarray,
                 device: torch.device) -> tuple[str, float]:
    """Run CRNN on a single cropped word image."""
    tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        log_probs = model(tensor)
        probs = torch.exp(log_probs)
        max_probs, pred_indices = probs.max(dim=2)
        pred_indices = pred_indices.squeeze(1).cpu().tolist()
        max_probs = max_probs.squeeze(1).cpu().tolist()

    text = encoder.decode(pred_indices)
    non_blank = [p for idx, p in zip(pred_indices, max_probs) if idx != 0]
    confidence = sum(non_blank) / len(non_blank) if non_blank else 0.0

    return text, confidence


def predict_full_image(model_path: str, image_path: str):
    """Full pipeline: detect words → crop → recognize with CRNN."""

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Loading model: {model_path}")
    model, encoder = load_model(model_path, device)

    print(f"Detecting words in: {image_path}")
    word_boxes = detect_words(image_path)
    print(f"Found {len(word_boxes)} word regions\n")

    # Load the raw image for cropping
    raw_image = cv2.imread(image_path)

    # Group by line for readable output
    lines = {}
    for wb in word_boxes:
        key = (wb["block_num"], wb["line_num"])
        if key not in lines:
            lines[key] = []
        lines[key].append(wb)

    # Sort words within each line by x position
    for key in lines:
        lines[key].sort(key=lambda w: w["x"])

    print(f"{'TESSERACT (detector)':30s} | {'CRNN (our model)':30s} | {'MATCH':5s}")
    print("-" * 75)

    total = 0
    matches = 0

    for key in sorted(lines.keys()):
        tess_words = []
        crnn_words = []

        for wb in lines[key]:
            crop = crop_and_preprocess(raw_image, wb)
            if crop is None:
                continue

            crnn_text, conf = predict_word(model, encoder, crop, device)
            tess_text = wb["text"]

            tess_words.append(tess_text)
            crnn_words.append(crnn_text)

            total += 1
            if crnn_text.lower() == tess_text.lower():
                matches += 1

        tess_line = " ".join(tess_words)
        crnn_line = " ".join(crnn_words)
        is_match = "==" if tess_line.lower() == crnn_line.lower() else "!="

        if tess_line.strip():
            print(f"{tess_line:30s} | {crnn_line:30s} | {is_match}")

    print("-" * 75)
    accuracy = (matches / total * 100) if total > 0 else 0
    print(f"\nWord-level accuracy: {matches}/{total} ({accuracy:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to trained model (.pt)")
    parser.add_argument("image", help="Path to input image")
    args = parser.parse_args()

    predict_full_image(args.model, args.image)
