"""
Inference with the trained CRNN model.
======================================

Load a saved model and run it on real images.

USAGE:
    python -m pytorch_ocr.predict ocr_model.pt images/simple.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from .model import CRNN, CharsetEncoder


def load_model(model_path: str, device: torch.device) -> tuple[CRNN, CharsetEncoder]:
    """Load a trained CRNN model from a checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    charset = checkpoint["charset"]
    num_classes = checkpoint["num_classes"]
    encoder = CharsetEncoder(charset)

    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, encoder


def preprocess_for_crnn(image_path: str, target_height: int = 32) -> np.ndarray:
    """
    Prepare an image for the CRNN model.

    Steps:
      1. Load as grayscale
      2. Resize to target_height while keeping aspect ratio
      3. Normalize pixel values to [0, 1]
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    # Resize keeping aspect ratio
    h, w = img.shape
    new_w = int(w * (target_height / h))
    new_w = max(new_w, 32)  # minimum width
    img = cv2.resize(img, (new_w, target_height))

    # Normalize to [0, 1]
    return img.astype(np.float32) / 255.0


def predict(model: CRNN, encoder: CharsetEncoder, image: np.ndarray,
            device: torch.device) -> tuple[str, float]:
    """
    Run inference on a single preprocessed image.

    Returns:
        (predicted_text, average_confidence)
    """
    # Add batch and channel dimensions: (H, W) → (1, 1, H, W)
    tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        log_probs = model(tensor)  # (seq_len, 1, num_classes)

        # Greedy decode: take the most probable character at each time step
        probs = torch.exp(log_probs)
        max_probs, pred_indices = probs.max(dim=2)

        pred_indices = pred_indices.squeeze(1).cpu().tolist()
        max_probs = max_probs.squeeze(1).cpu().tolist()

    # Decode indices to text
    text = encoder.decode(pred_indices)

    # Average confidence (excluding blanks)
    non_blank_probs = [p for idx, p in zip(pred_indices, max_probs) if idx != 0]
    avg_conf = sum(non_blank_probs) / len(non_blank_probs) if non_blank_probs else 0.0

    return text, avg_conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR prediction with trained CRNN")
    parser.add_argument("model", help="Path to saved model (.pt)")
    parser.add_argument("image", help="Path to input image")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model, encoder = load_model(args.model, device)
    image = preprocess_for_crnn(args.image)
    text, confidence = predict(model, encoder, image, device)

    print(f"Predicted: '{text}'")
    print(f"Confidence: {confidence:.2%}")
