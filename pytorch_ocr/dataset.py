"""
OCR Dataset & Data Loading
===========================

To train our CRNN, we need pairs of (image, text_label).

Two approaches:
  1. SYNTHETIC DATA  — Generate images with random text (great for starting)
  2. REAL DATA       — Use datasets like IAM Handwriting, ICDAR, COCO-Text

This module implements both a synthetic generator and a generic dataset loader.

IMAGE PREPARATION:
  - Fixed height (32px), variable width (we'll pad to a max width in batches)
  - Grayscale (1 channel)
  - Normalized to [0, 1] range

WHY FIXED HEIGHT?
  The CNN needs consistent spatial dimensions. Height is fixed at 32px
  (a good balance between detail and speed). Width varies because words
  have different lengths — we handle this with padding in the collate function.
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont

from .model import CharsetEncoder


class SyntheticOCRDataset(Dataset):
    """
    Generates synthetic word images on the fly.

    This is surprisingly effective for training! Many production OCR systems
    start with synthetic pre-training then fine-tune on real data.

    Each __getitem__ call:
      1. Picks a random word from the vocabulary
      2. Renders it as an image with random font size / slight noise
      3. Returns (image_tensor, encoded_label, label_length)
    """

    def __init__(
        self,
        charset: str,
        num_samples: int = 10000,
        image_height: int = 32,
        max_text_len: int = 20,
    ):
        self.encoder = CharsetEncoder(charset)
        self.num_samples = num_samples
        self.image_height = image_height
        self.max_text_len = max_text_len
        self.charset = charset

        # Sample words to render
        self.words = self._generate_word_list()

    def _generate_word_list(self) -> list[str]:
        """Create random words from the charset."""
        words = []
        common = [
            "hello", "world", "python", "deep", "learning", "image",
            "text", "neural", "network", "data", "model", "train",
            "test", "ocr", "read", "write", "code", "open", "file",
            "computer", "vision", "torch", "batch", "loss", "epoch",
        ]
        words.extend(common * 20)

        # Add random strings for diversity
        for _ in range(self.num_samples - len(words)):
            length = random.randint(3, self.max_text_len)
            word = "".join(random.choices(self.charset, k=length))
            words.append(word)

        return words[:self.num_samples]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        text = self.words[idx]

        # Render text to image
        image = self._render_text(text)

        # Encode label
        label = self.encoder.encode(text)

        # Convert to tensors
        image_tensor = torch.FloatTensor(image).unsqueeze(0)  # (1, H, W)
        label_tensor = torch.IntTensor(label)

        return image_tensor, label_tensor, len(label)

    def _render_text(self, text: str) -> np.ndarray:
        """Render text string as a grayscale numpy array."""
        font_size = random.randint(20, 28)

        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Helvetica.ttc", font_size
            )
        except OSError:
            font = ImageFont.load_default()

        # Estimate image width from text length
        width = max(len(text) * font_size, 128)
        img = Image.new("L", (width, self.image_height), color=255)
        draw = ImageDraw.Draw(img)

        # Center text vertically
        y = max(0, (self.image_height - font_size) // 2 - 2)
        draw.text((4, y), text, fill=0, font=font)

        # Convert to numpy and normalize to [0, 1]
        arr = np.array(img, dtype=np.float32) / 255.0

        return arr


def collate_fn(batch):
    """
    Custom collate: pad images to same width within a batch.

    WHY CUSTOM COLLATE?
    Images have different widths (different word lengths). PyTorch's default
    collate requires all tensors to have the same shape. We pad shorter
    images with white (1.0) on the right side.

    Returns:
        images:        (batch, 1, H, max_W)     — padded image batch
        labels:        (total_label_chars,)      — all labels concatenated
        label_lengths: (batch,)                  — length of each label
    """
    images, labels, label_lengths = zip(*batch)

    # Find max width in this batch
    max_width = max(img.shape[2] for img in images)
    height = images[0].shape[1]

    # Pad all images to max_width
    padded = []
    for img in images:
        w = img.shape[2]
        if w < max_width:
            pad = torch.ones(1, height, max_width - w)  # white padding
            img = torch.cat([img, pad], dim=2)
        padded.append(img)

    images_tensor = torch.stack(padded, dim=0)
    labels_tensor = torch.cat(labels, dim=0)
    label_lengths_tensor = torch.IntTensor(label_lengths)

    return images_tensor, labels_tensor, label_lengths_tensor
