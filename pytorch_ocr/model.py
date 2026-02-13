"""
Custom Deep Learning OCR Model — CRNN (CNN + RNN + CTC)
=======================================================

This is the architecture used by most modern OCR systems (including EasyOCR).
Here we build it from scratch so you understand every layer.

ARCHITECTURE OVERVIEW:
    ┌──────────────────────────────────────────────────────────────┐
    │  Input Image  (1 x 32 x 128)  — grayscale, fixed height    │
    └────────────────────┬─────────────────────────────────────────┘
                         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  CNN Backbone  — extracts visual features from each column   │
    │  (Conv → BN → ReLU → Pool) × 4 layers                       │
    │  Output: (512 × 1 × 32)  →  512 features at 32 time steps   │
    └────────────────────┬─────────────────────────────────────────┘
                         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  Sequence Reshaping  — treat width-axis as a time sequence   │
    │  (512 × 1 × 32) → (32 × 512)   (seq_len × features)        │
    └────────────────────┬─────────────────────────────────────────┘
                         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  BiLSTM  — captures left↔right context along the sequence    │
    │  2 layers, 256 hidden units each direction                   │
    │  Output: (32 × 512)                                          │
    └────────────────────┬─────────────────────────────────────────┘
                         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  Fully Connected  — maps features → character probabilities  │
    │  Output: (32 × num_classes)                                  │
    └────────────────────┬─────────────────────────────────────────┘
                         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  CTC Decoder  — converts probability sequence → text string  │
    │  Handles variable-length output without needing alignment    │
    └──────────────────────────────────────────────────────────────┘

WHY CRNN?
  - CNN alone can recognize isolated characters but can't handle variable
    length words or capture context (is it "rn" or "m"?)
  - RNN alone can't extract visual features from raw pixels
  - CRNN combines both: CNN sees, RNN reads
  - CTC loss lets us train without manually aligning each character
    to its position in the image
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for OCR.

    Args:
        num_classes: Number of output characters (alphabet size + 1 for CTC blank)
        image_height: Fixed input image height (default 32)
        hidden_size: LSTM hidden units per direction (default 256)
    """

    def __init__(self, num_classes: int, image_height: int = 32, hidden_size: int = 256):
        super().__init__()

        # ── CNN Backbone ──────────────────────────────────────────
        # Each block: Conv2d → BatchNorm → ReLU → MaxPool
        #
        # WHY BatchNorm?  Normalizes activations so training is faster
        #                 and more stable (each layer sees consistent input).
        #
        # WHY MaxPool?    Reduces spatial dimensions. We pool more in height
        #                 than width because we want to collapse the height
        #                 to 1 but preserve the width (time axis).

        self.cnn = nn.Sequential(
            # Block 1:  (1, 32, 128) → (64, 16, 64)
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2:  (64, 16, 64) → (128, 8, 32)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3:  (128, 8, 32) → (256, 4, 32)   [pool only height]
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Block 4:  (256, 4, 32) → (512, 2, 32)   [pool only height]
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        # After CNN: feature map is (batch, 512, H', W')
        # We need H' = 1 so we can squeeze it into a sequence.
        # With input height=32 and four 2x pooling on height: 32/2/2/2/2 = 2
        # So we add an adaptive pool to collapse height to 1:
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))  # (512, 1, W')

        # ── RNN (BiLSTM) ─────────────────────────────────────────
        # Reads the feature sequence left-to-right AND right-to-left.
        #
        # WHY Bidirectional?  When reading "hello", recognizing 'l' is
        #   easier if you know 'e' came before AND 'o' comes after.
        #
        # Input:  (seq_len, batch, 512)
        # Output: (seq_len, batch, hidden_size * 2)  ← both directions

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
        )

        # ── Classifier Head ──────────────────────────────────────
        # Maps LSTM output to character probabilities at each time step.
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images, shape (batch, 1, 32, width)

        Returns:
            Log-probabilities, shape (seq_len, batch, num_classes)
            Ready for CTC loss.
        """
        # CNN feature extraction
        conv = self.cnn(x)                        # (B, 512, H', W')
        conv = self.adaptive_pool(conv)           # (B, 512, 1, W')
        conv = conv.squeeze(2)                    # (B, 512, W')

        # Reshape for RNN: (B, C, W) → (W, B, C)   [seq_len, batch, features]
        conv = conv.permute(2, 0, 1)

        # Bidirectional LSTM
        rnn_out, _ = self.rnn(conv)               # (W, B, hidden*2)

        # Character classification at each time step
        output = self.fc(rnn_out)                 # (W, B, num_classes)

        # Log-softmax for CTC loss
        return torch.nn.functional.log_softmax(output, dim=2)


# ── Character Encoding ────────────────────────────────────────────

class CharsetEncoder:
    """
    Maps characters ↔ integer indices for the model.

    Index 0 is reserved for the CTC "blank" token (represents no character /
    repeated character boundary).

    Example:
        encoder = CharsetEncoder("abcdefghijklmnopqrstuvwxyz0123456789")
        encoder.encode("hello")  → [8, 5, 12, 12, 15]
        encoder.decode([8, 5, 12, 12, 15])  → "hello"
    """

    def __init__(self, charset: str):
        self.charset = charset
        # 0 = CTC blank, characters start at index 1
        self.char_to_idx = {c: i + 1 for i, c in enumerate(charset)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(charset)}
        self.blank_idx = 0

    @property
    def num_classes(self) -> int:
        """Total classes = characters + 1 blank."""
        return len(self.charset) + 1

    def encode(self, text: str) -> list[int]:
        """Convert text string → list of integer indices."""
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode(self, indices: list[int]) -> str:
        """
        CTC decode: collapse repeated indices and remove blanks.

        Example:  [0, 8, 8, 5, 0, 12, 12, 15] → "helo"
                  (blank, h, h, e, blank, l, l, o)
                  After collapsing repeats: h, e, l, o → "helo"

        This is "greedy CTC decoding" — takes the most probable character
        at each time step. More advanced: beam search decoding.
        """
        result = []
        prev = self.blank_idx
        for idx in indices:
            if idx != prev and idx != self.blank_idx:
                result.append(self.idx_to_char.get(idx, ""))
            prev = idx
        return "".join(result)
