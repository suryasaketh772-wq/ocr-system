"""
OCR Recognizer
==============

This module wraps two OCR backends:

1. **Tesseract** (via pytesseract)
   - Classic OCR engine originally developed by HP, now maintained by Google
   - Uses a two-pass approach: first finds words, then recognizes characters
   - Very fast, works well on clean printed text
   - Needs good preprocessing to shine

2. **EasyOCR** (deep learning based)
   - Uses a CRAFT text detector + CRNN text recognizer (neural networks)
   - Better at handling messy real-world images (photos, signs, receipts)
   - Slower on CPU but more robust
   - Supports 80+ languages out of the box

WHEN TO USE WHICH:
  - Clean scanned documents → Tesseract (faster, accurate enough)
  - Photos / noisy images   → EasyOCR  (more robust)
  - Need maximum speed      → Tesseract
  - Need maximum accuracy   → EasyOCR
"""

import pytesseract
from PIL import Image
import numpy as np


class OCRRecognizer:
    """Extracts text from preprocessed images using Tesseract or EasyOCR."""

    def __init__(self, engine: str = "tesseract", languages: list[str] = None):
        """
        Initialize the recognizer.

        Args:
            engine:    "tesseract" or "easyocr"
            languages: List of language codes. Default: ["en"]
                       Tesseract codes: "eng", "fra", "deu", ...
                       EasyOCR codes:   "en",  "fr",  "de",  ...
        """
        self.engine = engine.lower()
        self.languages = languages
        self._easyocr_reader = None  # lazy-loaded (heavy import)

        if self.engine not in ("tesseract", "easyocr"):
            raise ValueError(f"Unknown engine: {engine}. Use 'tesseract' or 'easyocr'.")

    def _get_easyocr_reader(self):
        """
        Lazy-load the EasyOCR reader.

        WHY LAZY?  Importing easyocr downloads model weights (~100 MB) on
        first use and loads them into memory.  We only pay this cost when
        the user actually calls recognize() with the easyocr engine.
        """
        if self._easyocr_reader is None:
            import easyocr
            langs = self.languages or ["en"]
            self._easyocr_reader = easyocr.Reader(langs, gpu=False)
        return self._easyocr_reader

    def recognize(self, image: np.ndarray) -> str:
        """
        Run OCR on a preprocessed image array.

        Args:
            image: NumPy array (grayscale or binary) from the preprocessor

        Returns:
            Extracted text as a string
        """
        if self.engine == "tesseract":
            return self._recognize_tesseract(image)
        else:
            return self._recognize_easyocr(image)

    def _recognize_tesseract(self, image: np.ndarray) -> str:
        """
        Extract text using Tesseract.

        HOW TESSERACT WORKS (simplified):
        1. Connected Component Analysis — finds blobs of dark pixels
        2. Line Finding — groups blobs into text lines
        3. Word Recognition (Pass 1) — tries to recognize each word
        4. Word Recognition (Pass 2) — re-examines words using context
           from Pass 1 to fix uncertain characters

        The --psm flag controls "Page Segmentation Mode":
          PSM 3 = "Fully automatic page segmentation" (default, good for docs)
          PSM 6 = "Assume a single uniform block of text"
          PSM 7 = "Treat the image as a single text line"

        The --oem flag controls the OCR Engine Mode:
          OEM 3 = default (uses LSTM neural net + legacy engine)
        """
        lang = "+".join(self.languages) if self.languages else "eng"

        # Convert to PIL Image (pytesseract expects PIL or file path)
        pil_image = Image.fromarray(image)

        config = "--oem 3 --psm 3"
        text = pytesseract.image_to_string(pil_image, lang=lang, config=config)
        return text.strip()

    def _recognize_easyocr(self, image: np.ndarray) -> str:
        """
        Extract text using EasyOCR.

        HOW EASYOCR WORKS (simplified):
        1. CRAFT Text Detector (neural net)
           - Scans the image and predicts a "heatmap" of where characters are
           - Groups nearby characters into word-level bounding boxes

        2. CRNN Text Recognizer (neural net)
           - For each detected word region:
             a. CNN backbone extracts visual features
             b. RNN (BiLSTM) captures sequence context
             c. CTC decoder converts features → character sequence

        The result is a list of (bbox, text, confidence) tuples.
        """
        reader = self._get_easyocr_reader()

        results = reader.readtext(image)

        # results = [(bbox, text, confidence), ...]
        lines = [text for (_, text, conf) in results if conf > 0.2]
        return "\n".join(lines)

    def recognize_with_details(self, image: np.ndarray) -> list[dict]:
        """
        Get detailed results: text + bounding boxes + confidence scores.

        Useful for drawing boxes around detected text or filtering low-confidence
        results.

        Returns:
            List of dicts with keys: "text", "confidence", "bbox"
        """
        if self.engine == "tesseract":
            return self._details_tesseract(image)
        else:
            return self._details_easyocr(image)

    def _details_tesseract(self, image: np.ndarray) -> list[dict]:
        """Get per-word bounding boxes and confidence from Tesseract."""
        lang = "+".join(self.languages) if self.languages else "eng"
        pil_image = Image.fromarray(image)

        data = pytesseract.image_to_data(
            pil_image, lang=lang, output_type=pytesseract.Output.DICT
        )

        results = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            if text and conf > 0:
                results.append({
                    "text": text,
                    "confidence": conf / 100.0,
                    "bbox": {
                        "x": data["left"][i],
                        "y": data["top"][i],
                        "w": data["width"][i],
                        "h": data["height"][i],
                    },
                })
        return results

    def _details_easyocr(self, image: np.ndarray) -> list[dict]:
        """Get per-word bounding boxes and confidence from EasyOCR."""
        reader = self._get_easyocr_reader()
        results = reader.readtext(image)

        return [
            {
                "text": text,
                "confidence": round(conf, 3),
                "bbox": {
                    "x": int(min(p[0] for p in bbox)),
                    "y": int(min(p[1] for p in bbox)),
                    "w": int(max(p[0] for p in bbox) - min(p[0] for p in bbox)),
                    "h": int(max(p[1] for p in bbox) - min(p[1] for p in bbox)),
                },
            }
            for bbox, text, conf in results
            if conf > 0.2
        ]
