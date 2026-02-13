"""
OCR System — Main Entry Point
==============================

Usage:
    python main.py <image_path>                    # Tesseract (default)
    python main.py <image_path> --engine easyocr   # EasyOCR
    python main.py <image_path> --details          # Show bounding boxes + confidence
    python main.py <image_path> --save output.txt  # Save result to file

HOW THE SYSTEM WORKS (big picture):
    ┌─────────┐     ┌──────────────┐     ┌────────────┐     ┌────────┐
    │  Image   │ ──► │ Preprocessor │ ──► │ Recognizer │ ──► │  Text  │
    │  (file)  │     │ (OpenCV)     │     │ (OCR)      │     │ output │
    └─────────┘     └──────────────┘     └────────────┘     └────────┘

    1. LOAD      — Read the image file from disk
    2. PREPROCESS — Grayscale → Denoise → Threshold → Deskew
    3. RECOGNIZE  — Feed the clean image to Tesseract or EasyOCR
    4. OUTPUT     — Print or save the extracted text
"""

import argparse
import sys
from pathlib import Path

from ocr_engine.preprocessor import ImagePreprocessor
from ocr_engine.recognizer import OCRRecognizer


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from images using OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python main.py images/simple.png\n"
               "  python main.py images/noisy.png --engine easyocr\n"
               "  python main.py images/multiline.png --details\n",
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--engine", choices=["tesseract", "easyocr"], default="tesseract",
        help="OCR engine to use (default: tesseract)",
    )
    parser.add_argument(
        "--details", action="store_true",
        help="Show word-level bounding boxes and confidence scores",
    )
    parser.add_argument(
        "--save", metavar="FILE",
        help="Save extracted text to a file",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # --- Step 1 & 2: Preprocess ---
    print(f"[1/3] Loading and preprocessing: {image_path}")
    preprocessor = ImagePreprocessor()

    if args.engine == "easyocr":
        processed = preprocessor.preprocess_for_easyocr(str(image_path))
    else:
        processed = preprocessor.preprocess(str(image_path))

    print(f"      Image shape: {processed.shape}  dtype: {processed.dtype}")

    # --- Step 3: Recognize ---
    print(f"[2/3] Running OCR with engine: {args.engine}")
    recognizer = OCRRecognizer(engine=args.engine)

    if args.details:
        results = recognizer.recognize_with_details(processed)
        print(f"[3/3] Found {len(results)} text regions:\n")
        for r in results:
            b = r["bbox"]
            print(f"  '{r['text']}'  conf={r['confidence']:.2f}  "
                  f"@ ({b['x']}, {b['y']}, {b['w']}x{b['h']})")
        text = " ".join(r["text"] for r in results)
    else:
        text = recognizer.recognize(processed)
        print(f"[3/3] Extracted text:\n")
        print(text)

    # --- Step 4: Save ---
    if args.save:
        Path(args.save).write_text(text, encoding="utf-8")
        print(f"\nSaved to: {args.save}")


if __name__ == "__main__":
    main()
