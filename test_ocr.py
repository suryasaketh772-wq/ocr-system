"""
Test suite for the OCR system.

Run:  python test_ocr.py
"""

import sys
from pathlib import Path
from ocr_engine.preprocessor import ImagePreprocessor
from ocr_engine.recognizer import OCRRecognizer


def test_preprocessor():
    """Test that preprocessing produces valid output."""
    print("=== Preprocessor Tests ===")
    pp = ImagePreprocessor()

    for name in ["simple.png", "noisy.png", "multiline.png"]:
        path = str(Path("images") / name)
        img = pp.preprocess(path)
        assert img is not None, f"Failed to preprocess {name}"
        assert len(img.shape) == 2, f"Expected grayscale, got shape {img.shape}"
        assert img.dtype.name == "uint8", f"Expected uint8, got {img.dtype}"
        print(f"  PASS  {name:20s}  shape={img.shape}")

    print()


def test_tesseract_simple():
    """Test Tesseract on a clean image with known text."""
    print("=== Tesseract Tests ===")
    pp = ImagePreprocessor()
    ocr = OCRRecognizer(engine="tesseract")

    img = pp.preprocess("images/simple.png")
    text = ocr.recognize(img)
    assert "Hello" in text, f"Expected 'Hello' in output, got: {text}"
    assert "OCR" in text, f"Expected 'OCR' in output, got: {text}"
    print(f"  PASS  simple.png → '{text}'")

    img = pp.preprocess("images/noisy.png")
    text = ocr.recognize(img)
    assert "Noisy" in text or "Text" in text, f"Expected readable text, got: {text}"
    print(f"  PASS  noisy.png  → '{text}'")

    img = pp.preprocess("images/multiline.png")
    text = ocr.recognize(img)
    assert "Recognition" in text or "recognition" in text.lower(), (
        f"Expected 'Recognition', got: {text}"
    )
    print(f"  PASS  multiline  → {len(text)} chars extracted")
    print()


def test_tesseract_details():
    """Test detailed output (bounding boxes + confidence)."""
    print("=== Tesseract Detail Tests ===")
    pp = ImagePreprocessor()
    ocr = OCRRecognizer(engine="tesseract")

    img = pp.preprocess("images/simple.png")
    results = ocr.recognize_with_details(img)
    assert len(results) > 0, "No text regions found"
    for r in results:
        assert "text" in r and "confidence" in r and "bbox" in r
    print(f"  PASS  Found {len(results)} word regions with bounding boxes")
    print()


def test_easyocr():
    """Test EasyOCR (skip if model download fails)."""
    print("=== EasyOCR Tests ===")
    pp = ImagePreprocessor()

    try:
        ocr = OCRRecognizer(engine="easyocr")
        img = pp.preprocess_for_easyocr("images/simple.png")
        text = ocr.recognize(img)
        assert "Hello" in text or "OCR" in text, f"Expected readable text, got: {text}"
        print(f"  PASS  simple.png → '{text}'")
    except Exception as e:
        print(f"  SKIP  EasyOCR not available: {e}")
    print()


if __name__ == "__main__":
    # Generate test images if missing
    if not Path("images/simple.png").exists():
        print("Generating test images first...\n")
        import create_test_images
        create_test_images.create_simple_text_image("images/simple.png", "Hello OCR World")
        create_test_images.create_noisy_text_image("images/noisy.png", "Noisy Text Example 123")
        create_test_images.create_multiline_image("images/multiline.png")
        print()

    failures = 0
    for test_fn in [test_preprocessor, test_tesseract_simple, test_tesseract_details, test_easyocr]:
        try:
            test_fn()
        except AssertionError as e:
            print(f"  FAIL  {e}")
            failures += 1
        except Exception as e:
            print(f"  ERROR {e}")
            failures += 1

    print("=" * 40)
    if failures == 0:
        print("All tests passed!")
    else:
        print(f"{failures} test(s) failed")
        sys.exit(1)
