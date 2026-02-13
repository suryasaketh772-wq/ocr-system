# OCR System

A complete Optical Character Recognition system built from scratch with Python. Supports Tesseract, EasyOCR, and a custom-trained deep learning model (CRNN) using PyTorch.

## Features

- **Image Preprocessing** — Grayscale, denoising, adaptive thresholding, deskewing (OpenCV)
- **Tesseract OCR** — Fast, accurate on clean documents
- **EasyOCR** — Deep learning based, robust on photos and noisy images
- **Custom CRNN Model** — CNN + BiLSTM + CTC built from scratch in PyTorch
- **FastAPI Web Server** — Upload images at `localhost:8000` and get extracted text
- **CLI Tool** — Extract text directly from the terminal
- **Train on Your Images** — Fine-tune the CRNN on real word crops from any image

## Project Structure

```
ocr_system/
├── ocr_engine/
│   ├── preprocessor.py          # Image preprocessing pipeline
│   └── recognizer.py            # Tesseract & EasyOCR wrappers
├── pytorch_ocr/
│   ├── model.py                 # CRNN architecture (CNN + BiLSTM + CTC)
│   ├── dataset.py               # Synthetic dataset generator
│   ├── train.py                 # Train on synthetic data
│   ├── train_from_image.py      # Train from extracted image text
│   ├── train_real_crops.py      # Train on real pixel crops (best results)
│   ├── predict.py               # Single-word inference
│   └── predict_full.py          # Full-page inference (detect + recognize)
├── server.py                    # FastAPI server + web upload interface
├── main.py                      # CLI entry point
├── test_ocr.py                  # Test suite
├── create_test_images.py        # Generate test images
├── images/                      # Sample test images
└── requirements.txt
```

## Setup

```bash
# Clone
git clone https://github.com/suryasaketh772-wq/ocr-system.git
cd ocr-system

# Install Tesseract (macOS)
brew install tesseract

# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### CLI — Extract text from an image

```bash
python main.py image.png                      # Tesseract (default)
python main.py image.png --engine easyocr     # EasyOCR
python main.py image.png --details            # Show bounding boxes + confidence
python main.py image.png --save output.txt    # Save to file
```

### Web Server — Upload via browser

```bash
python server.py
# Open http://localhost:8000
```

### API — curl

```bash
curl -X POST http://localhost:8000/extract-text -F "file=@image.png"
```

Response:
```json
{
  "extracted_text": "Hello OCR World",
  "engine": "tesseract",
  "filename": "image.png",
  "processing_time_seconds": 0.117
}
```

### Train Custom CRNN Model

```bash
# Train on synthetic data
python -m pytorch_ocr.train --epochs 20 --samples 5000

# Train on real word crops from an image (best accuracy)
python pytorch_ocr/train_real_crops.py your_image.png --epochs 30

# Run inference with trained model
python -m pytorch_ocr.predict_full your_model.pt your_image.png
```

### Run Tests

```bash
python test_ocr.py
```

## How It Works

```
┌─────────┐     ┌──────────────┐     ┌────────────┐     ┌────────┐
│  Image   │ ──► │ Preprocessor │ ──► │ Recognizer │ ──► │  Text  │
│  (file)  │     │ (OpenCV)     │     │ (OCR)      │     │ output │
└─────────┘     └──────────────┘     └────────────┘     └────────┘
```

1. **Load** — Read the image from disk
2. **Preprocess** — Grayscale → Denoise → Threshold → Deskew
3. **Recognize** — Feed clean image to Tesseract, EasyOCR, or custom CRNN
4. **Output** — Return extracted text via CLI, API, or web interface

### CRNN Architecture

```
Input Image (1 x 32 x W)
        │
   CNN Backbone (4 conv blocks) ── extracts visual features
        │
   BiLSTM (2 layers) ── captures left↔right context
        │
   Fully Connected ── maps to character probabilities
        │
   CTC Decoder ── converts probability sequence to text
```

## Tech Stack

- Python 3.13
- OpenCV — Image preprocessing
- Tesseract — Traditional OCR engine
- EasyOCR — Deep learning OCR
- PyTorch — Custom CRNN model
- FastAPI — Web server and API

## License

MIT
