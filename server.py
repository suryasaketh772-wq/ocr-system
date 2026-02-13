"""
OCR API Server (FastAPI)
========================

A local web server that accepts image uploads and returns extracted text.

ENDPOINTS:
    GET  /           → Web interface (HTML upload page)
    POST /extract-text → Upload image, get OCR text back as JSON

HOW IT WORKS:
    1. User uploads an image via the web form or curl
    2. Server saves it to a temp file
    3. Preprocessing pipeline cleans the image (grayscale, denoise, threshold)
    4. OCR engine (Tesseract by default, or our trained CRNN) extracts text
    5. Server returns JSON: {"extracted_text": "...", "engine": "...", "confidence": ...}

USAGE:
    python server.py                          # Start on port 8000
    python server.py --port 9000              # Custom port
    python server.py --engine easyocr         # Use EasyOCR instead
    python server.py --model ocr_model.pt     # Use your trained CRNN model

Then open http://localhost:8000 in your browser.
"""

import argparse
import logging
import tempfile
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from ocr_engine.preprocessor import ImagePreprocessor
from ocr_engine.recognizer import OCRRecognizer

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ocr-server")

# ── App Setup ─────────────────────────────────────────────────
app = FastAPI(
    title="OCR System API",
    description="Upload an image, get extracted text back.",
    version="1.0.0",
)

# These get set at startup based on CLI args
preprocessor = ImagePreprocessor()
recognizer: OCRRecognizer | None = None
crnn_predictor: dict | None = None  # for custom model mode

# Allowed image extensions
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


# ── HTML Web Interface ────────────────────────────────────────

WEB_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR Text Extractor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: #0f0f0f;
            color: #e0e0e0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 20px;
        }
        h1 {
            font-size: 2rem;
            margin-bottom: 8px;
            color: #fff;
        }
        .subtitle {
            color: #888;
            margin-bottom: 32px;
            font-size: 0.95rem;
        }
        .container {
            width: 100%;
            max-width: 700px;
        }
        .upload-zone {
            border: 2px dashed #333;
            border-radius: 12px;
            padding: 48px 24px;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.2s, background 0.2s;
            background: #1a1a1a;
        }
        .upload-zone:hover, .upload-zone.dragover {
            border-color: #4a9eff;
            background: #1a2a3a;
        }
        .upload-zone p { color: #888; margin-top: 8px; }
        .upload-icon { font-size: 2.5rem; color: #4a9eff; }
        input[type="file"] { display: none; }
        .preview-container {
            margin-top: 20px;
            display: none;
        }
        .preview-container.active { display: block; }
        .preview-img {
            max-width: 100%;
            max-height: 300px;
            border-radius: 8px;
            border: 1px solid #333;
        }
        .btn {
            display: inline-block;
            margin-top: 16px;
            padding: 12px 32px;
            background: #4a9eff;
            color: #fff;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            transition: background 0.2s;
        }
        .btn:hover { background: #3a8eef; }
        .btn:disabled { background: #333; color: #666; cursor: not-allowed; }
        .result-box {
            margin-top: 24px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 12px;
            padding: 24px;
            display: none;
        }
        .result-box.active { display: block; }
        .result-box h3 { color: #4a9eff; margin-bottom: 12px; }
        .result-text {
            background: #111;
            padding: 16px;
            border-radius: 8px;
            font-family: 'SF Mono', 'Menlo', monospace;
            font-size: 0.9rem;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 400px;
            overflow-y: auto;
            color: #ddd;
        }
        .meta {
            margin-top: 12px;
            font-size: 0.8rem;
            color: #666;
        }
        .spinner {
            display: none;
            margin-top: 20px;
            text-align: center;
            color: #4a9eff;
        }
        .spinner.active { display: block; }
        .error { color: #ff6b6b; }
    </style>
</head>
<body>
    <h1>OCR Text Extractor</h1>
    <p class="subtitle">Upload an image to extract text using AI</p>

    <div class="container">
        <!-- Upload Zone -->
        <div class="upload-zone" id="dropZone">
            <div class="upload-icon">&#128196;</div>
            <p><strong>Click to upload</strong> or drag and drop</p>
            <p style="font-size: 0.85rem">PNG, JPG, BMP, TIFF, WebP</p>
            <input type="file" id="fileInput" accept="image/*">
        </div>

        <!-- Image Preview -->
        <div class="preview-container" id="previewContainer">
            <img class="preview-img" id="previewImg">
            <button class="btn" id="extractBtn" onclick="extractText()">
                Extract Text
            </button>
        </div>

        <!-- Loading -->
        <div class="spinner" id="spinner">Processing image...</div>

        <!-- Result -->
        <div class="result-box" id="resultBox">
            <h3>Extracted Text</h3>
            <div class="result-text" id="resultText"></div>
            <div class="meta" id="resultMeta"></div>
        </div>
    </div>

    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const previewContainer = document.getElementById('previewContainer');
        const previewImg = document.getElementById('previewImg');
        const spinner = document.getElementById('spinner');
        const resultBox = document.getElementById('resultBox');
        const resultText = document.getElementById('resultText');
        const resultMeta = document.getElementById('resultMeta');
        const extractBtn = document.getElementById('extractBtn');
        let selectedFile = null;

        // Click to upload
        dropZone.addEventListener('click', () => fileInput.click());

        // Drag and drop
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        function handleFile(file) {
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                previewContainer.classList.add('active');
                resultBox.classList.remove('active');
            };
            reader.readAsDataURL(file);
        }

        async function extractText() {
            if (!selectedFile) return;

            extractBtn.disabled = true;
            spinner.classList.add('active');
            resultBox.classList.remove('active');

            const formData = new FormData();
            formData.append('file', selectedFile);

            try {
                const startTime = Date.now();
                const response = await fetch('/extract-text', {
                    method: 'POST',
                    body: formData,
                });
                const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Server error');
                }

                const data = await response.json();
                resultText.textContent = data.extracted_text || '(no text found)';
                resultMeta.textContent =
                    `Engine: ${data.engine} | Processing: ${elapsed}s`;
                resultBox.classList.add('active');
            } catch (error) {
                resultText.innerHTML = `<span class="error">${error.message}</span>`;
                resultMeta.textContent = '';
                resultBox.classList.add('active');
            } finally {
                extractBtn.disabled = false;
                spinner.classList.remove('active');
            }
        }
    </script>
</body>
</html>
"""


# ── Routes ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def web_interface():
    """Serve the upload page."""
    return WEB_PAGE


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    """
    Accept an image upload and return extracted text.

    Request:  POST /extract-text  with multipart form file
    Response: {"extracted_text": "...", "engine": "...", "filename": "..."}
    """
    # Validate file type
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Use: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Save upload to temp file
    start = time.time()
    log.info(f"Received image: {file.filename} ({file.content_type})")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Preprocess
        log.info("Preprocessing image...")
        if crnn_predictor:
            # Custom model mode: use CRNN full-page pipeline
            text = _predict_crnn(tmp_path)
            engine = "crnn (custom trained)"
        elif recognizer.engine == "easyocr":
            processed = preprocessor.preprocess_for_easyocr(tmp_path)
            text = recognizer.recognize(processed)
            engine = "easyocr"
        else:
            # Use light preprocessing for Tesseract: grayscale + denoise only.
            # The full pipeline (threshold + deskew) destroys detail in
            # screenshots, photos, and colored images. Tesseract works best
            # with clean grayscale input — it handles binarization internally.
            image = preprocessor.load_image(tmp_path)
            gray = preprocessor.to_grayscale(image)
            denoised = preprocessor.remove_noise(gray)
            text = recognizer.recognize(denoised)
            engine = "tesseract"

        elapsed = time.time() - start
        log.info(f"OCR complete in {elapsed:.2f}s — extracted {len(text)} chars")

        return JSONResponse({
            "extracted_text": text,
            "engine": engine,
            "filename": file.filename,
            "processing_time_seconds": round(elapsed, 3),
        })

    except Exception as e:
        log.error(f"OCR failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _predict_crnn(image_path: str) -> str:
    """Run the trained CRNN model on a full image using Tesseract for detection."""
    import cv2
    import numpy as np
    import torch
    import pytesseract
    from PIL import Image as PILImage

    model = crnn_predictor["model"]
    encoder = crnn_predictor["encoder"]
    device = crnn_predictor["device"]

    raw = cv2.imread(image_path)
    pil_img = PILImage.open(image_path)
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)

    lines = {}
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        if text and conf > 30:
            key = (data["block_num"][i], data["line_num"][i])
            if key not in lines:
                lines[key] = []
            lines[key].append({
                "x": data["left"][i], "y": data["top"][i],
                "w": data["width"][i], "h": data["height"][i],
            })

    result_lines = []
    for key in sorted(lines.keys()):
        word_preds = []
        for bbox in sorted(lines[key], key=lambda b: b["x"]):
            pad = 3
            y1 = max(0, bbox["y"] - pad)
            y2 = min(raw.shape[0], bbox["y"] + bbox["h"] + pad)
            x1 = max(0, bbox["x"] - pad)
            x2 = min(raw.shape[1], bbox["x"] + bbox["w"] + pad)

            crop = raw[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            ch, cw = gray.shape
            new_w = max(int(cw * (32 / ch)), 16)
            resized = cv2.resize(gray, (new_w, 32)).astype(np.float32) / 255.0

            tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                log_probs = model(tensor)
                pred_indices = log_probs.argmax(dim=2).squeeze(1).cpu().tolist()
                pred_text = encoder.decode(pred_indices)

            if pred_text.strip():
                word_preds.append(pred_text)

        if word_preds:
            result_lines.append(" ".join(word_preds))

    return "\n".join(result_lines)


# ── Startup ───────────────────────────────────────────────────

def start_server(engine: str = "tesseract", model_path: str = None,
                 host: str = "0.0.0.0", port: int = 8000):
    """Initialize OCR engine and start the server."""
    global recognizer, crnn_predictor

    if model_path:
        # Load custom CRNN model
        import torch
        from pytorch_ocr.model import CRNN, CharsetEncoder

        log.info(f"Loading custom CRNN model: {model_path}")
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        charset = checkpoint["charset"]
        encoder = CharsetEncoder(charset)
        model = CRNN(num_classes=checkpoint["num_classes"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        crnn_predictor = {"model": model, "encoder": encoder, "device": device}
        log.info(f"CRNN model loaded on {device}")
    else:
        recognizer = OCRRecognizer(engine=engine)
        log.info(f"OCR engine: {engine}")

    log.info(f"Starting server on http://localhost:{port}")
    log.info(f"Web interface: http://localhost:{port}")
    log.info(f"API endpoint:  POST http://localhost:{port}/extract-text")

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR API Server")
    parser.add_argument("--engine", choices=["tesseract", "easyocr"],
                        default="tesseract", help="OCR engine (default: tesseract)")
    parser.add_argument("--model", metavar="FILE",
                        help="Path to trained CRNN model (.pt) — overrides --engine")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    start_server(engine=args.engine, model_path=args.model,
                 host=args.host, port=args.port)
