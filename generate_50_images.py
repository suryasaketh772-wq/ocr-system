"""
Generate 50 diverse training images with varied text content.
"""

import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np


SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "Python is a powerful programming language",
    "Machine learning transforms raw data into insights",
    "Deep neural networks can recognize text in images",
    "Optical character recognition converts images to text",
    "Artificial intelligence is reshaping the world",
    "Computer vision enables machines to see and understand",
    "Natural language processing understands human text",
    "Data science combines statistics and programming",
    "Cloud computing provides scalable resources",
    "Open source software powers the modern internet",
    "Version control tracks changes in source code",
    "API endpoints connect frontend to backend services",
    "Database queries retrieve structured information",
    "Encryption protects sensitive user data",
    "Authentication verifies user identity securely",
    "Containers isolate applications for deployment",
    "Microservices break systems into smaller parts",
    "Continuous integration automates testing pipelines",
    "Agile development delivers software iteratively",
    "Transfer learning reuses pretrained model weights",
    "Convolutional networks extract spatial features",
    "Recurrent networks process sequential data",
    "Gradient descent optimizes model parameters",
    "Backpropagation computes gradients efficiently",
    "Batch normalization stabilizes neural network training",
    "Dropout regularization prevents model overfitting",
    "Learning rate controls the speed of training",
    "Loss functions measure prediction errors",
    "Activation functions add nonlinearity to networks",
    "Image preprocessing improves OCR accuracy",
    "Thresholding converts grayscale to binary images",
    "Noise reduction cleans up image artifacts",
    "Edge detection finds boundaries in images",
    "Feature extraction identifies important patterns",
    "Text detection locates words in photographs",
    "Character segmentation splits text into letters",
    "Language models predict the next word in sequence",
    "Tokenization breaks text into meaningful units",
    "Embeddings represent words as dense vectors",
    "Attention mechanism focuses on relevant parts",
    "Transformer architecture powers modern AI models",
    "GPU acceleration speeds up matrix operations",
    "Tensor operations form the basis of deep learning",
    "Model inference generates predictions from input",
    "Hyperparameter tuning improves model performance",
    "Cross validation estimates generalization ability",
    "Precision and recall measure classification quality",
    "Confusion matrix visualizes prediction results",
    "FastAPI serves machine learning models as endpoints",
]

FONTS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Times.ttc",
    "/System/Library/Fonts/Courier.dfont",
    "/System/Library/Fonts/Avenir.ttc",
    "/System/Library/Fonts/Georgia.ttf",
    "/System/Library/Fonts/Palatino.ttc",
    "/System/Library/Fonts/Menlo.ttc",
]


def get_font(size):
    for f in random.sample(FONTS, len(FONTS)):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def create_image(output_path: str, text: str, style: str = "clean"):
    """Create an image with the given text and style variant."""
    font_size = random.randint(24, 40)
    font = get_font(font_size)

    # Calculate image size based on text
    lines = text.split("\n")
    max_line = max(lines, key=len)
    width = max(len(max_line) * (font_size // 2 + 4), 400)
    height = max(len(lines) * (font_size + 20) + 60, 100)

    if style == "clean":
        bg_color = 255
        text_color = 0
    elif style == "dark":
        bg_color = 30
        text_color = 220
    elif style == "gray":
        bg_color = random.randint(200, 240)
        text_color = random.randint(10, 60)
    elif style == "tinted":
        bg_color = (random.randint(230, 255), random.randint(230, 255), random.randint(220, 245))
        text_color = (random.randint(0, 40), random.randint(0, 40), random.randint(0, 50))
    else:
        bg_color = 255
        text_color = 0

    if isinstance(bg_color, tuple):
        img = Image.new("RGB", (width, height), color=bg_color)
    else:
        img = Image.new("L", (width, height), color=bg_color)

    draw = ImageDraw.Draw(img)
    y = 20
    for line in lines:
        draw.text((20, y), line, fill=text_color, font=font)
        y += font_size + 16

    # Add noise for some styles
    if style in ("gray", "tinted"):
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(5, 15), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    img.save(output_path)


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "training_images"
    out_dir.mkdir(exist_ok=True)

    styles = ["clean", "clean", "gray", "tinted", "dark"]

    print("Generating 50 training images...")
    for i, sentence in enumerate(SENTENCES):
        style = styles[i % len(styles)]
        # Split long sentences into 2 lines
        words = sentence.split()
        if len(words) > 6:
            mid = len(words) // 2
            text = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
        else:
            text = sentence
        path = str(out_dir / f"img_{i+1:02d}.png")
        create_image(path, text, style)
        print(f"  [{i+1:2d}/50] {style:7s} | {sentence[:50]}")

    print(f"\nDone! 50 images saved to {out_dir}/")
