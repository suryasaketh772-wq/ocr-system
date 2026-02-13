"""
Generate test images for the OCR system.

Creates synthetic images with known text so we can verify our OCR pipeline
produces the expected output.
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def create_simple_text_image(output_path: str, text: str, font_size: int = 40):
    """Create a clean image with black text on white background."""
    # Create a white canvas
    img = Image.new("RGB", (800, 200), color="white")
    draw = ImageDraw.Draw(img)

    # Try to use a nice font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except OSError:
        font = ImageFont.load_default()

    draw.text((30, 60), text, fill="black", font=font)
    img.save(output_path)
    print(f"  Created: {output_path}")


def create_noisy_text_image(output_path: str, text: str, font_size: int = 36):
    """Create a slightly noisy image to test preprocessing."""
    import numpy as np

    img = Image.new("RGB", (800, 200), color=(240, 235, 230))  # off-white
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except OSError:
        font = ImageFont.load_default()

    # Dark gray text (not pure black — simulates a photo)
    draw.text((30, 60), text, fill=(30, 30, 40), font=font)

    # Add Gaussian noise
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 12, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    Image.fromarray(arr).save(output_path)
    print(f"  Created: {output_path}")


def create_multiline_image(output_path: str):
    """Create a multi-line document-style image."""
    lines = [
        "Optical Character Recognition",
        "converts images of text into",
        "machine-readable strings.",
        "",
        "Built with Python + OpenCV.",
    ]
    img = Image.new("RGB", (800, 400), color="white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
    except OSError:
        font = ImageFont.load_default()

    y = 30
    for line in lines:
        draw.text((30, y), line, fill="black", font=font)
        y += 55

    img.save(output_path)
    print(f"  Created: {output_path}")


if __name__ == "__main__":
    img_dir = Path(__file__).parent / "images"
    img_dir.mkdir(exist_ok=True)

    print("Generating test images...")
    create_simple_text_image(str(img_dir / "simple.png"), "Hello OCR World")
    create_noisy_text_image(str(img_dir / "noisy.png"), "Noisy Text Example 123")
    create_multiline_image(str(img_dir / "multiline.png"))
    print("Done! Test images saved to images/")
