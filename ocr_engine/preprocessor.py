"""
Image Preprocessor for OCR
===========================

WHY PREPROCESS?
Raw images are messy — they have color noise, uneven lighting, skew, and
artifacts that confuse OCR engines.  Preprocessing cleans the image so the
text detector sees crisp black text on a white background.

Pipeline:  Load → Grayscale → Denoise → Threshold → Deskew → (optional resize)
"""

import cv2
import numpy as np
from pathlib import Path


class ImagePreprocessor:
    """Cleans and prepares images for OCR text extraction."""

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from disk.

        HOW IT WORKS:
        cv2.imread reads the file into a NumPy array of shape (H, W, 3) in BGR
        color order.  We convert to RGB so downstream tools (Pillow, matplotlib)
        display colors correctly.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not decode image: {image_path}")

        # BGR → RGB (OpenCV loads as BGR, most libraries expect RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert to single-channel grayscale.

        HOW IT WORKS:
        Each pixel becomes one brightness value instead of three color channels.
        Formula:  gray = 0.299*R + 0.587*G + 0.114*B
        This matches human perception — we're most sensitive to green.

        WHY:  OCR only cares about contrast between text and background,
              not color.  Reducing to 1 channel also speeds up every later step.
        """
        if len(image.shape) == 2:
            return image  # already grayscale
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove small speckles and sensor noise.

        HOW IT WORKS:
        fastNlMeansDenoising uses a "non-local means" algorithm:
          1. For each pixel, look at similar patches elsewhere in the image
          2. Average those similar patches together
          3. This removes random noise while keeping edges (text strokes) sharp

        Parameters:
          h=10          → filter strength (higher = more smoothing)
          templateWindowSize=7  → size of the patch to compare
          searchWindowSize=21   → how far to search for similar patches
        """
        return cv2.fastNlMeansDenoising(image, h=10)

    def threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Convert to pure black & white (binary image).

        HOW IT WORKS:
        Adaptive thresholding looks at a local neighborhood around each pixel
        and picks a threshold based on the local mean.  This handles uneven
        lighting — a shadow across the page won't ruin half the text.

        - ADAPTIVE_THRESH_GAUSSIAN_C  → weight neighbors by Gaussian bell curve
        - THRESH_BINARY               → pixel > threshold → white, else black
        - blockSize=11                → neighborhood is 11x11 pixels
        - C=2                         → subtract this constant from the mean

        WHY:  Binary images give the cleanest input to OCR — every pixel is
              either "ink" or "paper", no ambiguity.
        """
        return cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Straighten slightly rotated text.

        HOW IT WORKS:
        1. Find all non-zero (black text) pixel coordinates
        2. Fit the tightest rotated rectangle around them (minAreaRect)
        3. The rectangle's angle tells us how much the text is tilted
        4. Rotate the entire image by the negative of that angle

        WHY:  Even 2-3 degrees of tilt can significantly hurt OCR accuracy.
              Tesseract especially struggles with skewed lines.
        """
        coords = np.column_stack(np.where(image > 0))
        if len(coords) < 10:
            return image  # not enough content to measure skew

        angle = cv2.minAreaRect(coords)[-1]

        # minAreaRect returns angles in [-90, 0); normalize to small correction
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # skip tiny rotations (< 0.5 degrees)
        if abs(angle) < 0.5:
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image, rotation_matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def preprocess(self, image_path: str) -> np.ndarray:
        """
        Run the full preprocessing pipeline.

        Returns a clean binary image ready for OCR.
        """
        image = self.load_image(image_path)
        gray = self.to_grayscale(image)
        denoised = self.remove_noise(gray)
        binary = self.threshold(denoised)
        straightened = self.deskew(binary)
        return straightened

    def preprocess_for_easyocr(self, image_path: str) -> np.ndarray:
        """
        Lighter preprocessing for EasyOCR.

        EasyOCR has its own built-in preprocessing, so we only do
        grayscale + denoise and let EasyOCR handle the rest.
        """
        image = self.load_image(image_path)
        gray = self.to_grayscale(image)
        denoised = self.remove_noise(gray)
        return denoised
