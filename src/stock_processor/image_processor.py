"""Traditional image processing - sharpen, denoise, resize, color conversion."""

import numpy as np
import cv2
from PIL import Image


def process_image(
    image: np.ndarray,
    min_dimension: int = 4000,
    sharpen_amount: float = 1.0,
    denoise_strength: int = 5,
) -> np.ndarray:
    """Apply traditional image processing pipeline.

    Args:
        image: Input image as numpy array (RGB)
        min_dimension: Minimum size for longest edge
        sharpen_amount: Sharpening intensity (0 = none, 1 = normal)
        denoise_strength: Denoising strength (0 = none)

    Returns:
        Processed image as numpy array (RGB, sRGB color space)
    """
    # Ensure we're working with the right format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Step 1: Denoise (if enabled)
    if denoise_strength > 0:
        image = _denoise(image, denoise_strength)

    # Step 2: Resize to meet minimum dimension
    image = _resize(image, min_dimension)

    # Step 3: Sharpen (if enabled)
    if sharpen_amount > 0:
        image = _sharpen(image, sharpen_amount)

    # Step 4: Ensure sRGB color space
    image = _ensure_srgb(image)

    return image


def _denoise(image: np.ndarray, strength: int) -> np.ndarray:
    """Apply non-local means denoising."""
    # Convert to BGR for OpenCV
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Non-local means denoising - good for preserving edges
    # h parameter controls filter strength
    denoised = cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        strength,       # h (luminance)
        strength,       # hColor (color components)
        7,              # templateWindowSize
        21,             # searchWindowSize
    )

    return cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)


def _resize(image: np.ndarray, min_dimension: int) -> np.ndarray:
    """Resize image so longest edge meets minimum dimension."""
    h, w = image.shape[:2]
    longest = max(h, w)

    if longest >= min_dimension:
        # Already large enough
        return image

    # Calculate scale factor
    scale = min_dimension / longest
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Use LANCZOS for high-quality upscaling
    pil_img = Image.fromarray(image)
    resized = pil_img.resize((new_w, new_h), Image.LANCZOS)

    return np.array(resized)


def _sharpen(image: np.ndarray, amount: float) -> np.ndarray:
    """Apply unsharp mask sharpening."""
    # Convert to BGR for OpenCV
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Create gaussian blurred version
    blurred = cv2.GaussianBlur(bgr, (0, 0), 3)

    # Unsharp mask: original + amount * (original - blurred)
    sharpened = cv2.addWeighted(bgr, 1 + amount, blurred, -amount, 0)

    # Clip values to valid range
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)


def _ensure_srgb(image: np.ndarray) -> np.ndarray:
    """Ensure image is in sRGB color space.

    Note: If the image was loaded from a RAW file with rawpy's sRGB output,
    it should already be in sRGB. This function is a placeholder for more
    sophisticated color management if needed.
    """
    # For now, assume image is already in sRGB from our RAW processing
    # or from a standard JPEG/PNG which are typically sRGB
    return image
