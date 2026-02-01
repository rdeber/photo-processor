"""Traditional image processing - sharpen, denoise, resize, color conversion."""

import numpy as np
import cv2
from PIL import Image


def process_image(
    image: np.ndarray,
    min_dimension: int = 4000,
    sharpen_amount: float = 1.0,
    denoise_strength: int = 5,
    auto_white_balance: bool = True,
    auto_exposure: bool = True,
    target_brightness: float = 0.45,
    contrast_strength: float = 1.1,
    straighten_mode: str = "auto",
) -> np.ndarray:
    """Apply traditional image processing pipeline.

    Args:
        image: Input image as numpy array (RGB)
        min_dimension: Minimum size for longest edge
        sharpen_amount: Sharpening intensity (0 = none, 1 = normal)
        denoise_strength: Denoising strength (0 = none)
        auto_white_balance: Apply gray world white balance correction
        auto_exposure: Automatically adjust brightness and contrast
        target_brightness: Target mean brightness (0-1 scale, default 0.45)
        contrast_strength: Contrast multiplier (1.0 = no change, >1 = more contrast)
        straighten_mode: Geometry correction mode - "auto", "horizontal", "vertical", or "none"

    Returns:
        Processed image as numpy array (RGB, sRGB color space)
    """
    # Ensure we're working with the right format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Step 1: Auto straighten (before other processing to avoid artifacts)
    if straighten_mode != "none":
        image = _auto_straighten(image, mode=straighten_mode)

    # Step 2: Auto white balance (before other adjustments)
    if auto_white_balance:
        image = _auto_white_balance(image)

    # Step 3: Auto exposure (brightness and contrast)
    if auto_exposure:
        image = _auto_brightness_contrast(image, target_brightness, contrast_strength)

    # Step 4: Denoise (if enabled)
    if denoise_strength > 0:
        image = _denoise(image, denoise_strength)

    # Step 5: Resize to meet minimum dimension
    image = _resize(image, min_dimension)

    # Step 6: Sharpen (if enabled)
    if sharpen_amount > 0:
        image = _sharpen(image, sharpen_amount)

    # Step 7: Ensure sRGB color space
    image = _ensure_srgb(image)

    return image


def _auto_white_balance(image: np.ndarray) -> np.ndarray:
    """Apply gray world white balance correction.

    The gray world algorithm assumes the average color of a scene should be
    neutral gray. It scales each color channel to achieve this balance.
    """
    # Calculate mean of each channel
    r_mean = np.mean(image[:, :, 0])
    g_mean = np.mean(image[:, :, 1])
    b_mean = np.mean(image[:, :, 2])

    # Calculate overall mean (gray target)
    gray_mean = (r_mean + g_mean + b_mean) / 3

    # Avoid division by zero
    if r_mean == 0 or g_mean == 0 or b_mean == 0:
        return image

    # Calculate scaling factors
    r_scale = gray_mean / r_mean
    g_scale = gray_mean / g_mean
    b_scale = gray_mean / b_mean

    # Apply scaling with clipping
    result = image.astype(np.float32)
    result[:, :, 0] = np.clip(result[:, :, 0] * r_scale, 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * g_scale, 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * b_scale, 0, 255)

    return result.astype(np.uint8)


def _auto_brightness_contrast(
    image: np.ndarray,
    target_brightness: float = 0.45,
    contrast_strength: float = 1.1,
) -> np.ndarray:
    """Automatically adjust brightness and contrast.

    Uses histogram analysis to bring the image to a target brightness level
    and applies contrast enhancement.

    Args:
        image: Input RGB image
        target_brightness: Target mean brightness (0-1 scale)
        contrast_strength: Contrast multiplier (1.0 = no change)
    """
    # Convert to LAB color space for perceptual brightness adjustment
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    # Split into channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Calculate current mean brightness (L channel is 0-255)
    current_brightness = np.mean(l_channel) / 255.0

    # Calculate brightness adjustment needed
    if current_brightness > 0:
        brightness_ratio = target_brightness / current_brightness
        # Limit adjustment to avoid extreme changes
        brightness_ratio = np.clip(brightness_ratio, 0.5, 2.0)
    else:
        brightness_ratio = 1.0

    # Apply brightness adjustment to L channel
    l_float = l_channel.astype(np.float32)
    l_adjusted = l_float * brightness_ratio

    # Apply contrast enhancement (around the new mean)
    new_mean = np.mean(l_adjusted)
    l_contrasted = (l_adjusted - new_mean) * contrast_strength + new_mean

    # Clip and convert back
    l_final = np.clip(l_contrasted, 0, 255).astype(np.uint8)

    # Merge channels and convert back to RGB
    lab_adjusted = cv2.merge([l_final, a_channel, b_channel])
    bgr_result = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)

    return cv2.cvtColor(bgr_result, cv2.COLOR_BGR2RGB)


def _auto_straighten(image: np.ndarray, mode: str = "auto") -> np.ndarray:
    """Automatically straighten image based on detected lines.

    Uses Hough line detection to find dominant horizontal and vertical lines,
    then rotates/corrects the image to level them.

    Args:
        image: Input RGB image
        mode: "auto" (both H+V), "horizontal" (horizon only), "vertical" (verticals only)

    Returns:
        Straightened and cropped image
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=min(w, h) // 8,
        maxLineGap=20,
    )

    if lines is None:
        return image

    # Separate horizontal and vertical line angles
    horizontal_angles = []
    vertical_angles = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Horizontal lines: angles near 0 or 180
        if abs(angle) < 30 or abs(angle) > 150:
            # Normalize to deviation from horizontal
            if abs(angle) > 90:
                angle = angle - 180 if angle > 0 else angle + 180
            horizontal_angles.append(angle)

        # Vertical lines: angles near 90 or -90
        elif 60 < abs(angle) < 120:
            # Normalize to deviation from vertical
            deviation = angle - 90 if angle > 0 else angle + 90
            vertical_angles.append(deviation)

    # Calculate rotation angle based on mode
    rotation_angle = 0.0

    if mode in ("auto", "horizontal") and horizontal_angles:
        # Use median to be robust against outliers
        h_correction = np.median(horizontal_angles)
        rotation_angle = -h_correction

    if mode in ("auto", "vertical") and vertical_angles:
        v_correction = np.median(vertical_angles)
        # For auto mode, average with horizontal if both detected
        if mode == "auto" and horizontal_angles:
            rotation_angle = (rotation_angle - v_correction) / 2
        else:
            rotation_angle = -v_correction

    # Only apply if correction is meaningful but not too extreme
    if abs(rotation_angle) < 0.1 or abs(rotation_angle) > 15:
        return image

    # Apply rotation
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # Calculate new bounding box to avoid cropping
    cos_angle = abs(np.cos(np.radians(rotation_angle)))
    sin_angle = abs(np.sin(np.radians(rotation_angle)))
    new_w = int(h * sin_angle + w * cos_angle)
    new_h = int(h * cos_angle + w * sin_angle)

    # Adjust rotation matrix for new size
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2

    # Rotate with white background (will be cropped)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Crop to remove border artifacts - calculate largest inscribed rectangle
    crop_margin = int(max(w, h) * abs(np.sin(np.radians(rotation_angle))))
    crop_x = crop_margin
    crop_y = crop_margin
    crop_w = new_w - 2 * crop_margin
    crop_h = new_h - 2 * crop_margin

    # Ensure valid crop dimensions
    if crop_w > 0 and crop_h > 0:
        rotated = rotated[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

    return rotated


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
