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
    then applies rotation and perspective correction.

    Args:
        image: Input RGB image
        mode: "auto" (H+V+perspective), "horizontal" (horizon only),
              "vertical" (verticals only), "none" (skip)

    Returns:
        Straightened and cropped image
    """
    if mode == "none":
        return image

    # Step 1: Apply rotation correction (horizontal/vertical leveling)
    image = _apply_rotation_correction(image, mode)

    # Step 2: Apply perspective correction (auto mode only)
    if mode == "auto":
        image = _apply_perspective_correction(image)

    return image


def _apply_rotation_correction(image: np.ndarray, mode: str) -> np.ndarray:
    """Apply rotation to level horizon and/or verticals."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Edge detection with lower thresholds for more edges
    edges = cv2.Canny(gray, 30, 100, apertureSize=3)

    # Detect lines using Hough transform with relaxed parameters
    min_line_len = min(w, h) // 12  # Shorter minimum line length
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,  # Lower threshold for more line detection
        minLineLength=min_line_len,
        maxLineGap=30,  # Larger gap tolerance
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
            if abs(angle) > 90:
                angle = angle - 180 if angle > 0 else angle + 180
            horizontal_angles.append(angle)

        # Vertical lines: angles near 90 or -90
        elif 60 < abs(angle) < 120:
            deviation = angle - 90 if angle > 0 else angle + 90
            vertical_angles.append(deviation)

    # Calculate rotation angle based on mode
    rotation_angle = 0.0

    if mode in ("auto", "horizontal") and horizontal_angles:
        h_correction = np.median(horizontal_angles)
        rotation_angle = -h_correction

    if mode in ("auto", "vertical") and vertical_angles:
        v_correction = np.median(vertical_angles)
        if mode == "auto" and horizontal_angles:
            rotation_angle = (rotation_angle - v_correction) / 2
        else:
            rotation_angle = -v_correction

    # Only apply if correction is meaningful but not too extreme
    if abs(rotation_angle) < 0.05 or abs(rotation_angle) > 15:
        return image

    # Apply rotation
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    cos_angle = abs(np.cos(np.radians(rotation_angle)))
    sin_angle = abs(np.sin(np.radians(rotation_angle)))
    new_w = int(h * sin_angle + w * cos_angle)
    new_h = int(h * cos_angle + w * sin_angle)

    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Crop to remove border artifacts
    crop_margin = int(max(w, h) * abs(np.sin(np.radians(rotation_angle))))
    crop_x = crop_margin
    crop_y = crop_margin
    crop_w = new_w - 2 * crop_margin
    crop_h = new_h - 2 * crop_margin

    if crop_w > 0 and crop_h > 0:
        rotated = rotated[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

    return rotated


def _apply_perspective_correction(image: np.ndarray) -> np.ndarray:
    """Correct perspective distortion from converging vertical lines.

    Detects vertical lines that converge toward a vanishing point
    and applies a perspective transform to make them parallel.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Edge detection with relaxed thresholds
    edges = cv2.Canny(gray, 30, 100, apertureSize=3)

    # Detect lines with relaxed parameters
    min_line_len = min(w, h) // 10
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=min_line_len,
        maxLineGap=40,
    )

    if lines is None:
        return image

    # Find vertical lines and their angles
    left_angles = []  # Lines on left half
    right_angles = []  # Lines on right half

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Only consider near-vertical lines (60-120 degrees)
        if 60 < abs(angle) < 120:
            mid_x = (x1 + x2) / 2
            # Deviation from perfect vertical (90 degrees)
            deviation = angle - 90 if angle > 0 else angle + 90

            if mid_x < w / 2:
                left_angles.append(deviation)
            else:
                right_angles.append(deviation)

    # Need lines on both sides to detect convergence
    if len(left_angles) < 2 or len(right_angles) < 2:
        return image

    left_median = np.median(left_angles)
    right_median = np.median(right_angles)

    # Convergence: left lines lean right (+), right lines lean left (-)
    # Or vice versa for downward convergence
    convergence = left_median - right_median

    # Only correct if there's meaningful convergence (but not extreme)
    if abs(convergence) < 0.5 or abs(convergence) > 25:
        return image

    # Calculate perspective correction strength
    # Positive convergence = lines converge upward, need to widen top
    # Negative convergence = lines converge downward, need to widen bottom
    correction_factor = convergence * 0.002  # Scale factor for subtle correction

    # Define source points (original corners)
    src_pts = np.float32([
        [0, 0],           # Top-left
        [w, 0],           # Top-right
        [w, h],           # Bottom-right
        [0, h],           # Bottom-left
    ])

    # Calculate horizontal shift for perspective correction
    shift = int(w * abs(correction_factor))

    if convergence > 0:
        # Converging upward - widen top
        dst_pts = np.float32([
            [-shift, 0],          # Top-left moves left
            [w + shift, 0],       # Top-right moves right
            [w, h],               # Bottom-right stays
            [0, h],               # Bottom-left stays
        ])
    else:
        # Converging downward - widen bottom
        dst_pts = np.float32([
            [0, 0],               # Top-left stays
            [w, 0],               # Top-right stays
            [w + shift, h],       # Bottom-right moves right
            [-shift, h],          # Bottom-left moves left
        ])

    # Calculate perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Calculate new dimensions to contain the transformed image
    new_w = w + 2 * shift
    new_h = h

    # Adjust destination points for the larger canvas
    if convergence > 0:
        dst_pts = np.float32([
            [0, 0],
            [new_w, 0],
            [new_w - shift, h],
            [shift, h],
        ])
    else:
        dst_pts = np.float32([
            [shift, 0],
            [new_w - shift, 0],
            [new_w, h],
            [0, h],
        ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply perspective transform
    corrected = cv2.warpPerspective(
        image,
        matrix,
        (new_w, new_h),
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Crop to remove stretched edges
    crop_x = shift
    crop_w = w
    if crop_x >= 0 and crop_x + crop_w <= new_w:
        corrected = corrected[:, crop_x : crop_x + crop_w]

    return corrected


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
