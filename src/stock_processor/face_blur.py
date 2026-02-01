"""Face detection and blurring."""

import numpy as np
import cv2


def blur_faces(image: np.ndarray, blur_strength: int = 25) -> np.ndarray:
    """Detect faces in image and apply blur to anonymize them.

    Uses face_recognition library if available (more accurate),
    falls back to OpenCV Haar cascades otherwise.

    Args:
        image: Input image as numpy array (RGB)
        blur_strength: Gaussian blur kernel size (must be odd)

    Returns:
        Image with faces blurred
    """
    # Ensure blur strength is odd (required for Gaussian blur)
    if blur_strength % 2 == 0:
        blur_strength += 1

    # Try face_recognition first (more accurate)
    try:
        import face_recognition
        return _blur_with_face_recognition(image, blur_strength)
    except ImportError:
        pass

    # Fall back to OpenCV Haar cascades
    return _blur_with_opencv(image, blur_strength)


def _blur_with_face_recognition(
    image: np.ndarray,
    blur_strength: int,
) -> np.ndarray:
    """Use face_recognition library for detection."""
    import face_recognition

    face_locations = face_recognition.face_locations(image, model="hog")

    if not face_locations:
        return image

    result = image.copy()

    for top, right, bottom, left in face_locations:
        face_region = result[top:bottom, left:right]
        blurred_face = cv2.GaussianBlur(
            face_region,
            (blur_strength, blur_strength),
            0
        )
        result[top:bottom, left:right] = blurred_face

    return result


def _blur_with_opencv(image: np.ndarray, blur_strength: int) -> np.ndarray:
    """Use OpenCV Haar cascades for detection (fallback)."""
    try:
        # Load the pre-trained face detector
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)

        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        if len(faces) == 0:
            return image

        result = image.copy()

        for x, y, w, h in faces:
            # Add padding around face
            padding = int(0.2 * max(w, h))
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)

            face_region = result[y1:y2, x1:x2]
            blurred_face = cv2.GaussianBlur(
                face_region,
                (blur_strength, blur_strength),
                0
            )
            result[y1:y2, x1:x2] = blurred_face

        return result

    except Exception as e:
        print(f"Warning: Face detection failed: {e}")
        return image
