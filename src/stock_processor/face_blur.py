"""Face detection and blurring."""

import numpy as np
import cv2


def blur_faces(image: np.ndarray, blur_strength: int = 25) -> np.ndarray:
    """Detect faces in image and apply blur to anonymize them.

    Args:
        image: Input image as numpy array (RGB)
        blur_strength: Gaussian blur kernel size (must be odd)

    Returns:
        Image with faces blurred
    """
    try:
        import face_recognition

        # Find all face locations in the image
        face_locations = face_recognition.face_locations(image, model="hog")

        if not face_locations:
            return image

        # Create a copy to modify
        result = image.copy()

        # Ensure blur strength is odd (required for Gaussian blur)
        if blur_strength % 2 == 0:
            blur_strength += 1

        for top, right, bottom, left in face_locations:
            # Extract face region
            face_region = result[top:bottom, left:right]

            # Apply strong Gaussian blur
            blurred_face = cv2.GaussianBlur(
                face_region,
                (blur_strength, blur_strength),
                0
            )

            # Replace face region with blurred version
            result[top:bottom, left:right] = blurred_face

        return result

    except ImportError:
        print("Warning: face_recognition not installed, skipping face blur")
        return image
    except Exception as e:
        print(f"Warning: Face detection failed: {e}")
        return image
