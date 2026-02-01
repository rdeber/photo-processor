"""AI-powered metadata generation using Claude Vision."""

import numpy as np


def generate_metadata(
    image: np.ndarray,
    keywords_min: int = 42,
    keywords_max: int = 47,
) -> dict:
    """Generate title, description, and keywords for stock photo.

    Uses Claude Vision API to analyze the image and generate
    stock-photography-optimized metadata.

    Args:
        image: Input image as numpy array (RGB)
        keywords_min: Minimum number of keywords to generate
        keywords_max: Maximum number of keywords to generate

    Returns:
        dict with keys: title, description, keywords
    """
    # TODO: Implement Claude Vision API call
    # For now, return placeholder metadata
    return {
        "title": "Untitled Stock Photo",
        "description": "Stock photo awaiting AI-generated description.",
        "keywords": ["stock", "photo", "image"],
    }


def _build_prompt(keywords_min: int, keywords_max: int) -> str:
    """Build the prompt for Claude Vision API."""
    return f"""Analyze this photograph for use on stock photography sites (Adobe Stock, Shutterstock, iStock).

Generate the following metadata:

1. TITLE: A concise, descriptive title (5-10 words) that captures the essence of the image.
   Should be searchable and compelling.

2. DESCRIPTION: A thorough, unique description (2-3 sentences) that describes:
   - What is shown in the image
   - The mood/atmosphere
   - Potential use cases for buyers

3. KEYWORDS: Generate {keywords_min}-{keywords_max} high-converting keywords that:
   - Describe exactly what's in the image (subjects, objects, actions)
   - Include location/setting descriptors
   - Include mood/emotion descriptors
   - Include conceptual terms (what the image represents)
   - Include technical terms (composition, lighting, style)
   - Are commonly searched on stock sites

Return your response in this exact JSON format:
{{
    "title": "Your title here",
    "description": "Your description here",
    "keywords": ["keyword1", "keyword2", "keyword3", ...]
}}
"""
