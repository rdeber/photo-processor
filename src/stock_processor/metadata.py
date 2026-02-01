"""AI-powered metadata generation using Claude Vision."""

import base64
import io
import json
import os

import numpy as np
from PIL import Image


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
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Warning: ANTHROPIC_API_KEY not set, using placeholder metadata")
        return _placeholder_metadata()

    try:
        return _call_claude_vision(image, keywords_min, keywords_max, api_key)
    except Exception as e:
        print(f"Warning: Claude API call failed: {e}")
        return _placeholder_metadata()


def _call_claude_vision(
    image: np.ndarray,
    keywords_min: int,
    keywords_max: int,
    api_key: str,
) -> dict:
    """Call Claude Vision API to analyze image and generate metadata."""
    import anthropic

    # Convert numpy array to base64 JPEG
    image_base64 = _image_to_base64(image)

    # Build the prompt
    prompt = _build_prompt(keywords_min, keywords_max)

    # Call Claude API
    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )

    # Extract the response text
    response_text = message.content[0].text

    # Parse JSON from response
    return _parse_response(response_text)


def _image_to_base64(image: np.ndarray) -> str:
    """Convert numpy array to base64-encoded JPEG."""
    # Convert to PIL Image
    pil_image = Image.fromarray(image)

    # Resize if too large (Claude has image size limits)
    max_dimension = 2048
    if max(pil_image.size) > max_dimension:
        ratio = max_dimension / max(pil_image.size)
        new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
        pil_image = pil_image.resize(new_size, Image.LANCZOS)

    # Convert to JPEG bytes
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)

    # Encode to base64
    return base64.standard_b64encode(buffer.read()).decode("utf-8")


def _parse_response(response_text: str) -> dict:
    """Parse Claude's response to extract metadata."""
    # Try to find JSON in the response
    try:
        # First, try to parse the entire response as JSON
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code block
    if "```json" in response_text:
        start = response_text.find("```json") + 7
        end = response_text.find("```", start)
        if end > start:
            try:
                return json.loads(response_text[start:end].strip())
            except json.JSONDecodeError:
                pass

    # Try to find JSON object in response
    start = response_text.find("{")
    end = response_text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(response_text[start:end])
        except json.JSONDecodeError:
            pass

    # Failed to parse, return placeholder
    print("Warning: Could not parse Claude response as JSON")
    return _placeholder_metadata()


def _build_prompt(keywords_min: int, keywords_max: int) -> str:
    """Build the prompt for Claude Vision API."""
    return f"""Analyze this photograph for use on stock photography sites (Adobe Stock, Shutterstock, iStock).

Generate the following metadata:

1. TITLE: A concise, descriptive title (5-10 words) that captures the essence of the image.
   Should be searchable and compelling. Do not use quotes in the title.

2. DESCRIPTION: A thorough, unique description (2-3 sentences) that describes:
   - What is shown in the image
   - The mood/atmosphere
   - Potential use cases for buyers

3. KEYWORDS: Generate exactly {keywords_min}-{keywords_max} high-converting keywords that:
   - Describe exactly what's in the image (subjects, objects, actions)
   - Include location/setting descriptors if applicable
   - Include mood/emotion descriptors
   - Include conceptual terms (what the image represents)
   - Include technical photography terms (composition, lighting, style)
   - Are commonly searched on stock sites
   - Use single words or short phrases (2-3 words max per keyword)
   - Are lowercase

Return ONLY valid JSON in this exact format, with no additional text:
{{
    "title": "Your title here",
    "description": "Your description here",
    "keywords": ["keyword1", "keyword2", "keyword3", ...]
}}"""


def _placeholder_metadata() -> dict:
    """Return placeholder metadata when API is unavailable."""
    return {
        "title": "Untitled Stock Photo",
        "description": "Stock photo awaiting AI-generated description.",
        "keywords": ["stock", "photo", "image"],
    }
