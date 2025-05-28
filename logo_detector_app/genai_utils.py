# genai_utils.py
import os
import io
import base64
from PIL import Image
import google.genai as genai
from google.genai import types
from config import GEMINI_API_KEY, MODEL_NAME

# Initialize client once
_genai_client = genai.Client(api_key=GEMINI_API_KEY)

# System prompt for bounding-box images
_SYSTEM_PROMPT = (
    "You are an assistant that looks at images with red bounding boxes around logos. "
    "If there is only one red bounding box, write a single line in the format: "
    "'The logo name is ...'. If there are multiple bounding boxes, write: "
    "'The logo names are ...'. Do not include any other details or confidence scores."
)

def extract_brand_names(image) -> str:
    """
    Sends the image (with boxes drawn) to Gemini 2.0 Flash and returns
    the plain-text response.
    """
    # Encode PIL image to PNG bytes
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # Build the conversation contents
    contents = [
        types.Content(
            role="user",
            parts=[
                # image part
                types.Part.from_bytes(
                    mime_type="image/png",
                    data=img_bytes
                ),
                # text part
                types.Part.from_text(
                    text=_SYSTEM_PROMPT
                    )
            ]
        )
    ]

    # Stream the generation
    response_text = []
    stream = _genai_client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(response_mime_type="text/plain")
    )
    for chunk in stream:
        response_text.append(chunk.text)

    return "".join(response_text).strip()
