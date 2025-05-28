import google.generativeai as genai
from config import GEMINI_API_KEY, MODEL_NAME

genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = (
    "You are an assistant that looks at images with red bounding boxes around logos. "
    "If there is only one red bounding box, write a single line in the format: "
    "'The logo name is ...'. If there are multiple bounding boxes, write: "
    "'The logo names are ...'. Do not include any other details or scores."
)

def extract_brand_names(image_with_boxes):
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content([image_with_boxes, SYSTEM_PROMPT])
    return response.text
