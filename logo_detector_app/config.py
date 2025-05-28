import os
from dotenv import load_dotenv
load_dotenv()

# Streamlit page settings
PAGE_TITLE = "🧠 Logo Detection & Brand Recognition"
LAYOUT = "centered"

# model dir and gemini key
MODEL_DIR = os.getenv("MODEL_DIR","src/model")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = " gemini-2.0-flash"

# Object detection threshold
DETECTION_THRESHOLD = 0.5
