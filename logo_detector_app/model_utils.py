from transformers import AutoImageProcessor, AutoModelForObjectDetection
import streamlit as st
from config import MODEL_DIR

@st.cache_resource
def load_processor_and_model():
    print(MODEL_DIR)
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model     = AutoModelForObjectDetection.from_pretrained(MODEL_DIR)
    model.eval()
    return processor, model
