import io
from PIL import Image

import streamlit as st

from config import PAGE_TITLE, LAYOUT
from model_utils import load_processor_and_model
from model_detection import detect_objects, draw_boxes
from genai_utils import extract_brand_names

def main():
    st.set_page_config(page_title=PAGE_TITLE, layout=LAYOUT)
    st.title(PAGE_TITLE)

    processor, model = load_processor_and_model()

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if not uploaded:
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    with st.spinner("Detecting logos…"):
        result = detect_objects(image, processor, model)

    image_boxes = draw_boxes(image, result, model.config.id2label)
    st.image(image_boxes, caption="Detected Logos", use_container_width=True)

    # Download button
    buf = io.BytesIO()
    image_boxes.save(buf, format="PNG")
    st.download_button(
        "Download with Bounding Boxes",
        data=buf.getvalue(),
        file_name="detected_logos.png",
        mime="image/png"
    )

    # Brand name extraction
    with st.spinner("Recognizing brand names…"):
        text = extract_brand_names(image_boxes)

    st.subheader("GenAI Logo Recognition ::")
    st.write(text)

if __name__ == "__main__":
    main()
