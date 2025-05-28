from PIL import ImageDraw
import torch
from config import DETECTION_THRESHOLD

def detect_objects(image, processor, model, threshold=DETECTION_THRESHOLD):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device="cpu")
    results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )
    return results[0]  # only one image

def draw_boxes(image, result, id2label):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        x, y, x2, y2 = [round(v,2) for v in box.tolist()]
        draw.rectangle((x, y, x2, y2), outline="red", width=2)
        text = f"{id2label[label.item()]}: {score:.2f}"
        draw.text((x, y), text, fill="black")
    return img
