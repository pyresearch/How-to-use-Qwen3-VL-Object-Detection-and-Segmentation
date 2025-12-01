import torch
import json
import re
import numpy as np
import cv2
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from PIL import Image
import supervision as sv

# Before running this code, ensure you have the latest transformers installed:
# pip install git+https://github.com/huggingface/transformers

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

TEMPERATURE = 0.5

# Load the processor and model directly on GPU with reduced precision for memory efficiency
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,  # Use float16 for GPU memory savings; adjust if needed
    device_map="cpu"  # Directly shift to GPU (assumes CUDA is available)
)

# Helper function to generate response
def generate_response(image, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to("cuda")  # Explicitly move inputs to GPU
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=TEMPERATURE,
        do_sample=True if TEMPERATURE > 0 else False
    )
    generated_text = processor.batch_decode(
        generated_ids[:, inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )[0]
    return generated_text

# Helper function to parse detections from JSON response
def parse_detections(response_text, width, height):
    # Extract JSON if wrapped in code block
    json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = response_text
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        print("Error parsing JSON")
        return None
    xyxy = []
    class_name = []
    masks = []
    for item in data:
        if "box_2d" in item and "label" in item:
            xyxy.append(item["box_2d"])
            class_name.append(item["label"])
            if "mask" in item:
                # Assume mask is a list of [x, y] points forming a polygon
                poly = np.array(item["mask"], dtype=np.int32)
                # Fix for invalid masks: if it looks like [[x1, y1, x2, y2]], interpret as box and create rectangular polygon
                if poly.ndim == 2 and poly.shape == (1, 4):
                    x1, y1, x2, y2 = poly[0]
                    poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                # Ensure at least 3 points for polygon
                if poly.shape[0] < 3:
                    print(f"Skipping invalid mask with {poly.shape[0]} points for label {item['label']}")
                    masks.append(None)
                    continue
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 1)
                masks.append(mask.astype(bool))
            else:
                masks.append(None)  # Placeholder if no mask
    xyxy = np.array(xyxy, dtype=np.float32)
    class_name = np.array(class_name)
    has_masks = any(m is not None for m in masks)
    if has_masks:
        # Filter out None and stack
        masks = [m for m in masks if m is not None]
        if masks:  # Only stack if there are masks
            masks = np.stack(masks)
        else:
            masks = None
    else:
        masks = None
    detections = sv.Detections(
        xyxy=xyxy,
        mask=masks,
        class_id=np.arange(len(xyxy)),
        data={"class_name": class_name}
    )
    return detections

# First part: Object detection
IMAGE_PATH = "demotest.jpg"  # Adjust as needed
PROMPT = (
    "Detect People present: Mark Zuckerberg, Elon Musk, Jensen Huang, Sundar Pichai, Tim Cook, Satya Nadella, Sam Altman."
    " Output a JSON list of bounding boxes where each entry contains the 2D bounding box in the key \"box_2d\", "
    "and the text label in the key \"label\". Use descriptive labels."
)
image = Image.open(IMAGE_PATH)
width, height = image.size
# Optional resize (Qwen can handle original, but for consistency)
target_height = int(1024 * height / width)
resized_image = image.resize((1024, target_height), Image.Resampling.LANCZOS)
response_text = generate_response(resized_image, PROMPT)
print(response_text)
# Note: If coordinates are in resized space, scale them back to original
# Assuming model outputs normalized or adjust as needed; may require scaling
detections = parse_detections(response_text, width, height)
if detections is None:
    # Handle error
    pass
thickness = sv.calculate_optimal_line_thickness(resolution_wh=(width, height))
text_scale = sv.calculate_optimal_text_scale(resolution_wh=(width, height))
box_annotator = sv.BoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(
    smart_position=True,
    text_color=sv.Color.BLACK,
    text_scale=text_scale,
    text_position=sv.Position.CENTER,
)
annotated = np.array(image)  # Convert to numpy for supervision
for annotator in (box_annotator, label_annotator):
    annotated = annotator.annotate(scene=annotated, detections=detections)
sv.plot_image(annotated)

# Second part: Segmentation
IMAGE_PATH = "demo.jpeg"  # Adjust as needed
PROMPT = (
    "Give the segmentation masks only  person. "
    "Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key \"box_2d\", "
    "the segmentation mask in key \"mask\", and the text label in the key \"label\". Use descriptive labels. "
    "The mask should be a list of [x, y] points forming the polygon, e.g., [[x1,y1], [x2,y2], [x3,y3], ...]."
)
image = Image.open(IMAGE_PATH)
width, height = image.size
# Optional resize
target_height = int(1024 * height / width)
resized_image = image.resize((1024, target_height), Image.Resampling.LANCZOS)
response_text = generate_response(resized_image, PROMPT)
print(response_text)
detections = parse_detections(response_text, width, height)
if detections is None:
    # Handle error
    pass
thickness = sv.calculate_optimal_line_thickness(resolution_wh=(width, height))
text_scale = sv.calculate_optimal_text_scale(resolution_wh=(width, height))
box_annotator = sv.BoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(
    smart_position=True,
    text_color=sv.Color.BLACK,
    text_scale=text_scale,
    text_position=sv.Position.CENTER,
)
masks_annotator = sv.MaskAnnotator()
annotated = np.array(image)  # Convert to numpy for supervision
for annotator in (box_annotator, label_annotator, masks_annotator):
    annotated = annotator.annotate(scene=annotated, detections=detections)
sv.plot_image(annotated)