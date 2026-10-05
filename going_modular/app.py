import json
from pathlib import Path

import gradio as gr
import torch
from torchvision.transforms import v2

from modelbuilder import PlantNet


# -----------------------------
# Configuration
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "models").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
MODEL_PATH = PROJECT_ROOT / "models" / "PlantNet_feature_extractor.pth"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Class names
# -----------------------------
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

class_names = [n.replace("___", " - ").replace("_", " ") for n in class_names]
# -----------------------------
# Model
# -----------------------------
model = PlantNet(output_shape=len(class_names))

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()


# -----------------------------
# Inference preprocessing
# -----------------------------
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((224, 224)),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# -----------------------------
# Prediction
# -----------------------------
def predict(image):
    if image is None:
        return None

    image = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(image)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=min(5, len(class_names)),
    )

    return {
        class_names[index.item()]: float(probability.item())
        for probability, index in zip(top_probabilities, top_indices)
    }


# -----------------------------
# Gradio interface
# -----------------------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type="pil",
        label="Upload a Plant Leaf Image",
    ),
    outputs=gr.Label(
        num_top_classes=5,
        label="Plant Disease Prediction",
    ),
    title="Plant Disease Detection",
    description=(
        "Upload an image of a plant leaf and the EfficientNet-B0 "
        "model will predict the most likely plant disease."
    ),
)


if __name__ == "__main__":
    demo.launch()
