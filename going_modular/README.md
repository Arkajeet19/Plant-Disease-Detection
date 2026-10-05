---
title: Plant Disease Detection
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
---

# Plant Disease Detection

Upload a close-up photo of a plant leaf and the model predicts the crop and disease (or healthy) from the classes it was trained on.

- **Model:** EfficientNet-B0 pretrained on ImageNet, fine-tuned on the New Plant Diseases Dataset (Augmented)
- **Input:** a single leaf photo, resized to 224x224
- **Output:** the top 5 predicted classes with confidence scores

The training images are single leaves on plain backgrounds, so photos with cluttered backgrounds or several leaves may give less reliable results.
