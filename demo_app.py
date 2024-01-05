import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from outfit_compatibility_model import (
    OutfitCompatibilityModel,
)
from torchvision import transforms
import os

# Load your trained model
model = OutfitCompatibilityModel()
checkpoint = torch.load(
    "checkpoint/disjoint/best_state.pt", map_location=torch.device("cpu")
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

preproces = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
)


def load_images_tensors(transform, images_paths):
    images = []
    for path in images_paths:
        img = Image.open(path).convert("RGB")
        img = transform(img)

        images.append(img)

    return torch.stack(images, dim=0)


st.title("Outfit Compatibility Prediction")

# Upload outfit images
uploaded_files = st.file_uploader(
    "Upload outfit images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

text_inputs = []

if uploaded_files:

    for i, uploaded_file in enumerate(uploaded_files):
        st.image(uploaded_file,
                 caption=f"Uploaded Image {i + 1}", use_column_width=True)
        text_input = st.text_input(f"Text for Image {i + 1}", "")
        text_inputs.append(text_input)

    if st.button("Check Compatibility"):
        temp_dir = "uploaded"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        images_paths = []
        for uploaded_file in uploaded_files:
            path = os.path.join(temp_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getvalue())
                images_paths.append(path)

        images = load_images_tensors(
            transform=transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            ),
            images_paths=images_paths,
        ).unsqueeze(0)

        print(f"texts: {text_inputs}")

        texts = [text_inputs]
        lengths = [images.size(1)]

        print(
            f"demo_app: images' shape: {images.shape}; texts: {texts}, lengths: {lengths} "
        )

        with torch.no_grad():
            prediction = model(images, texts, lengths)

        print(f"demo_app: prediction {prediction} ")
        st.write(f"Compatibility Predictions (0 to 1): {prediction[0]}")

        if prediction[0] >= 0.5:
            result = "Compatible"
        else:
            result = "Not Compatible"

        st.write(f"=> Compatibility Predictions: {result}")
