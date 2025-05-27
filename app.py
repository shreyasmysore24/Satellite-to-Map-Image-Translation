import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from Models import Generator
import config
from utils import load_checkpoint
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision.utils import save_image
import os

def calculate_land_cover(map_path):
    img = cv2.imread(map_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Color ranges
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    water_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    vegetation_mask = cv2.inRange(hsv, lower_green, upper_green)

    total_pixels = img.shape[0] * img.shape[1]
    water_pixels = np.count_nonzero(water_mask)
    vegetation_pixels = np.count_nonzero(vegetation_mask)

    visualization = img.copy()
    visualization[water_mask != 0] = [0, 0, 255]  # Blue for water
    visualization[vegetation_mask != 0] = [0, 255, 0]  # Green for vegetation

    return {
        'water': (water_pixels / total_pixels) * 100,
        'vegetation': (vegetation_pixels / total_pixels) * 100,
        'land': 100 - ((water_pixels + vegetation_pixels) / total_pixels) * 100,
        'visualization': visualization
    }

def generate_map(input_image):
    model = Generator(in_channels=3).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    load_checkpoint(config.CHECKPOINT_GEN, model, optimizer, config.LEARNING_RATE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    input_tensor = transform(input_image).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        output = model(input_tensor)

    output = output * 0.5 + 0.5

    generated_map_path = "generated_map.png"
    save_image(output, generated_map_path)

    return generated_map_path

# --- Streamlit App ---

st.title("Satellite to Map Generator and Land Cover Analyzer 🌎🛰️")

uploaded_file = st.file_uploader("Upload a Satellite Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Satellite Image", use_column_width=True)

    if st.button("Generate Map and Analyze"):
        with st.spinner('Running model and analyzing...'):
            generated_map_path = generate_map(image)
            generated_map = Image.open(generated_map_path)
            analysis = calculate_land_cover(generated_map_path)

        st.subheader("Generated Map")
        st.image(generated_map, use_column_width=True)

        st.subheader("Feature Detection (Blue = Water, Green = Vegetation)")
        st.image(analysis['visualization'], use_column_width=True, channels="RGB")

        st.subheader("Land Cover Analysis Results")
        st.markdown(f"""
        - **Water**: {analysis['water']:.2f} %
        - **Vegetation**: {analysis['vegetation']:.2f} %
        - **Land (Other)**: {analysis['land']:.2f} %
        """)

        # Clean up if you want
        if os.path.exists(generated_map_path):
            os.remove(generated_map_path)
