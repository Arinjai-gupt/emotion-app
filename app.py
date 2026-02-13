import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load Model
# -----------------------------
model = models.resnet18(pretrained=False)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 7)

model.load_state_dict(torch.load("emotion_resnet18_finetuned.pth", map_location=device))
model = model.to(device)
model.eval()

# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3,1,1)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Emotion Recognition App", layout="centered")

st.title("🧠 Facial Emotion Recognition")
st.markdown("Upload a face image and the model will predict the emotion.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    st.success(f"Predicted Emotion: {classes[predicted.item()]}")
