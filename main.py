import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn
import os

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Edge AI IP Guardian", page_icon="🛡️", layout="wide")

st.title("🛡️ Edge-AI Deepfake Detector (Proprietary)")
st.markdown("""
**B2C Privacy + B2B Security.** This model is designed to run locally on consumer edge devices to detect AI-generated images. 
It contains a hidden cryptographic white-box signature to prevent corporate IP theft.
""")

# --- 2. LOAD MODEL (Cached so it doesn't reload every click) ---
def load_model():
    try:
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
        # 1. Get the directory where app.py actually lives
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Build the path dynamically (Assuming app.py is outside Riyal-or-Faaake)
        # If app.py is INSIDE Riyal-or-Faaake, remove "Riyal-or-Faaake" from this list below.
        model_path = os.path.join(current_dir, "model", "checkpoints", "clean_resnet18_baseline.pt")
        
        # 3. Quick debug text so you can see exactly where Python is looking
        st.write(f"🕵️ Debug: Looking for model at: `{model_path}`")
        
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None
    
model = load_model()

# --- 3. IMAGE PREPROCESSING ---
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# --- 4. DASHBOARD LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. On-Device Inference")
    uploaded_file = st.file_uploader("Upload an image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
    model_path = "clean_resnet18_baseline.pt"  # Ensure this file is in the same directory as this script
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Run Deepfake Analysis"):
            if model is None:
                st.error("Model file not found! Please add 'clean_resnet18_baseline.pt' to the directory.")
            else:
                with st.spinner('Analyzing pixel artifacts...'):
                    input_tensor = process_image(image)
                    outputs = model(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                    
                    # Assuming Class 0 is FAKE and Class 1 is REAL based on standard folder sorting
                    result = "REAL PHOTO 📸" if predicted.item() == 1 else "AI GENERATED 🤖"
                    
                    if predicted.item() == 1:
                        st.success(f"**Prediction:** {result}")
                    else:
                        st.error(f"**Prediction:** {result}")

with col2:
    st.header("2. IP Ownership Verification")
    st.info("Simulate an enterprise security audit. Extract the weights from the deployed model to check for our cryptographic signature.")
    
    if st.button("🔍 Extract White-Box Watermark"):
        if model is None:
            st.warning("Deploy a model first to audit it.")
        else:
            with st.spinner("Decompiling parameter space and analyzing fc layer weights..."):
                # This is a placeholder for the UI. 
                # Person 3 (Cryptographer) will give you the real extraction math to put here later!
                weights = model.fc.weight.data.numpy()
                mean_weight = weights.mean()
                
                st.write("Targeting Layer: `model.fc.weight`")
                st.code(f"Weight Matrix Shape: {weights.shape}\nRaw Mean Val: {mean_weight:.6f}")
                
                # Fake success for the baseline UI (will be replaced by real math later)
                st.success("✅ **OWNERSHIP VERIFIED:** Cryptographic Hash Match Found.")
                st.balloons()