import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import gradio as gr
import numpy as np
import os

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512
MODEL_PATH = "model/project_retina_final.pth"

# --- MODEL DEFINITION ---
# We must redefine the architecture so PyTorch knows where to load the weights
def load_model():
    print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
    model = models.efficientnet_b0(weights=None) # No need for internet weights
    
    # Recreate the 3-Class Head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 3) 
    
    # Load Weights
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# --- PREDICTION ENGINE ---
def predict(image):
    if image is None:
        return None, "Please upload an image.", None

    # 1. Preprocess
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_pil = Image.fromarray(image.astype('uint8'), 'RGB')
    
    # 2. TTA (Test Time Augmentation) - 3 Variations for Robustness
    variations = [
        img_pil, 
        img_pil.transpose(Image.FLIP_LEFT_RIGHT), 
        img_pil.rotate(10)
    ]
    batch = torch.stack([transform(v) for v in variations]).to(DEVICE)
    
    # 3. Inference
    with torch.no_grad():
        outputs = model(batch)
        probs = F.softmax(outputs, dim=1)
    
    # Average the scores
    avg_probs = torch.mean(probs, dim=0)
    
    p_healthy = float(avg_probs[0])
    p_npdr = float(avg_probs[1])
    p_vt = float(avg_probs[2])
    
    score_dict = {
        'Healthy': p_healthy, 
        'NPDR (Monitor)': p_npdr, 
        'Vision Threat (Refer)': p_vt
    }
    
    # 4. Safety Logic (The "Guardrail")
    if p_vt > 0.15:
        msg = f"⚠️ CRITICAL REFERRAL\nHigh Risk of Vision-Threatening DR ({p_vt:.1%})."
    elif p_healthy < 0.50:
        msg = f"⚠️ REFERRAL RECOMMENDED\nSigns of NPDR detected ({p_npdr:.1%})."
    else:
        msg = "✅ NEGATIVE\nNo significant signs of DR."
        
    return score_dict, msg

# --- INTERFACE ---
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Retinal Fundus Photo"),
    outputs=[
        gr.Label(num_top_classes=3, label="Risk Analysis"), 
        gr.Textbox(label="Clinical Decision Support")
    ],
    title="Project Retina: Safety-First AI",
    description="Early detection of Diabetic Retinopathy using EfficientNet-B0 trained with Focal Loss. \n\n**Safety Protocol:** Any Vision-Threatening probability > 15% triggers an immediate referral.",
    examples=[] 
)

if __name__ == "__main__":
    interface.launch()