from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import random

app = FastAPI(
    title="NeuroScan Advanced Diagnostics & Training Platform",
    version="4.0.0"
)

# Allow the frontend to talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. SERVE IMAGES TO THE BROWSER FOR THE GAME ---
app.mount("/static", StaticFiles(directory="Data_Clean"), name="static")

# PyTorch/ImageFolder default alphabetical class mapping order
CLASSES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

print("⏳ Initializing PyTorch Architecture & Loading Weights...")

# --- 2. LOAD DEEP LEARNING MODEL ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None) 
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 4)
)

model.load_state_dict(torch.load("resnet_brain_weights.pth", map_location=device, weights_only=True))
model = model.to(device)
model.eval() 

print("✅ Deep Learning Model Online.")

inference_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3. ENDPOINT: CLINICAL DIAGNOSIS PREDICTION ---
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img_tensor = inference_transforms(img).unsqueeze(0).to(device) 
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        prediction_idx = torch.argmax(probabilities).item()
        primary_diagnosis = CLASSES[prediction_idx]
        
        model_consensus = {
            CLASSES[i]: round(prob.item() * 100, 2) for i, prob in enumerate(probabilities)
        }
        
        return {
            "primary_diagnosis": primary_diagnosis,
            "model_consensus": model_consensus
        }
    except Exception as e:
        return {"error": str(e)}

# --- 4. ENDPOINT: HEAD-TO-HEAD VS AI GAME ---
@app.get("/api/game/start")
async def start_game():
    test_dir = "Data_Clean/Test"
    all_images = []
    
    # Map out the file paths and true labels
    for c in CLASSES:
        class_dir = os.path.join(test_dir, c)
        if os.path.exists(class_dir):
            for file in os.listdir(class_dir):
                if file.endswith((".jpg", ".jpeg", ".png")):
                    all_images.append({
                        "local_path": os.path.join(class_dir, file),
                        "url": f"http://127.0.0.1:8000/static/Test/{c}/{file}",
                        "label": c
                    })
    
    # Shuffle and pick 20 random testing images
    random.shuffle(all_images)
    game_images = all_images[:20]
    
    quiz_list = []
    
    # Pre-calculate what the AI model predicts for these exact 20 images
    for img in game_images:
        try:
            pil_img = Image.open(img["local_path"]).convert('RGB')
            img_tensor = inference_transforms(pil_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                
            prediction_idx = torch.argmax(outputs[0]).item()
            ai_prediction = CLASSES[prediction_idx]
        except Exception:
            ai_prediction = "Error during inference"
            
        quiz_list.append({
            "url": img["url"],
            "label": img["label"],
            "ai_prediction": ai_prediction
        })
    
    return {"quiz": quiz_list}