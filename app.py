from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

print("⏳ Initializing PyTorch Architecture & Loading Weights...")

# 1. Rebuild the exact same network architecture used in training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None) # We don't need internet weights, we have our own
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 4)
)

# 2. Load your custom trained 11-million parameter weights
model.load_state_dict(torch.load("resnet_brain_weights.pth", map_location=device, weights_only=True))
model = model.to(device)
model.eval() # Set model to evaluation mode (turns off dropout)

print("✅ Deep Learning Model Online.")

# 3. Define the exact same transformations used on the test data
inference_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # PyTorch expects RGB images for ResNet, not Grayscale!
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Apply strict tensor transformations
        img_tensor = inference_transforms(img).unsqueeze(0).to(device) # Add batch dimension
        
        # Pass tensor through the neural network
        with torch.no_grad():
            outputs = model(img_tensor)
            
            # Apply Softmax to convert raw logits into probability percentages
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # Get highest probability
        prediction_idx = torch.argmax(probabilities).item()
        primary_diagnosis = CLASSES[prediction_idx]
        
        # Map probabilities back to dictionary for the frontend UI
        model_consensus = {
            CLASSES[i]: round(prob.item() * 100, 2) for i, prob in enumerate(probabilities)
        }
        
        return {
            "primary_diagnosis": primary_diagnosis,
            "model_consensus": model_consensus
        }

    except Exception as e:
        return {"error": str(e)}