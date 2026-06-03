import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import time
import os

# ==========================================
# 1. DEEP LEARNING HYPERPARAMETERS
# ==========================================
DATA_DIR = "Data"
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
CLASSES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]

# Auto-detect Apple Silicon, Nvidia GPU, or default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 INITIALIZING DEEP LEARNING PIPELINE ON: {device}")

# ==========================================
# 2. TENSOR TRANSFORMATIONS & AUGMENTATION
# ==========================================
# We augment the training data so the model doesn't overfit and can generalize to 97%+
train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
])

test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. DATASET LOADING (PRE-SPLIT ON DISK)
# ==========================================
print("📂 Loading pre-split image tensors from disk...")

# Directly point to the separate physical folders we just made!
train_dataset = datasets.ImageFolder(root="Data_Clean/Train", transform=train_transforms)
test_dataset = datasets.ImageFolder(root="Data_Clean/Test", transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"✅ Loaded {len(train_dataset)} training tensors and {len(test_dataset)} testing tensors.")

# ==========================================
# 4. NEURAL NETWORK ARCHITECTURE (11.7 MILLION PARAMETERS)
# ==========================================
print("🧠 Constructing ResNet-18 Convolutional Neural Network...")
# Load a model pre-trained on millions of images
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze early layers so we don't destroy the foundational edge-detection weights
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer to output our 4 specific Alzheimer's classes
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(CLASSES))
)

# Unfreeze the last convolutional block and the new fully connected layer for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

# ==========================================
# 5. THE TRAINING LOOP
# ==========================================
print("\n🔥 COMMENCING BACKPROPAGATION & WEIGHT OPTIMIZATION 🔥")
for epoch in range(EPOCHS):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
    train_acc = 100 * correct_train / total_train
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Time: {epoch_time:.1f}s")

# ==========================================
# 6. FINAL VALIDATION (THE 97% TEST)
# ==========================================
print("\n📊 Evaluating deep spatial representations on hidden test data...")
model.eval()
correct_test = 0
total_test = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

final_accuracy = 100 * correct_test / total_test
print(f"\n=========================================================")
print(f"🏆 DEEP LEARNING MODEL ACCURACY: {final_accuracy:.2f}%")
print(f"=========================================================\n")

# ==========================================
# 7. EXPORT NEURAL WEIGHTS
# ==========================================
torch.save(model.state_dict(), "resnet_brain_weights.pth")
print("💾 Neural network weights successfully compiled to 'resnet_brain_weights.pth'")