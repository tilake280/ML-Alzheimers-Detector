import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import time
import os

# Hyperparameters
DATA_DIR = "Data"
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
CLASSES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]

# Use CUDA or Apple Silicon if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Transforms
# Augment the training data to reduce overfitting
train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
])

test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets (already split into Train/Test folders on disk)
print("Loading datasets...")

train_dataset = datasets.ImageFolder(root="Data_Clean/Train", transform=train_transforms)
test_dataset = datasets.ImageFolder(root="Data_Clean/Test", transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"Loaded {len(train_dataset)} training images and {len(test_dataset)} test images.")

# Model
print("Building ResNet-18...")
# Start from ImageNet pretrained weights
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all layers to keep the pretrained features
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer to output the 4 classes
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(CLASSES))
)

# Unfreeze the last conv block for fine-tuning (the new fc head is already trainable)
for param in model.layer4.parameters():
    param.requires_grad = True

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

# Training loop
print("\nStarting training...")
for epoch in range(EPOCHS):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_acc = 100 * correct_train / total_train
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Time: {epoch_time:.1f}s")

# Evaluate on the test set
print("\nEvaluating on test set...")
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
print(f"\nTest accuracy: {final_accuracy:.2f}%\n")

# Save weights
torch.save(model.state_dict(), "resnet_brain_weights.pth")
print("Saved weights to 'resnet_brain_weights.pth'")
