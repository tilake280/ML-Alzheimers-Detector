import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# TODO: swap this dummy data out with the real MRI dataset from Kaggle later.
# just generating 100 random 64x64 images for now to test the pipeline locally.
np.random.seed(42)
torch.manual_seed(42)

num_samples = 100
fake_mri_scans = np.random.randn(num_samples, 1, 64, 64).astype(np.float32)
fake_labels = np.random.randint(0, 4, size=num_samples) # 4 classes of dementia

# simple dataset wrapper for pytorch
class BrainScanDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

dataset = BrainScanDataset(fake_mri_scans, fake_labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


# custom cnn to pull features from the raw pixels
class CustomBrainCNN(nn.Module):
    def __init__(self):
        super(CustomBrainCNN, self).__init__()
        # basic conv blocks
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # drops dim to 32x32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # drops dim to 16x16
        )
        
        # classification head
        self.fc1 = nn.Linear(32 * 16 * 16, 64) # this 64-dim vector is what we want for sklearn
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 4) 
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # flatten
        embeddings = self.relu(self.fc1(x))
        logits = self.fc2(embeddings)
        
        # return both logits (for pytorch training) and embeddings (for the classic ML models)
        return logits, embeddings 

model = CustomBrainCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# quick training loop just to get some weight updates
print("training the feature extractor...")
model.train()
for epoch in range(5): 
    total_loss = 0
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"epoch {epoch+1}/5 - loss: {total_loss/len(dataloader):.4f}")


# --- Feature Extraction Phase ---
# push the data through the network to get our embeddings
model.eval()
extracted_features = []
actual_labels = []

with torch.no_grad():
    for images, labels in dataloader:
        _, embeddings = model(images)
        extracted_features.append(embeddings.numpy())
        actual_labels.append(labels.numpy())

# concat batches
X_features = np.concatenate(extracted_features, axis=0)
y_labels = np.concatenate(actual_labels, axis=0)

# shove it into pandas to keep it organized
df_features = pd.DataFrame(X_features)
df_features['target_dementia_level'] = y_labels

# standard 80/20 train-val split
train_size = int(0.8 * num_samples)
X_train, X_val = X_features[:train_size], X_features[train_size:]
y_train, y_val = y_labels[:train_size], y_labels[train_size:]


# --- Sklearn Modeling Phase ---

# testing out logreg
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
lr_preds = log_reg.predict(X_val)
lr_accuracy = accuracy_score(y_val, lr_preds)

# testing out random forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)
rf_preds = random_forest.predict(X_val)
rf_accuracy = accuracy_score(y_val, rf_preds)

print("\nresults:")
print(f"log reg acc: {lr_accuracy * 100:.2f}%")
print(f"rf acc: {rf_accuracy * 100:.2f}%")

# plotting the eval metrics
models = ['Logistic Regression', 'Random Forest']
accuracies = [lr_accuracy * 100, rf_accuracy * 100]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['skyblue', 'salmon'])
plt.ylabel('Accuracy (%)')
plt.title("Dementia Model Eval")
plt.ylim(0, 100)
for i, v in enumerate(accuracies):
    plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
plt.show()