# Alzheimer's Progression Detector
 
A web app that classifies the stage of Alzheimer's from an axial brain MRI. It uses a fine-tuned ResNet-18 CNN built with PyTorch, served through a FastAPI backend with a simple HTML/JS frontend where you drag in a scan and get a prediction.
 
It classifies into four stages: NonDemented, VeryMildDemented, MildDemented, and ModerateDemented. On a held-out test set it reaches about [X]% accuracy.
 
This is a portfolio project to learn transfer learning and model deployment, not a medical tool. See the disclaimer at the bottom.
 
## How it works
 
The model is ResNet-18 pretrained on ImageNet, fine-tuned on MRI scans. I froze the early convolutional layers (which already detect general features like edges), unfroze the last block (layer4) so it could adapt to brain structure, and replaced the final layer with a 512-unit fully connected head plus dropout (p=0.3) to reduce overfitting.
 
Training: 15 epochs, Adam optimizer (lr=0.001), cross-entropy loss. Inputs are resized to 128x128 and converted to 3 channels so they fit ResNet's expected shape.
 
To avoid data leakage, `make_folders.py` splits the raw dataset into separate Train (80%) and Test (20%) folders on disk before any training, so test images are never seen during training.
 
At inference, `app.py` loads the trained weights, runs a forward pass, and returns softmax probabilities across the four classes. The frontend shows those as a confidence breakdown.
 
## Model details
 
| Setting | Value |
| --- | --- |
| Architecture | ResNet-18 (transfer learning) |
| Input size | 128x128, 3 channels |
| Trainable layers | layer4 + custom FC head |
| FC head | 512 units, dropout p=0.3 |
| Loss | Cross-entropy |
| Optimizer | Adam, lr=0.001 |
| Epochs | 15 |
| Classes | NonDemented, VeryMildDemented, MildDemented, ModerateDemented |
| Dataset | [N] MRI images, [split] train/test |
 
## Setup
 
Requires Python 3.12+.
 
```
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install torch torchvision fastapi uvicorn pillow numpy
```
 
## Running it
 
1. Put your raw dataset in a `Data` folder in the root, with the four class subfolders.
2. Split it into Train/Test:
```
   python make_folders.py
```
3. Train the model (produces `resnet_brain_weights.pth`):
```
   python train.py
```
4. Start the API:
```
   uvicorn app:app --reload
```
   It runs at `http://127.0.0.1:8000`.
5. Open `index.html` with a local server (VS Code Live Server works), then drag an MRI scan into the drop zone to get a prediction.
 
## Files
 
| File | What it does |
| --- | --- |
| `train.py` | Model architecture and training loop |
| `app.py` | FastAPI inference server |
| `make_folders.py` | Splits the dataset into Train/Test |
| `resnet_brain_weights.pth` | Trained model weights |
| `index.html`, `style.css`, `script.js` | Frontend |
 
## Disclaimer
 
This is an educational portfolio project to demonstrate a deep learning pipeline and full-stack deployment. It is not a medical device and must not be used for real diagnosis.
