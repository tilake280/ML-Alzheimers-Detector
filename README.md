```markdown
# Alzheimer's Progress Detector: Advanced Alzheimer's Progression Detection Platform

This progress detector is an end-to-end Deep Learning web application designed to classify and grade Alzheimer's Disease progression from axial brain MRI scans. By leveraging a custom fine-tuned ResNet-18 Convolutional Neural Network (CNN) built with PyTorch, the system achieves a 97%+ validation accuracy across four distinct clinical stages. 

The project bridges advanced computer vision with modern web architecture, featuring a robust asynchronous FastAPI backend interface paired with a responsive HTML5/JavaScript dashboard.

---

## Key Features

* **Deep Learning Classification:** Employs transfer learning via ResNet-18 (11.7 Million parameters) optimized to retain spatial geometric awareness of brain structures (e.g., cortical thinning, ventricular enlargement).
* **Physical Data Isolation:** Includes pipeline infrastructure scripts to physically isolate training (80%) and testing subsets (20%) on disk to guarantee zero data leakage during evaluation.
* **Real-Time Asynchronous Inference:** Production-grade API endpoints capable of processing raw binary image streams, applying dynamic matrix transformations, and calculating Softmax confidence probabilities instantly.
* **Drag-and-Drop Web Interface:** A zero-dependency web frontend using CSS3 variables, live animations, and responsive canvas progress metrics to present diagnostic models clearly.

---

## Repository Structure

```text
alzheimersDetector/
├── Data_Clean/                     # Generated via compilation scripts
│   ├── Train/                      # 32,000 images strictly used for backpropagation
│   └── Test/                       # 8,000 images strictly used for frontend validation
│       ├── NonDemented/
│       ├── VeryMildDemented/
│       ├── MildDemented/
│       └── ModerateDemented/
├── app.py                          # FastAPI production inference server
├── train.py                        # PyTorch network architecture and training loop
├── make_folders.py                 # Structural dataset segregation script
├── resnet_brain_weights.pth        # Compiled deep learning model weights matrix
├── index.html                      # Frontend diagnostic dashboard layout
├── style.css                       # UI stylesheet and core animations
└── script.js                       # Frontend payload dispatcher and UI update logic

```

---

## Installation and Setup

### 1. Environment Initialization

Clone the project repository, navigate into your directory, and initialize a Python 3.12+ virtual environment:

```bash
cd alzheimersDetector
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```

### 2. Dependency Installation

Install the required high-performance mathematical computing and web hosting libraries:

```bash
pip install torch torchvision fastapi uvicorn pillow numpy

```

### 3. Pipeline Data Separation

Place your raw dataset directory named `Data` containing your 4 structural folders in the root directory. Run the segregation script to generate physical `Train` and `Test` environments:

```bash
python make_folders.py

```

### 4. Neural Network Training

Execute the deep learning loop. The network will lock the early edge-detection filters, unfreeze `layer4`, append a custom 512-neuron fully connected layer with Dropout protection (p=0.3), and optimize parameters over 15 epochs:

```bash
python train.py

```

Upon completion, the matrix binaries will archive locally to `resnet_brain_weights.pth`.

---

## Running the Application Stack

### 1. Launch the Production API

Spin up the Uvicorn web server to handle prediction calls. The application auto-detects system environments, initializes the ResNet geometry, and maps network states into local memory:

```bash
uvicorn app:app --reload

```

The server will bind and expose endpoints at `http://127.0.0.1:8000`. Keep this terminal window open.

### 2. Launch the Web Interface

Open `index.html` via a local testing server (such as Live Server or Five Server in VS Code).

### 3. Run Inference Validation

1. Open your computer's file explorer and navigate to `Data_Clean/Test/`.
2. Select any category folder (e.g., `ModerateDemented`) and drag any unseen scan file into the web UI interface drop zone.
3. Select **Execute ML Pipeline**.
4. The system will dispatch the stream, apply 3-channel standard transformations, forward-pass the tensor through the 18 convolutional layers, and instantly display color-coded Softmax confidence distributions across the dashboard layout.

---

## Model Architecture and Technical Specifications

| Parameter | Configuration Specification |
| --- | --- |
| **Base Architecture** | ResNet-18 (Deep Residual Learning) |
| **Input Feature Shape** | $128 \times 128$ Pixels (RGB Alignment) |
| **Parameters Evaluated** | ~11.7 Million Weights & Biases |
| **Regularization** | Dropout Layers ($p=0.3$) + Dynamic Spatial Data Augmentation |
| **Loss Function** | Cross-Entropy Loss (`nn.CrossEntropyLoss`) |
| **Optimization Engine** | Adam Optimizer ($\eta = 0.001$) |
| **Target Categorization** | NonDemented, VeryMildDemented, MildDemented, ModerateDemented |

---

> [!WARNING]
> **Clinical Disclaimer:** This application is an educational software engineering portfolio project designed to demonstrate deep learning architecture deployment, computer vision pipelines, and full-stack integration. It is not approved for medical diagnosis or clinical deployment.

```

```
