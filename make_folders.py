import os
import shutil
import random

SOURCE_DIR = "Data"
TARGET_DIR = "Data_Clean"
CLASSES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
SPLIT_RATIO = 0.8  # 80% for training, 20% for testing

print("📁 Reorganizing files on disk for absolute test isolation...")

# 1. Create the new clean folder structures
for split in ['Train', 'Test']:
    for class_name in CLASSES:
        os.makedirs(os.path.join(TARGET_DIR, split, class_name), exist_ok=True)

# 2. Physically sort the files
for class_name in CLASSES:
    source_class_path = os.path.join(SOURCE_DIR, class_name)
    
    # Gather all images inside this specific category folder
    all_images = []
    for root, dirs, files in os.walk(source_class_path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                all_images.append(os.path.join(root, file))
                
    # Shuffle using a fixed seed so it's perfectly organized
    random.seed(42)
    random.shuffle(all_images)
    
    # Calculate the split boundary
    split_index = int(len(all_images) * SPLIT_RATIO)
    train_images = all_images[:split_index]
    test_images = all_images[split_index:]
    
    # Copy files to their physical Train destination
    for img_path in train_images:
        dest = os.path.join(TARGET_DIR, 'Train', class_name, os.path.basename(img_path))
        shutil.copy(img_path, dest)
        
    # Copy files to their physical Test destination
    for img_path in test_images:
        dest = os.path.join(TARGET_DIR, 'Test', class_name, os.path.basename(img_path))
        shutil.copy(img_path, dest)

print("🎉 Success! Your data is physically split. You can delete make_folders.py now.")
print("Check your project directory for the new 'Data_Clean' folder!")