import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.image import img_to_array

# Set dataset paths
DATASET_PATH = r"C:\Users\vibha\Desktop\Vibhav1\projects\finaldsp\data"  # Change this to your actual dataset path
IMG_SIZE = 224  # Resize to 224x224

# Image transformations (PyTorch)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  # Convert to tensor (0-1 range)
    transforms.Normalize([0.5], [0.5])  # Normalize (-1 to 1)
])

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []

        # Load all images
        for label, folder in enumerate(["real", "fake"]):
            folder_path = os.path.join(root_dir, folder)
            for filename in os.listdir(folder_path):
                if filename.endswith((".jpg", ".png", ".jpeg")):  # Adjust formats if needed
                    self.image_paths.append(os.path.join(folder_path, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Read image using OpenCV
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Apply transformations
        if transform:
            image = transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# Load datasets
train_dataset = DeepfakeDataset(os.path.join(DATASET_PATH, "train"))
val_dataset = DeepfakeDataset(os.path.join(DATASET_PATH, "val"))
test_dataset = DeepfakeDataset(os.path.join(DATASET_PATH, "test"))

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images, Test: {len(test_dataset)} images")

# Optional: Save preprocessed data as NumPy arrays (for TensorFlow/Keras)
def save_numpy_data(dataset, filename):
    images, labels = [], []
    for img, label in dataset:
        images.append(img.numpy())
        labels.append(label.numpy())
    np.savez_compressed(filename, images=np.array(images), labels=np.array(labels))

# Save preprocessed datasets (optional)
save_numpy_data(train_dataset, "train_data.npz")
save_numpy_data(val_dataset, "val_data.npz")
save_numpy_data(test_dataset, "test_data.npz")
