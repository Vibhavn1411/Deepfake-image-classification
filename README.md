# Deepfake-image-classification
This is the project based on the classification based on deepfake images using python
# Deepfake Image and Expression Swap Detection

## Overview
This project focuses on detecting deepfake images and expression swaps using machine learning models. The primary aim is to develop an accurate and robust solution capable of distinguishing real images from deepfake ones. The model incorporates a hybrid architecture combining CNN and LSTM to capture both spatial and temporal features.

## Dataset
The dataset used for this project is from [Kaggle's Deepfake and Real Images Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images). It contains separate folders for real and fake images, and the images are preprocessed before training.

### Dataset Structure
The dataset is organized into three main directories:
- `train`: Training images (real and fake)
- `val`: Validation images (real and fake)
- `test`: Test images (real and fake)

The images are in JPG, PNG, or JPEG formats.

## Preprocessing
The preprocessing steps include:
- Reading images using OpenCV and converting them from BGR to RGB.
- Resizing images to 224x224 pixels.
- Normalizing pixel values to a range of [-1, 1].
- Converting images to tensors.

The following script was used for preprocessing:
```python
import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.image import img_to_array

DATASET_PATH = r"C:\Users\vibha\Desktop\Vibhav1\projects\finaldsp\data"  # Change this to your actual dataset path
IMG_SIZE = 224

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        for label, folder in enumerate(["real", "fake"]):
            folder_path = os.path.join(root_dir, folder)
            for filename in os.listdir(folder_path):
                if filename.endswith((".jpg", ".png", ".jpeg")):
                    self.image_paths.append(os.path.join(folder_path, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if transform:
            image = transform(image)
        return image, torch.tensor(label, dtype=torch.long)

train_dataset = DeepfakeDataset(os.path.join(DATASET_PATH, "train"))
val_dataset = DeepfakeDataset(os.path.join(DATASET_PATH, "val"))
test_dataset = DeepfakeDataset(os.path.join(DATASET_PATH, "test"))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images, Test: {len(test_dataset)} images")

# Optional: Save preprocessed data as NumPy arrays
def save_numpy_data(dataset, filename):
    images, labels = [], []
    for img, label in dataset:
        images.append(img.numpy())
        labels.append(label.numpy())
    np.savez_compressed(filename, images=np.array(images), labels=np.array(labels))

save_numpy_data(train_dataset, "train_data.npz")
save_numpy_data(val_dataset, "val_data.npz")
save_numpy_data(test_dataset, "test_data.npz")
```

## Model Architecture
The hybrid deepfake detection model is built using a pre-trained ResNet-18 as the feature extractor and an LSTM for capturing temporal dependencies. The architecture includes:
- ResNet-18 feature extractor with frozen initial layers.
- Two-layer LSTM with 128 hidden units and dropout.
- Fully connected layers for classification.

## Training and Evaluation
### Training Parameters
- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: Cross-Entropy Loss
- Batch Size: 32
- Number of Epochs: 10

### Performance Metrics
The project evaluates the model using precision, recall, F1-score, and confusion matrix visualizations. Epoch-wise plots for training and validation accuracy and loss are also provided.

## Instructions to Run
1. Clone the repository.
2. Ensure that the dataset is organized as mentioned above.
3. Install the required libraries using:
   ```bash
   pip install -r requirements.txt
   ```
4. Execute the training script:
   ```bash
   python train_model.py
   ```

## Results
The results include a classification report and plots for accuracy vs. epoch and loss vs. epoch, along with a confusion matrix.

## Future Work
- Fine-tune the ResNet layers for better feature extraction.
- Experiment with additional datasets.
- Implement real-time detection.

## License
This project is licensed under the MIT License.

