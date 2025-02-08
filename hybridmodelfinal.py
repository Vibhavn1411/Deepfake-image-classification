import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

train_data = np.load("train_data.npz")
val_data = np.load("val_data.npz")

X_train, y_train = train_data["images"], train_data["labels"]
X_val, y_val = val_data["images"], val_data["labels"]
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
class HybridDeepfakeModel(nn.Module):
    def __init__(self):
        super(HybridDeepfakeModel, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False
        for param in resnet.layer4.parameters():
            param.requires_grad = True
        self.features = nn.Sequential(*list(resnet.children())[:-1])  
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = x.unsqueeze(1)
        _, (hidden, _) = self.lstm(x)
        x = hidden[-1]
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridDeepfakeModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10):
    epoch_results = []
    train_acc_values, val_acc_values = [], []
    train_loss_values, val_loss_values = [], []
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        start_time = time.time()
        mini_batch_accuracies, mini_batch_losses = [], []

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            mini_batch_accuracy = (predicted == labels).sum().item() / labels.size(0)
            mini_batch_loss = loss.item()
            mini_batch_accuracies.append(mini_batch_accuracy)
            mini_batch_losses.append(mini_batch_loss)
        train_acc = correct / total
        model.eval()
        val_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_acc = correct / total
        time_elapsed = time.time() - start_time
        avg_mini_batch_accuracy = np.mean(mini_batch_accuracies)
        avg_mini_batch_loss = np.mean(mini_batch_losses)
        epoch_results.append([
            epoch + 1, 
            len(train_loader), 
            f"{time_elapsed:.2f} sec", 
            f"{avg_mini_batch_accuracy:.4f}", 
            f"{train_acc:.4f}", 
            f"{val_acc:.4f}", 
            f"{avg_mini_batch_loss:.4f}", 
            f"{train_loss / len(train_loader):.4f}", 
            f"{val_loss / len(val_loader):.4f}"
        ])
        train_acc_values.append(train_acc)
        val_acc_values.append(val_acc)
        train_loss_values.append(train_loss / len(train_loader))
        val_loss_values.append(val_loss / len(val_loader))
        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}] - Time: {time_elapsed:.2f} sec - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
    return epoch_results, all_labels, all_preds, train_acc_values, val_acc_values, train_loss_values, val_loss_values
epoch_results, y_true, y_pred, train_acc_values, val_acc_values, train_loss_values, val_loss_values = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10)
epoch_df = pd.DataFrame(epoch_results, columns=["Epoch", "Iterations", "Time Elapsed", "Mini-Batch Accuracy", "Training Accuracy", "Validation Accuracy", "Mini-Batch Loss", "Training Loss", "Validation Loss"])
print("\nUpdated Epoch Table:")
print(epoch_df)
class_report = classification_report(y_true, y_pred, target_names=["Real", "Fake"])
print("\nClassification Report:")
print(class_report)
overall_accuracy = precision_score(y_true, y_pred, average='micro')
weighted_accuracy = precision_score(y_true, y_pred, average='weighted')
macro_accuracy = precision_score(y_true, y_pred, average='macro')
print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
print(f"Weighted Accuracy: {weighted_accuracy:.4f}")
print(f"Macro Accuracy: {macro_accuracy:.4f}")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

epochs = np.arange(1, 11)



plt.figure(figsize=(8, 6))
plt.plot(epochs, train_acc_values, 'o-', label="Train Accuracy")
plt.plot(epochs, val_acc_values, 'x-', label="Validation Accuracy")
plt.title("Accuracy vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(epochs, train_loss_values, 'o-', label="Train Loss")
plt.plot(epochs, val_loss_values, 'x-', label="Validation Loss")
plt.title("Loss vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.savefig("sharp_loss_plot.png")
plt.show()
