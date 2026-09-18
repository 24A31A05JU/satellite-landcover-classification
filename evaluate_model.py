
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(".")
DEVICE = torch.device("cpu")
BATCH_SIZE = 32

# Load class names
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

class_to_idx = {
    name: i for i, name in enumerate(class_names)
}

# Load test CSV
test_df = pd.read_csv("outputs/test.csv")

print("Test images:", len(test_df))
print("Classes:", class_names)


# Dataset
class SatelliteDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = ROOT / image_path

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label = class_to_idx[row["label"]]
        return image, label


test_loader = DataLoader(
    SatelliteDataset(test_df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# Same CNN architecture used for training
class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# Load saved model
model = SmallCNN(len(class_names)).to(DEVICE)

model.load_state_dict(
    torch.load(
        "models/satellite_cnn.pth",
        map_location=DEVICE,
    )
)

model.eval()

all_predictions = []
all_labels = []

print("Evaluating model...")


with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.tolist())


# Calculate metrics
accuracy = accuracy_score(all_labels, all_predictions)

print("\n--- TEST RESULTS ---")
print(f"Test accuracy: {accuracy * 100:.2f}%")

print("\n--- CLASSIFICATION REPORT ---")
print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        digits=4,
    )
)

# Save classification report
report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0,
    output_dict=True,
)

Path("outputs").mkdir(exist_ok=True)

pd.DataFrame(report).transpose().to_csv(
    "outputs/classification_report.csv"
)

# Confusion matrix
cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names))),
)

plt.figure(figsize=(12, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names,
)

plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.title("EuroSAT Test Set - Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300,
)

print("\nSaved:")
print("outputs/classification_report.csv")
print("outputs/confusion_matrix.png")
print("Evaluation complete!")