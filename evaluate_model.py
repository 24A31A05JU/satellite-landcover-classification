
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(".")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

# Load the exact class order used during training
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

class_to_idx = {name: i for i, name in enumerate(class_names)}

test_df = pd.read_csv("outputs/test.csv")

print("Device:", DEVICE)
print("Test images:", len(test_df))
print("Classes:", class_names)

# Match ResNet18 preprocessing used during training
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class SatelliteDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = ROOT / image_path

        image = Image.open(image_path).convert("RGB")
        image = test_transform(image)
        label = class_to_idx[row["label"]]

        return image, label


test_loader = DataLoader(
    SatelliteDataset(test_df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Recreate the same ResNet18 architecture
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(
    model.fc.in_features,
    len(class_names)
)

model.load_state_dict(
    torch.load(
        "models/satellite_resnet18.pth",
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()

all_predictions = []
all_labels = []

print("Evaluating ResNet18...")

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.tolist())

accuracy = accuracy_score(all_labels, all_predictions)

print("\n--- TEST RESULTS ---")
print(f"Test accuracy: {accuracy * 100:.2f}%")

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0,
    digits=4
))

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0,
    output_dict=True
)

Path("outputs").mkdir(exist_ok=True)

pd.DataFrame(report).transpose().to_csv(
    "outputs/classification_report.csv"
)

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names)))
)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.title("EuroSAT Test Set - ResNet18 Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.close()

print("\nSaved:")
print("outputs/classification_report.csv")
print("outputs/confusion_matrix.png")
print("Evaluation complete!")
