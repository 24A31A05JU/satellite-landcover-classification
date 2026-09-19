
from pathlib import Path
import json
import random

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models

# Run from satellite_landcover_project folder
ROOT = Path(".")
OUTPUTS = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

SEED = 42
EPOCHS = 3
BATCH_SIZE = 32

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def load_csv(path):
    df = pd.read_csv(path)

    path_col = next(
        (c for c in ["path", "image_path", "filepath", "file_path"]
         if c in df.columns), None
    )
    label_col = next(
        (c for c in ["label", "class", "class_name", "category"]
         if c in df.columns), None
    )

    if path_col is None or label_col is None:
        raise ValueError(f"Unexpected CSV columns: {list(df.columns)}")

    return df.rename(
        columns={path_col: "image_path", label_col: "label"}
    )[["image_path", "label"]]


train_df = load_csv(OUTPUTS / "train.csv")
val_df = load_csv(OUTPUTS / "validation.csv")

class_names = sorted(train_df["label"].unique())
class_to_idx = {name: i for i, name in enumerate(class_names)}

print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Classes:", class_names)


# ResNet18 preprocessing
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class SatelliteDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

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


train_loader = DataLoader(
    SatelliteDataset(train_df, train_transform),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    SatelliteDataset(val_df, val_transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# Load pretrained ResNet18
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Freeze pretrained feature layers
for param in model.parameters():
    param.requires_grad = False

# Train a new classifier for EuroSAT
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)


# Training and validation
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / max(total, 1)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Validation accuracy: {accuracy:.2f}%"
    )


# Save model and class labels
torch.save(model.state_dict(), MODEL_DIR / "satellite_resnet18.pth")

with open(MODEL_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f)

print("ResNet18 model saved!")
print("Training complete!")
