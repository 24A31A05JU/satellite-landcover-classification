
from pathlib import Path
import json

import pandas as pd
import torch
from torch import nn
from PIL import Image
from torchvision import transforms

ROOT = Path(".")
DEVICE = torch.device("cpu")

# Load class names
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

# Define the same CNN architecture used during training
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
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


model = SmallCNN(len(class_names))
model.load_state_dict(
    torch.load("models/satellite_cnn.pth", map_location=DEVICE)
)
model.eval()

# Read test dataset
test_df = pd.read_csv("outputs/test.csv")

# Select one test image
row = test_df.sample(n=1, random_state=42).iloc[0]
image_path = Path(str(row["image_path"]))

if not image_path.is_absolute():
    image_path = ROOT / image_path

image = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_index = probabilities.argmax(dim=1).item()

predicted_class = class_names[predicted_index]
actual_class = row["label"]
confidence = probabilities[0, predicted_index].item() * 100

print("\n--- SATELLITE IMAGE PREDICTION ---")
print("Image:", image_path)
print("Actual class:", actual_class)
print("Predicted class:", predicted_class)
print(f"Model confidence: {confidence:.2f}%")
print("Correct prediction:", predicted_class == actual_class)