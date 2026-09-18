from pathlib import Path
from PIL import Image
import pandas as pd


# Load training CSV
train_df = pd.read_csv("outputs/train.csv")

print("Training records:", len(train_df))

# Get first image
image_path = Path(train_df.iloc[0]["image_path"])

print("First image:")
print(image_path)

# Open image
image = Image.open(image_path)

print("Image size:", image.size)
print("Image mode:", image.mode)