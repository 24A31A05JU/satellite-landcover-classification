from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# Dataset location
data_folder = Path("../data/eurosat/2750")


# Find all images
images = list(data_folder.rglob("*.jpg"))

print("Total images found:", len(images))


# Create image paths and labels
data = []

for image in images:
    label = image.parent.name
    data.append({
        "image_path": str(image),
        "label": label
    })


df = pd.DataFrame(data)


# First split: 70% training, 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)


# Second split: 15% validation, 15% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

train_df.to_csv("outputs/train.csv", index=False)
val_df.to_csv("outputs/validation.csv", index=False)
test_df.to_csv("outputs/test.csv", index=False)


print()
print("Dataset split completed!")
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))