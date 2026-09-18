from pathlib import Path
from collections import Counter

data_folder = Path("../data/eurosat/2750")

classes = []

for image in data_folder.rglob("*.jpg"):
    class_name = image.parent.name
    classes.append(class_name)

counts = Counter(classes)

print("Number of classes:", len(counts))
print()

for class_name, count in sorted(counts.items()):
    print(f"{class_name}: {count} images")