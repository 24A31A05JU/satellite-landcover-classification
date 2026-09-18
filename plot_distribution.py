from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

data_folder = Path("../data/eurosat/2750")

classes = []

for image in data_folder.rglob("*.jpg"):
    class_name = image.parent.name
    classes.append(class_name)

counts = Counter(classes)

names = sorted(counts.keys())
values = [counts[name] for name in names]

plt.figure(figsize=(12, 6))
plt.bar(names, values)

plt.title("EuroSAT Land-Cover Class Distribution")
plt.xlabel("Land-Cover Class")
plt.ylabel("Number of Images")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("outputs/class_distribution.png", dpi=300)

plt.show()

print("Graph saved successfully!")
print("Location: outputs/class_distribution.png")