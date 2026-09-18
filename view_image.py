from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

data_folder = Path("../data/eurosat/2750")

image_path = next(data_folder.rglob("*.jpg"))

print("Opening image:")
print(image_path)

image = Image.open(image_path)

print("Image size:", image.size)

plt.imshow(image)
plt.axis("off")
plt.show()