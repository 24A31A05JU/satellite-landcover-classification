from pathlib import Path

data_folder = Path("../data")

print("Looking inside:", data_folder.resolve())
print()

folders = [item for item in data_folder.rglob("*") if item.is_dir()]

print("Folders found:")
for folder in folders[:20]:
    print(folder)

print()
print("Looking for image files...")

images = list(data_folder.rglob("*.jpg"))

print("Number of JPG images found:", len(images))

if images:
    print("\nFirst 10 images:")
    for image in images[:10]:
        print(image)