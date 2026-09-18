from torchvision.datasets import EuroSAT

print("Starting dataset download...")

dataset = EuroSAT(
    root="../data",
    download=True
)

print("Dataset downloaded successfully!")
print("Total images:", len(dataset))