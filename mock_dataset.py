import torch
from torch.utils.data import Dataset

class MockOutfitDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.images = torch.rand(
            (num_samples, 3, 224, 224)
        )  # Mock image data, assuming 3 channels and size 224x224
        self.texts = [
            "mock description " + str(i) for i in range(num_samples)
        ]  # Mock text descriptions
        self.labels = torch.randint(
            2, size=(num_samples,), dtype=torch.float32
        )  # Mock binary labels (0 or 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        text = self.texts[idx]
        label = self.labels[idx]
        return image, text, label
