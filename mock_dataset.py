import torch
from torch.utils.data import Dataset


class MockOutfitDataset(Dataset):
    def __init__(self, num_outfits=20, items_per_outfit=5):
        self.num_outfits = num_outfits
        self.items_per_outfit = items_per_outfit

        # num_outfits: Number of outfits in the dataset.
        # items_per_outfit: Number of items in each outfit.
        # 3: Assuming each image has three channels (e.g., RGB).
        # 224: Image height.
        # 224: Image width.
        self.images = torch.rand((num_outfits, items_per_outfit, 3, 224, 224))
        self.texts = [
            ["mock description " + str(j) for j in range(items_per_outfit)]
            for _ in range(num_outfits)
        ]
        self.labels = torch.randint(2, size=(num_outfits,), dtype=torch.float32)
        print(f"MockOutfitDataset; images: {self.images.shape}, texts: {self.texts}")

    def __len__(self):
        return self.num_outfits

    def __getitem__(self, idx):
        outfit_images = self.images[idx]
        outfit_texts = self.texts[idx]
        outfit_labels = self.labels[idx]
        return {
            "outfit_images": outfit_images,
            "outfit_texts": outfit_texts,
            "outfit_labels": outfit_labels,
        }
