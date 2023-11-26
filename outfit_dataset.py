import json
import logging
import os
from torch.utils.data import Dataset
from PIL import Image
from utils import get_dict_first_n_items

logging.basicConfig(level=logging.DEBUG)


def default_image_loader(path):
    return Image.open(path).convert("RGB")


# Data Preprocessing
class OutfitDataset(Dataset):
    def __init__(
        self,
        datadir,
        polyvore_split,
        split,
        transform=None,
        loader=default_image_loader,
    ):
        self.data = {1, 2, 3}
        # region Setting directories to images and data json file
        rootdir = os.path.join(datadir, "polyvore_outfits", polyvore_split)
        """Root of the polyvore split (disjoint/non-disjoit)"""

        data_json_file_path = os.path.join(rootdir, "%s.json" % split)
        """JSON File for phase (train, val, test)"""

        metadata_file_path = os.path.join(
            datadir, "polyvore_outfits", "polyvore_item_metadata.json"
        )

        self.imagePath = os.path.join(datadir, "polyvore_outfits", "images")

        # endregion

        # region Read JSON for outfits' itemsId and index
        try:
            with open(data_json_file_path, "r") as file:
                # Load JSON content
                outfit_data = json.load(file)
                """Includes the outfit_id and the items (id + index) (Taken from data json file)"""
                
                # Log JSON content
                # logging.debug(f"OutfitDataset - JSON Content: {json.dumps(outfit_data, indent=2)}")

        except FileNotFoundError:
            logging.error(f"OutfitDataset - File not found: {data_json_file_path}")
        except json.JSONDecodeError:
            logging.error(
                f"OutfitDataset - Error decoding JSON in file: {data_json_file_path}"
            )
        # endregion

        # region Read JSON for outfits' items' metadata
        try:
            with open(metadata_file_path, "r") as file:
                # Load JSON content
                item_metadata = json.load(file)

                # Log JSON content
                # logging.debug(f"OutfitDataset - JSON Content: {json.dumps(item_metadata, indent=2)}")

        except FileNotFoundError:
            logging.error(f"OutfitDataset - File not found: {metadata_file_path}")
        except json.JSONDecodeError:
            logging.error(
                f"OutfitDataset - Error decoding JSON in file: {metadata_file_path}"
            )
        # endregion

        imageNames = set()
        itemIdentifier2ItemId = {}
        for outfit in outfit_data:
            outfit_id = outfit["set_id"]
            for item in outfit["items"]:
                itemId = item["item_id"]
                itemIdentifier2ItemId["%s_%i" % (outfit_id, item["index"])] = itemId
                imageNames.add(itemId)

        logging.debug(
            f"OutfitDataset - itemIdentifier2ItemId's 1st 10 items: {get_dict_first_n_items(itemIdentifier2ItemId, 10)}"
        )
        logging.debug(f"imageNames 1st 10 items: {list(imageNames)[:10]}")

        imageNames = list(imageNames)
        itemIdToIndex = {}
        for index, itemId in enumerate(imageNames):
            itemIdToIndex[itemId] = index
        logging.debug(
            f"OutfitDataset - itemIdToIndex's 1st 10 items: {get_dict_first_n_items(itemIdToIndex, 10)}"
        )

        self.outfit_data = outfit_data
        self.itemIdentifier2ItemId = itemIdentifier2ItemId
        self.imageNames = imageNames
        self.transform = transform
        self.loader = loader
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Implement data loading logic based on your specific dataset structure
        sample = self.data[idx]
        return sample
