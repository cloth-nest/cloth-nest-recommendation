import json
import logging
import os
import numpy as np
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
        compatibility_task_file_path = os.path.join(
            rootdir, "compatibility_%s.txt" % split
        )

        self.images_path = os.path.join(datadir, "polyvore_outfits", "images")
        self.transform = transform
        self.loader = loader
        self.split = split
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

        # region Load imageNames and map itemIdentifier (<setId>_<index>) to itemId
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
        # endregion

        # region Map item_id to their index in imageNames list
        imageNames = list(imageNames)
        itemIdToIndex = {}
        for index, itemId in enumerate(imageNames):
            itemIdToIndex[itemId] = index
        logging.debug(
            f"OutfitDataset - itemIdToIndex's 1st 10 items: {get_dict_first_n_items(itemIdToIndex, 10)}"
        )
        # endregion

        # region Map item_ids to their text description
        itemIdToDescription = {}
        for itemId in imageNames:
            desc = item_metadata[itemId]["title"]
            if not desc:
                desc = item_metadata[itemId]["url_name"]

            # The encoding then decoding help remove any unknown characters
            desc = (
                desc.replace("\n", "")
                .encode("utf-8", "ignore")
                .strip()
                .lower()
                .decode("utf-8")
            )

            itemIdToDescription[itemId] = desc

        logging.debug(
            f"OutfitDataset - itemIdToDescription's 1st 10 items: {get_dict_first_n_items(itemIdToDescription, 10)}"
        )
        # endregion

        # region Extract compatibility questions
        compatibility_questions = load_compatibility_questions(
            compatibility_task_file_path, itemIdentifier2ItemId
        )
        logging.debug(
            f"OutfitDataset - compatibility_questions's 1st 10 items: {compatibility_questions[:10]}"
        )
        # endregion

        outfits = []
        (item_ids, label) = compatibility_questions[0]
        cur_outfit = {}
        items_descriptions = [itemIdToDescription[item_id] for item_id in item_ids]
        cur_outfit["outfit_images"] = self.load_images(item_ids)
        for image in cur_outfit["outfit_images"]:
            logging.debug(f"OutfitDataset - image: {image.shape}")

        logging.debug(f"OutfitDataset - cur_outfit_images: {cur_outfit}")

        cur_outfit["outfit_label"] = label
        cur_outfit["outfit_texts"] = items_descriptions

        logging.debug(f"OutfitDataset - cur_outfit: {cur_outfit}")

        self.compatibility_questions = compatibility_questions
        self.itemIdToDescription = itemIdToDescription
        self.outfit_data = outfit_data
        self.itemIdentifier2ItemId = itemIdentifier2ItemId
        self.imageNames = imageNames

    def load_images(self, image_ids):
        images = []
        for image_id in image_ids:
            image_path = os.path.join(self.images_path, "%s.jpg" % image_id)
            img = self.loader(image_path)
            if self.transform is not None:
                img = self.transform(img)

            images.append(img)

        return images

    def __len__(self):
        return len(self.compatibility_questions)

    def __getitem__(self, idx):
        # Implement data loading logic based on your specific dataset structure
        sample = self.data[idx]
        return sample


def load_compatibility_questions(file_path, itemIdentifierToItemId):
    """Returns the list of compatibility questions for the
    split"""
    with open(file_path, "r") as f:
        lines = f.readlines()

    compatibility_questions = []
    for line in lines:
        data = line.strip().split()
        # [1:] means taking items from index 1 -> end in list
        compat_question, _, _ = parse_iminfo(data[1:], itemIdentifierToItemId)
        compatibility_questions.append((compat_question, int(data[0])))

    return compatibility_questions


def parse_iminfo(question, itemIdentifierToItemId, gt=None):
    """Maps the questions from the FITB and compatibility tasks back to
    their index in the precomputed matrix of features

    question: List of images to measure compatibility between
    itemIdToIndex: Dictionary mapping an image name to its location in a
        precomputed matrix of features
    gt: optional, the ground truth outfit set this item belongs to
    """
    questions = []
    is_correct = np.zeros(len(question), np.bool_)
    for index, itemIdentifier in enumerate(question):
        set_id = itemIdentifier.split("_")[0]
        if gt is None:
            gt = set_id

        itemId = itemIdentifierToItemId[itemIdentifier]
        questions.append(itemId)
        is_correct[index] = set_id == gt

    return questions, is_correct, gt
