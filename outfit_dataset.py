import json
import logging
import os
import torch
from torch.utils.data import Dataset

from consts import (
    POLYVORE_ITEM_METADATA_FILENAME,
    POLYVORE_ITEMS_IMAGES_DIR_NAME,
    POLYVORE_OUTFITS_DIR_NAME,
)
from utils import get_dictionary_first_n_items, load_json
from PIL import Image


class OutfitDataset(Dataset):
    def __init__(
        self,
        data_directory,
        polyvore_split,
        split,
        transform=None,
    ):
        """
        Create a Dataset of outfits for compatibility prediction.

        Parameters:
        - data_directory (str): The path to data directory which should contain the "polyvore_outfits" directory
        - polyvore_split (str): The split of the polyvore dataset, could be disjoint or non-disjoint
        - split (str): Can be train/valid/test
        - transform: The transformation for items' images
        """
        try:
            # region [1] Loading file paths
            item_metadata_json_path = os.path.join(
                data_directory,
                POLYVORE_OUTFITS_DIR_NAME,
                POLYVORE_ITEM_METADATA_FILENAME,
            )

            polyvore_split_root_dir = os.path.join(
                data_directory, POLYVORE_OUTFITS_DIR_NAME, polyvore_split
            )

            data_json_file_path = os.path.join(
                polyvore_split_root_dir, f"{split}.json")

            compatibility_file_path = os.path.join(
                polyvore_split_root_dir, f"compatibility_{split}.txt"
            )

            logging.debug(
                f"outfit_dataset.py - __init__() - [1] Loading file paths: \n - item_metadata_json_path: {item_metadata_json_path}; \n - data_json_file_path: {data_json_file_path}; \n - compatibility_file_path: {compatibility_file_path} \n"
            )
            # endregion

            # region [2] Load outfits data (set id, item id, item ordering index) from JSON
            outfits_data = load_json(
                json_file_path=data_json_file_path,
                message_prefix="outfit_dataset.py - __init__() - [2]",
            )
            # endregion

            # region [3] Load items metadata (item id, description, name) from JSON
            items_metadata = load_json(
                json_file_path=item_metadata_json_path,
                message_prefix="outfit_dataset.py - __init__() - [3]",
            )
            # endregion

            # region [4] Create necessary mappings:
            # - self.item_identifier_to_item_id;
            # - self.item_id_to_item_description;
            # - self.item_id_to_item_img_path

            self.load_necessary_mappings(
                outfits_data=outfits_data,
                items_metadata=items_metadata,
                polyvore_split_root_dir=polyvore_split_root_dir,
            )
            # endregion

            # region [5] Loading Compatibility Prediction questions
            self.load_compatibility_questions(
                compatibility_file_path=compatibility_file_path,
                item_identifier_to_item_id=self.item_identifier_to_item_id,
            )
            # endregion

        except Exception as e:
            logging.exception(
                f"outfit_dataset.py - __init__() - exception: {e}")
            raise e

    def __len__(self):
        return len(self.compatibility_questions)

    def __getitem__(self, idx):
        (label, items_ids) = self.compatibility_questions[idx]

        items_descriptions = [
            self.item_id_to_item_description[item_id] for item_id in items_ids
        ]

        items_img_paths = [
            self.item_id_to_item_img_path[item_id] for item_id in items_ids
        ]

        return {
            "outfit_items_images": self.load_images_tensors(items_img_paths),
            "outfit_items_descriptions": items_descriptions,
            "outfit_label": label,
        }

    def load_images_tensors(self, images_paths):
        images = []
        for path in images_paths:
            img = Image.open(path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)

            images.append(img)

        return torch.stack(images, dim=0)

    def load_necessary_mappings(
        self, outfits_data, items_metadata, polyvore_split_root_dir
    ):
        item_identifier_to_item_id = {}
        item_id_to_item_description = {}
        items_ids = []
        item_id_to_item_img_path = {}

        for outfit in outfits_data:
            set_id = outfit["set_id"]

            for item in outfit["items"]:
                item_index = item["index"]
                item_id = item["item_id"]
                item_identifier = f"{set_id}_{item_index}"

                item_identifier_to_item_id[item_identifier] = item_id

        for item_id, item_metadata in items_metadata.items():
            items_ids.append(item_id)
            item_description = item_metadata["description"]
            item_title = item_metadata["title"]
            item_url_name = item_metadata["url_name"]

            if item_description != "":
                item_id_to_item_description[item_id] = item_description
            elif item_title != "":
                item_id_to_item_description[item_id] = item_title
            else:
                item_id_to_item_description[item_id] = item_url_name

        for item_id in items_ids:
            item_id_to_item_img_path[item_id] = os.path.join(
                polyvore_split_root_dir,
                POLYVORE_ITEMS_IMAGES_DIR_NAME,
                f"{item_id}.jpg",
            )

        self.item_identifier_to_item_id = item_identifier_to_item_id
        self.item_id_to_item_description = item_id_to_item_description
        self.item_id_to_item_img_path = item_id_to_item_img_path

        first_3_items_item_identifier_map = get_dictionary_first_n_items(
            item_identifier_to_item_id, 3
        )

        first_3_items_item_description_map = get_dictionary_first_n_items(
            item_id_to_item_description, 3
        )
        first_3_items_item_image_map = get_dictionary_first_n_items(
            item_id_to_item_img_path, 3
        )

        logging.debug(
            f"outfit_dataset.py - __init__() - [4]: \n item_identifier_to_item_id's first 3 items: {first_3_items_item_identifier_map} \n item_id_to_item_description's first 3 items: {first_3_items_item_description_map} \n item_id_to_item_img_path's first 3 items: {first_3_items_item_image_map} \n "
        )

    def load_compatibility_questions(
        self, compatibility_file_path, item_identifier_to_item_id
    ):
        with open(compatibility_file_path, "r") as compatibility_file:
            lines = compatibility_file.readlines()

        # Each compatibility question has 2 part: An integer (0/1) denoting if outfit is compatibility & list of that outfit's items' ids
        compatibility_questions = []
        for line in lines:
            question = line.split()
            is_outfit_compatible = question[0]
            outfit_items_ids = []

            for item_identifier in question[1:]:
                outfit_items_ids.append(
                    item_identifier_to_item_id[item_identifier])

            compatibility_questions.append(
                (is_outfit_compatible, outfit_items_ids))

        self.compatibility_questions = compatibility_questions
        logging.debug(
            f"outfit_dataset.py - __init__() - [5]: \n compatibility_questions's first 3 items: {compatibility_questions[:3]} \n"
        )
