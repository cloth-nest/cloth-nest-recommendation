import json
import logging
import os
from torch.utils.data import Dataset

from consts import (
    POLYVORE_ITEM_METADATA_FILENAME,
    POLYVORE_OUTFITS_DIR_NAME,
)


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
                f"outfit_dataset.py - __init__() - [1] Loading file paths: \n - item_metadata_json_path: {item_metadata_json_path}; \n - data_json_file_path: {data_json_file_path}; \n - compatibility_file_path: {compatibility_file_path}"
            )
            # endregion

            try:
                with open(data_json_file_path, "r") as data_json_file:
                    outfit_data = json.load(data_json_file)

            except FileNotFoundError:
                logging.error(
                    f"outfit_dataset.py - __init__() - [2] File not found: {data_json_file_path}"
                )
            except json.JSONDecodeError:
                logging.error(
                    f"outfit_dataset.py - __init__() - [2] Error decoding JSON in file: {data_json_file_path}"
                )

        except Exception as e:
            logging.exception(
                f"outfit_dataset.py - __init__() - exception: {e}")
            raise e
