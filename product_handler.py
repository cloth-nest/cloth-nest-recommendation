import logging
import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from numpy.linalg import norm
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import preprocess_input
from operations import extract_features
import utils

products_info_catalog_file = "product_info.pkl"
products_features_file = "product_features.pkl"


def add_product_as_recommend_candidate(new_product_info, model):
    """
    Add a product (a recommendation candidate) to the recommendation system's storage for later processing and recommendation

    Products not added with this method won't appear in recommendation

    Parameters:
    - new_product_info: A dictionary of product's info, should include 2 fields: 'id', 'image_url'

    """
    try:
        products_info_catalog = []
        products_features = []

        if os.path.exists(products_info_catalog_file):
            products_info_catalog = pickle.load(open(products_info_catalog_file, "rb"))
            logging.debug(
                f"product_handler.py - add_product_recommend_candidate() - [0A] products_info.length: {len(products_info_catalog)}"
            )
        else:
            logging.info(
                f"product_handler.py - add_product_recommend_candidate() - [0A] Adding info for the 1st product of catalog: {new_product_info}"
            )

        if os.path.exists(products_features_file):
            products_features = pickle.load(open(products_features_file, "rb"))
            logging.debug(
                f"product_handler.py - add_product_recommend_candidate() - [0A] products_features.length: {len(products_features)}"
            )
        else:
            logging.info(
                f"product_handler.py - add_product_recommend_candidate() - [0B] Extracting feature for 1st product of catalog: {new_product_info}"
            )

        if utils.check_if_product_exist(
            product_id=new_product_info["id"], products_info_catalog=products_info_catalog
        ):
            logging.error(
                f"product_handler.py - add_product_recommend_candidate() - [X] Product with id - {new_product_info['id']} already exists in catalog"
            )
            return False

        new_product_feature = extract_features(
            image_url=new_product_info["image_url"], model=model
        )
        logging.debug(
            f"product_handler.py - add_product_recommend_candidate() - [0A] new_product_feature: {new_product_feature}"
        )

        products_features.append(new_product_feature)
        pickle.dump(products_features, open(products_features_file, "wb"))

        products_info_catalog.append(new_product_info)
        pickle.dump(products_info_catalog, open(products_info_catalog_file, "wb"))

        return True
    except Exception as e:
        logging.error(
            f"product_handler.py - add_product_recommend_candidate() - [X] EXCEPTION: {e}"
        )
        raise e
