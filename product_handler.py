import logging
import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from numpy.linalg import norm
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import preprocess_input
from consts import (
    FEATURES_CATALOG_FIELD_PRODUCT_FEATURE,
    FEATURES_CATALOG_FIELD_PRODUCT_ID,
    PRODUCT_CATALOG_FIELD_PRODUCT_ID,
    PRODUCT_CATALOG_FIELD_PRODUCT_IMAGE,
    PRODUCTS_FEATURES_CATALOG_FILE,
    PRODUCTS_INFO_CATALOG_FILE,
)
from operations import extract_features
import utils


def add_products_as_recommend_candidate(new_products, model):
    """
    Add products to the recommendation system's storage for later processing and recommendation

    Products not added with this method won't appear in recommendation

    Parameters:
    - new_products: A list of products' info, each should include 2 fields: 'id', 'image_url'

    """
    try:
        logging.info("-" * 50)
        products_info_catalog = []
        products_features_catalog = []

        if os.path.exists(PRODUCTS_INFO_CATALOG_FILE):
            products_info_catalog = pickle.load(open(PRODUCTS_INFO_CATALOG_FILE, "rb"))
            logging.debug(
                f"product_handler.py - add_product_recommend_candidate() - [0A] products_info.length: {len(products_info_catalog)}"
            )
        else:
            logging.info(
                f"product_handler.py - add_product_recommend_candidate() - [0A] Adding info for the 1st product of catalog: {new_products[0]}"
            )

        if os.path.exists(PRODUCTS_FEATURES_CATALOG_FILE):
            products_features_catalog = pickle.load(
                open(PRODUCTS_FEATURES_CATALOG_FILE, "rb")
            )
            logging.debug(
                f"product_handler.py - add_product_recommend_candidate() - [0A] products_features.length: {len(products_features_catalog)}"
            )
        else:
            logging.info(
                f"product_handler.py - add_product_recommend_candidate() - [0B] Extracting feature for 1st product of catalog: {new_products[0]}"
            )

        new_products_info = []
        new_products_features = []
        for new_product_info in new_products:
            if utils.check_if_product_exist(
                product_id=new_product_info[PRODUCT_CATALOG_FIELD_PRODUCT_ID],
                products_info_catalog=products_info_catalog,
            ):
                logging.error(
                    f"product_handler.py - add_product_recommend_candidate() - [X] Product with id - {new_product_info[PRODUCT_CATALOG_FIELD_PRODUCT_ID]} already exists in catalog"
                )
                return False

            new_product_feature = extract_features(
                image_url=new_product_info[PRODUCT_CATALOG_FIELD_PRODUCT_IMAGE],
                model=model,
            )
            logging.debug(
                f"product_handler.py - add_product_recommend_candidate() - [1] new_product_feature: {new_product_feature}"
            )
            new_products_features.append(
                {
                    FEATURES_CATALOG_FIELD_PRODUCT_ID: new_product_info[
                        PRODUCT_CATALOG_FIELD_PRODUCT_ID
                    ],
                    FEATURES_CATALOG_FIELD_PRODUCT_FEATURE: new_product_feature,
                }
            )
            new_products_info.append(new_product_info)

        products_features_catalog.extend(new_products_features)
        pickle.dump(
            products_features_catalog, open(PRODUCTS_FEATURES_CATALOG_FILE, "wb")
        )

        products_info_catalog.extend(new_products_info)
        pickle.dump(products_info_catalog, open(PRODUCTS_INFO_CATALOG_FILE, "wb"))
        logging.info("-" * 50)

        return True
    except Exception as e:
        logging.exception(
            f"product_handler.py - add_product_recommend_candidate() - [X] EXCEPTION: {e}"
        )
        return False
