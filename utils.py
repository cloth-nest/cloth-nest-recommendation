import logging
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import math

from consts import (
    FEATURES_CATALOG_FIELD_PRODUCT_FEATURE,
    FEATURES_CATALOG_FIELD_PRODUCT_ID,
    PRODUCT_CATALOG_FIELD_PRODUCT_ID,
)


def check_if_product_exist(product_id, products_info_catalog):
    """
    Check if a product already exists in the Recommendation System's catalog by comparing id
    """

    logging.debug(
        f"utils.py - check_if_product_exist(); product_id: {product_id}, products_info_catalog: {products_info_catalog}"
    )

    # Check if the list is not empty
    if products_info_catalog:
        for product_dictionary in products_info_catalog:
            if PRODUCT_CATALOG_FIELD_PRODUCT_ID in product_dictionary and math.isclose(
                product_dictionary[PRODUCT_CATALOG_FIELD_PRODUCT_ID], product_id
            ):
                return True

        return False
    else:
        return False


def get_feature_by_product_id(product_id, products_features_catalog):
    for product_feature in products_features_catalog:
        if product_feature.get(FEATURES_CATALOG_FIELD_PRODUCT_ID) == product_id:
            return product_feature.get(FEATURES_CATALOG_FIELD_PRODUCT_FEATURE)
    # Return a default value or raise an exception if the "id" is not found
    return None


def image_to_numpy_array(image_url):
    try:
        # It seems that some websites have anti-crawling measures so we add this to counter that. Reference: https://stackoverflow.com/questions/70500064/retrieving-an-image-gives-403-error-while-it-works-with-browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
        }

        with requests.get(image_url, headers=headers) as response:
            if response.status_code == 200:
                with Image.open(BytesIO(response.content)) as img:
                    return np.array(img.resize((224, 224)))
            else:
                # Print an error message if the request was not successful
                print(
                    f"Failed to fetch image from URL. Status code {response.status_code}"
                )
                return None
    except Exception as e:
        logging.exception(
            f"utils.py - image_to_numpy_array() with url {image_url} exception: {e}"
        )
