import logging
import requests
from PIL import Image
import numpy as np
from io import BytesIO


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
            if "id" in product_dictionary and product_dictionary["id"] == product_id:
                return True

        return False
    else:
        return False


def image_to_numpy_array(image_url):
    try:
        with requests.get(image_url) as response:
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
