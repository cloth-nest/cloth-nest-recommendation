from tensorflow.keras.preprocessing import image
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.applications.resnet50 import preprocess_input
import requests
from PIL import Image
from io import BytesIO
import logging

from utils import image_to_numpy_array

logging.getLogger("PIL").setLevel(logging.WARNING)


def extract_features(image_url, model):
    """
    Extract features from an image and returns the result embedding
    """
    try:
        logging.debug(f"operations.py - extract_features() - image_url: {image_url}")

        img_array = image_to_numpy_array(image_url=image_url)

        if img_array is None:
            raise Exception(
                f"operations.py - extract_features() failed to extract image with url: {image_url}"
            )
        logging.debug(
            f"operations.py - extract_features() - img_array's shape: {img_array.shape}"
        )

        # Deep Learning models often handle data in batches, input's shape should be (batch_size, height, width, channels) so we expand the dimension to add batch size (although there's only 1 item)
        expanded_img_array = np.expand_dims(img_array, axis=0)

        # Each model expects input in a certain format, so this pre-process the data for Resnet50
        preprocessed_img = preprocess_input(expanded_img_array)
        logging.debug(
            f"operations.py - extract_features() - preprocessed_img's shape: {preprocessed_img.shape}"
        )

        unflattened_result = model.predict(preprocessed_img)
        logging.debug(
            f"operations.py - extract_features() - unflattened_result's shape: {unflattened_result.shape}"
        )

        flattened_result = unflattened_result.flatten()
        logging.debug(
            f"operations.py - extract_features() - flattened_result's shape: {flattened_result.shape}"
        )

        normalized_result = flattened_result / norm(flattened_result)
        logging.debug(
            f"operations.py - extract_features() - normalized_result's shape: {normalized_result.shape}"
        )

        return normalized_result

    except Exception as e:
        logging.error(
            f"operations.py - extract_features() - EXCEPTION - image_url: {image_url} error: {e.with_traceback}"
        )
        raise e
