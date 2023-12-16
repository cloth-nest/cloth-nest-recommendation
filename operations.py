from tensorflow.keras.preprocessing import image
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.applications.resnet50 import preprocess_input
import requests
from PIL import Image
from io import BytesIO
import logging


def extract_features(image_url, model):
    """
    Extract features from an image and returns the result embedding
    """
    
    logging.debug(f"operations.py - extract_features() - image_url: {image_url}")

    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).resize((224, 224))

    # Tensorflow, Keras works with numerical data, so we need to convert image to Numpy array
    img_array = image.img_to_array(img)

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
