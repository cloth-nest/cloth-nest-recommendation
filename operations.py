from tensorflow.keras.preprocessing import image
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.applications.resnet50 import preprocess_input
import requests
from PIL import Image
from io import BytesIO


def extract_features(image_url, model):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).resize((224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)

    preprocessed_img = preprocess_input(expanded_img_array)  
    result = model.predict(preprocessed_img).flatten() 
    normalized_result = result / norm(result)
    
    return normalized_result
