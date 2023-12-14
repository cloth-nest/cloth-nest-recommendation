import logging
import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from numpy.linalg import norm
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import preprocess_input


rootDir = "data"
imagePath = os.path.join(rootDir, "images")
images = [os.path.join(imagePath, files) for files in os.listdir(imagePath)]
pickle.dump(images, open("images.pkl", "wb"))


def extract_img_feature(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    ex_img_arr = np.expand_dims(img_arr, axis=0)
    pre_pr_img = preprocess_input(ex_img_arr)
    result = model.predict(pre_pr_img).flatten()
    normal_result = result / norm(result)
    return normal_result


def extract_features(images, model):
    feature_list = []
    for file in tqdm(images):
        feature_list.append(extract_img_feature(file, model))

    pickle.dump(feature_list, open("features.pkl", "wb"))
