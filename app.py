import logging
import os
import pickle
from flask import Flask
from consts import PRODUCTS_FEATURES_CATALOG_FILE, PRODUCTS_INFO_CATALOG_FILE
from product_handler import (
    add_products_as_recommend_candidate,
)
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalMaxPool2D
import tensorflow as tf
from recommender import endpoint_recommendation_blueprint
from flask import jsonify, request, current_app

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPool2D()])


def load_app_catalogs(app):
    try:
        products_info_catalog = []
        products_features_catalog = []

        if os.path.exists(PRODUCTS_INFO_CATALOG_FILE):
            with open(PRODUCTS_INFO_CATALOG_FILE, "rb") as file:
                products_info_catalog = pickle.load(file)
        else:
            logging.info(f"Recommendation System doesn't contain any products")
        app.products_info_catalog = products_info_catalog

        if os.path.exists(PRODUCTS_FEATURES_CATALOG_FILE):
            with open(PRODUCTS_FEATURES_CATALOG_FILE, "rb") as file:
                products_features_catalog = pickle.load(file)
        else:
            logging.info(f"Recommendation System doesn't contain any products")
        app.products_features_catalog = products_features_catalog

        logging.info(
            f"load_app_catalogs() is called - app.products_info_catalog: {len(app.products_info_catalog)}; app.products_features_catalog: {len(app.products_features_catalog)}"
        )
    except Exception as e:
        logging.exception(f"load_app_catalogs() has exception: {e} {e.with_traceback}")
        raise e


app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

app.model = model
app.register_blueprint(endpoint_recommendation_blueprint)

load_app_catalogs(app=app)


@app.route("/", methods=["GET"])
def hello():
    return "Ok"


@app.route("/product/recommend/catalog", methods=["POST"])
def add_products_to_recommend():
    try:
        new_products_info = request.get_json()
        model = current_app.model

        logging.info(f"Endpoint POST/product received JSON: {new_products_info}")

        if not isinstance(new_products_info, list):
            return (
                    jsonify({
                        "statusCode": 400,
                        "message": "Invalid JSON format. Expected a list.",
                        "error": {
                            "code": "R03",
                            "message": "Invalid JSON format. Expected a list."
                        }
                        }),
                    400,
                )

        (
            add_success,
            already_existed_product_id,
        ) = add_products_as_recommend_candidate(
            new_products=new_products_info, model=model
        )

        if add_success:
            load_app_catalogs(app=current_app)
            return (
                    jsonify({
                        "statusCode": 201,
                        "message": "Products added successfully",
                        }),
                    201,
                )
        else:
            return (
                    jsonify({
                        "statusCode": 400,
                        "message": f"Product with id {already_existed_product_id} already exists",
                        "error": {
                            "code": "R04",
                            "message": f"Product with id {already_existed_product_id} already exists"
                        }
                        }),
                    400,
                )
    except Exception as e:
        logging.exception(f"Endpoint POST/product has exception: {e.with_traceback}")
        return (
            jsonify({
                "statusCode": 500,
                "message": str(e),
                "error": {
                    "code": "R01",
                    "message": str(e)
                }
            }),
            500,
        )


# This prevents our Flask app to run automatically when used in other modules. WIth this code, Flask app only runs whtn this file is run as "main script" (The command in terminal is "python this_file.py") => The "__name__" will == "__main__" in that case
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
