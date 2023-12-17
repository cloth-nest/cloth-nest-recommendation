from flask import Blueprint
from sklearn.neighbors import NearestNeighbors
import consts
import logging
from flask import Blueprint, jsonify, request, current_app

from utils import check_if_product_exist, get_feature_by_product_id

endpoint_recommendation_blueprint = Blueprint("recommendation", __name__)


@endpoint_recommendation_blueprint.route(
    "/product/recommend/<int:product_id>", methods=["GET"]
)
def get_product_recommendations(product_id):
    try:
        recommendations_count = request.args.get("recommendations_count", default=10)

        if not check_if_product_exist(
            product_id=product_id,
            products_info_catalog=current_app.products_info_catalog,
        ):
            return (
                jsonify({"error": f"Product with id {product_id} does not exist"}),
                400,
            )

        products_features_catalog = current_app.products_features_catalog
        product_feature = get_feature_by_product_id(
            product_id=product_id,
            products_features_catalog=current_app.products_features_catalog,
        )

        if product_feature is None:
            return (
                jsonify({"error": f"Product with id {product_id} does not exist"}),
                400,
            )

        return produce_recommedations(
            query_product_feature=product_feature,
            features_catalog=products_features_catalog,
            recommendations_count=recommendations_count,
        )

    except Exception as e:
        logging.exception(f"Endpoint GET/product has exception: {e.with_traceback}")
        return jsonify({"error": str(e)}), 500


def produce_recommedations(
    query_product_feature, features_catalog, recommendations_count
):
    """
    Produce recommendations for a given product

    Params:
    - query_product_feature: Self-explained
    - features_catalog: A list of dictionaries(including 2 fields: product_id & product_feature)
    - items_num_to_recommend: Self-explained

    Returns:
    The indices of the recommended products in "features_catalog"
    """
    try:
        if recommendations_count > len(features_catalog):
            recommendations_count = len(features_catalog)

        model = NearestNeighbors(n_neighbors=recommendations_count, algorithm="auto")

        product_features = [
            product_feature[consts.FEATURES_CATALOG_FIELD_PRODUCT_FEATURE]
            for product_feature in features_catalog
        ]

        model.fit(product_features)  ## fit with feature list
        distances, indices = model.kneighbors([query_product_feature])

        logging.debug(
            f"recommender.py - produce_recommedantions(); indices of products to recommend: {indices}"
        )

        filtered_products = [features_catalog[i] for i in indices[0]]
        filtered_products_ids = [
            product_feature[consts.FEATURES_CATALOG_FIELD_PRODUCT_ID]
            for product_feature in filtered_products
        ]
        logging.debug(
            f"recommender.py - produce_recommedantions(); ids of products to recommend: {filtered_products_ids}"
        )

        return filtered_products_ids
    except Exception as exception:
        logging.exception(
            f"recommender.py - produce_recommedantions(); Exception: {exception}"
        )
        raise exception
