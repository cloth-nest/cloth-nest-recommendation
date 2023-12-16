def check_if_product_exist(product_id, products_info_catalog):
    """
    Check if a product already exists in the Recommendation System's catalog by comparing id
    """

    # Check if the list is not empty
    if products_info_catalog:
        for product_dictionary in products_info_catalog:
            if "id" in product_dictionary and product_dictionary["id"] == product_id:
                return False

        return True
    else:
        return False
