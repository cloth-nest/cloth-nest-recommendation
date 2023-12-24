import json

import logging


def load_json(json_file_path, message_prefix):

    try:

        with open(json_file_path, "r") as data_json_file:

            json_data = json.load(data_json_file)

            return json_data

    except FileNotFoundError as e:

        logging.error(f"{message_prefix} File not found: {json_file_path}")

        raise e

    except json.JSONDecodeError as e:

        logging.error(
            f"{message_prefix} Error decoding JSON in file: {json_file_path}")

        raise e


def get_dictionary_first_n_items(dict, n):
    return list(dict.items())[:n]
