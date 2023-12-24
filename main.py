import argparse
import logging
import zipfile

# region Adding CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str,
                    help="Path to input data", default=None)
# endregion


def main():
    # DEBUG - INFO - WARNING - ERROR
    logging.basicConfig(level=logging.INFO)

    # region Load CLI arguments
    global args
    args = parser.parse_args()

    if args.data is not None:
        logging.info(f"main.py - [1] - args.data: {args.data}")
        zip_ref = zipfile.ZipFile(args.data, "r")
        zip_ref.extractall("")
        zip_ref.close()
    # endregion


# This prevents our script to run automatically when used in other modules. With this code, the script only runs when this file is run as "main script" (The command in terminal is "python this_file.py") => The "__name__" variable will equal to "__main__" in that case
if __name__ == "__main__":
    main()
