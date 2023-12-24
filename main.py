import argparse
import logging
import zipfile

# region Adding CLI arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--datazip", type=str, default=None, help="Path to input data zip file"
)

parser.add_argument(
    "--datadir", type=str, default="data", help="Path to data directory"
)

parser.add_argument(
    "--polyvore_split",
    default="disjoint",
    type=str,
    help="sThe split of the polyvore data (either disjoint or nondisjoint)",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    help="Number of epochs to train for (default: 10)",
)

global args
args = parser.parse_args()
# endregion


def main():
    # DEBUG - INFO - WARNING - ERROR
    logging.basicConfig(level=logging.INFO)

    # region Extracting data zip file
    if args.datazip is not None:
        logging.info(f"main.py - [1] - args.datazip: {args.datazip}")
        zip_ref = zipfile.ZipFile(args.data, "r")
        zip_ref.extractall("")
        zip_ref.close()
    # endregion


# This prevents our script to run automatically when used in other modules. With this code, the script only runs when this file is run as "main script" (The command in terminal is "python this_file.py") => The "__name__" variable will equal to "__main__" in that case
if __name__ == "__main__":
    main()
