import argparse
import logging
import zipfile
import torch
from torchvision import transforms
from outfit_dataset import OutfitDataset

# region Adding CLI arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--datazip", type=str, default=None, help="Path to input data zip file"
)

parser.add_argument(
    "--datadir", type=str, default="data", help="Path to data directory"
)

parser.add_argument(
    "--batch_size", type=int, default=50, help="Batch size in training, default is 50"
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
    logging.basicConfig(level=logging.DEBUG)

    # region [1] Extracting data zip file
    if args.datazip is not None:
        logging.info(f"main.py - [1] - args.datazip: {args.datazip}")
        zip_ref = zipfile.ZipFile(args.data, "r")
        zip_ref.extractall("")
        zip_ref.close()
    # endregion

    # region [2] Loading Datasets
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = OutfitDataset(
        data_directory=args.datadir,
        polyvore_split=args.polyvore_split,
        split="train",
        transform=transform,
    )

    valid_dataset = OutfitDataset(
        data_directory=args.datadir,
        polyvore_split=args.polyvore_split,
        split="valid",
        transform=transform,
    )

    test_dataset = OutfitDataset(
        data_directory=args.datadir,
        polyvore_split=args.polyvore_split,
        split="test",
        transform=transform,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True
    )
    # endregion


# This prevents our script to run automatically when used in other modules. With this code, the script only runs when this file is run as "main script" (The command in terminal is "python this_file.py") => The "__name__" variable will equal to "__main__" in that case
if __name__ == "__main__":
    main()
