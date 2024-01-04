import argparse
import logging
import zipfile
from matplotlib import pyplot as plt
from sklearn.metrics import (
    precision_score,
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
)
import torch
from torchvision import transforms
from outfit_compatibility_model import OutfitCompatibilityModel
from outfit_dataset import OutfitDataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import shutil

# region Adding CLI arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--datazip", type=str, default=None, help="Path to input data zip file"
)

parser.add_argument(
    "--run_real",
    type=int,
    default=0,
    help="0 = train with few data to see  model run; 1 = train with whole dataset. Default is 0",
)

parser.add_argument(
    "--log_level",
    type=int,
    default=0,
    help="0 = Print >= warnings, 1 = print >= info, 2 = print all",
)

parser.add_argument(
    "--datadir", type=str, default="data", help="Path to data directory"
)

parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="checkpoint",
    help="Path to the directory to save checkpoints",
)

parser.add_argument(
    "--batch_size", type=int, default=50, help="Batch size in training, default is 50"
)

parser.add_argument(
    "--polyvore_split",
    default="disjoint",
    type=str,
    help="The split of the polyvore data (disjoint or nondisjoint)",
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


# Check if CUDA (GPU support) is available

if torch.cuda.is_available():
    device = f"cuda:{torch.cuda.current_device()}"
else:
    device = "cpu"


torch.set_default_device(device)


def main():

    # DEBUG - INFO - WARNING - ERROR
    if args.log_level == 1:
        logging.basicConfig(level=logging.INFO)
    elif args.log_level == 2:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    # region [1] Extracting data zip file
    if args.datazip is not None:
        logging.info(f"main.py - [1] - args.datazip: {args.datazip}")
        zip_ref = zipfile.ZipFile(args.datazip, "r")
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

    kwargs = {"num_workers": 8, "pin_memory": True} if torch.cuda.is_available() else {
    }

    train_dataset = OutfitDataset(
        data_directory=args.datadir,
        polyvore_split=args.polyvore_split,
        split="train",
        transform=transform,
        run_real=args.run_real,
    )

    valid_dataset = OutfitDataset(
        data_directory=args.datadir,
        polyvore_split=args.polyvore_split,
        split="valid",
        transform=transform,
        run_real=args.run_real,
    )

    test_dataset = OutfitDataset(
        data_directory=args.datadir,
        polyvore_split=args.polyvore_split,
        split="test",
        transform=transform,
        run_real=args.run_real,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        generator=torch.Generator(device=device),
        **kwargs,
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        generator=torch.Generator(device=device),
        **kwargs,
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        generator=torch.Generator(device=device),
        **kwargs,
    )

    logging.warning(
        f"- Number of outfits in train_data_loader: {len(train_data_loader.dataset)} \n- Number of outfits in valid_data_loader: {len(valid_data_loader.dataset)} \n -Number of outfits in test_data_loader: {len(test_data_loader.dataset)}"
    )
    # endregion

    # region [3] Define Model & Training Utilities
    model = OutfitCompatibilityModel()
    model
    focal_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # endregion

    # region [4] Train Model
    train_losses = []
    valid_auc_scores = []
    train(
        model=model,
        optimizer=optimizer,
        focal_loss=focal_loss,
        num_epochs=args.epochs,
        dataloader=train_data_loader,
        val_dataloader=valid_data_loader,
        losses=train_losses,
        auc_scores=valid_auc_scores,
    )
    show_plots(values=train_losses, label="Loss")
    show_plots(values=valid_auc_scores, label="Valid AUC")

    # endregion

    # region [5] Test Model

    final_auc, _, _, _, _ = test(model=model, dataloader=test_data_loader)

    logging.warning(f"final_auc: {final_auc}")
    # endregion


def train(
    model,
    optimizer,
    focal_loss,
    num_epochs,
    dataloader,
    val_dataloader,
    losses,
    auc_scores,
):
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()

        for index, batch in enumerate(dataloader):
            outfit_images = batch["outfit_images"].to(device)
            outfit_texts = batch["outfit_texts"]
            outfit_items_nums = batch["outfit_items_nums"]
            outfit_labels = batch["outfit_labels"].to(device)

            logging.debug(
                f"\nBATCH {index} - images.shape: {outfit_images.shape}")
            logging.debug(f"batch - texts[0]: {outfit_texts[0]}")
            logging.debug(f"batch - labels.shape: {outfit_labels.shape}")

            optimizer.zero_grad()
            outputs = model(outfit_images, outfit_texts, outfit_items_nums)
            loss = focal_loss(outputs, outfit_labels)
            loss.backward()
            optimizer.step()

            # We need to multiple the loss.item() with outfit_images.size(0) which is the batch_size (the number of outfits in this batch) because by default, the loss will be calculate for this batch then divided by batch_size.
            # Reference:https://discuss.pytorch.org/t/how-to-calculate-loss-per-epoch/118519/2
            # Reference: https://stackoverflow.com/questions/61092523/what-is-running-loss-in-pytorch-and-how-is-it-calculated
            running_loss += loss.item() * outfit_images.size(0)

        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)

        epoch_auc, epoch_accuracy, epoch_precision, epoch_recall, epoch_f1 = test(
            model=model, dataloader=val_dataloader
        )
        auc_scores.append(epoch_auc)

        directory = f"{args.checkpoint_dir}/{args.polyvore_split}/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = f"{directory}checkpoint_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "auc": epoch_auc,
            },
            filename,
        )

        if epoch_auc >= max(auc_scores):
            shutil.copyfile(filename, f"{directory}best_state.pt")

        logging.warning(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss} AUC: {epoch_auc} Accuracy: {epoch_accuracy} Precision: {epoch_precision} Recall: {epoch_recall} F1: {epoch_f1}"
        )


def test(model, dataloader):
    model.eval()
    all_val_labels = []
    all_val_outputs = []
    correct = 0
    with torch.no_grad():
        for val_batch in dataloader:
            val_images = val_batch["outfit_images"].to(device)
            val_texts = val_batch["outfit_texts"]
            val_outfit_items_nums = val_batch["outfit_items_nums"]
            val_labels = val_batch["outfit_labels"].to(device)

            val_outputs = model(val_images, val_texts, val_outfit_items_nums)
            val_outputs_binary = (val_outputs > 0.5).float()

            all_val_labels.extend(val_labels.cpu().numpy())
            all_val_outputs.extend(val_outputs_binary.cpu().numpy())

            correct += (val_labels == val_outputs_binary).sum().item()

    logging.info(
        f"test - all_val_labels{all_val_labels},all_val_outputs: {all_val_outputs}"
    )
    auc = roc_auc_score(all_val_labels, all_val_outputs)
    accuracy = accuracy_score(all_val_labels, all_val_outputs)
    precision = precision_score(all_val_labels, all_val_outputs)
    recall = recall_score(all_val_labels, all_val_outputs)
    f1 = f1_score(all_val_labels, all_val_outputs)

    return auc, accuracy, precision, recall, f1


def custom_collate(batch):
    outfits_images = []
    outfits_texts = []
    outfits_items_nums = []
    outfits_labels = [float(outfit["outfit_label"]) for outfit in batch]

    # Find the maximum number of items in any outfit in this batch
    max_items = max(len(outfit["outfit_items_images"]) for outfit in batch)

    for outfit in batch:
        # outfit_items_images is a tensor with this shape (item_count, image_channel, image_height, image_width)
        outfit_items_images = outfit["outfit_items_images"]

        # This will create a tensor with this shape (max_items, image_channel, image_height, image_width) where all values is 0.
        # *outfit_items_images.size()[1:] is an unpacking operation => it will unpack individual sizes of this tensor's dimensions starting from 1st dimension, which means the result will be (image_channel, image_height, image_width)
        padded_images = torch.zeros(max_items, *outfit_items_images.size()[1:])

        # Then we simply fill the first few rows with our existing images tensors
        outfit_actual_item_num = outfit_items_images.shape[0]
        padded_images[:outfit_actual_item_num] = outfit_items_images

        # Similarly, pad the texts so we will have tensors with equal size
        padded_texts = outfit["outfit_items_descriptions"] + [""] * (
            max_items - len(outfit["outfit_items_descriptions"])
        )
        outfits_items_nums.append(outfit_actual_item_num)
        outfits_images.append(padded_images)
        outfits_texts.append(padded_texts)

    return {
        "outfit_images": torch.stack(outfits_images).to(device),
        "outfit_items_nums": outfits_items_nums,
        "outfit_texts": outfits_texts,
        "outfit_labels": torch.tensor(outfits_labels, dtype=torch.float).to(device),
    }


def save_checkpoint(state, is_best, filename="best-model-parameters.pt"):
    """Saves checkpoint to disk"""
    directory = f"runs/{args.polyvore_split}/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "runs/%s/" %
                        (args.name) + "model_best.pth.tar")


def show_plots(values, label):
    plt.plot(values, label=label)

    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel(label, fontsize=16)
    plt.legend(fontsize=16)
    plt.show()


# This prevents our script to run automatically when used in other modules. With this code, the script only runs when this file is run as "main script" (The command in terminal is "python this_file.py") => The "__name__" variable will equal to "__main__" in that case
if __name__ == "__main__":
    main()
