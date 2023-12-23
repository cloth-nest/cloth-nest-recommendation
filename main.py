import os
import mlflow
import torch
import argparse
import torch.optim as optim
from torchvision import transforms
from outfit_model import OutfitCompatibilityModel
from outfit_dataset import OutfitDataset
import torch.nn as nn
from utils import save_checkpoint
import logging
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="path to input data", default=None)


def main():
    # DEBUG - INFO - WARNING - ERROR
    global args
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.data is not None:
        print("input data:", args.data)

        zip_ref = zipfile.ZipFile(args.data, "r")
        zip_ref.extractall("")
        zip_ref.close()

    data_dir = "data"

    # Should be disjoint/nondisjoint
    polyvore_split = "disjoint"

    # Should be traim/valid/test
    split_valid = "valid"
    split_train = "train"
    split_test = "test"

    default_batch_size = 50

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Organizes your dataset into batches. If we don't do this, our texts matrix will be transposed
    # Batch size = number of samples processed in one iteration
    # Number of batches = total samples divided by batch_size
    # Each this case, a sample = an outfit
    def custom_collate(batch):
        outfits_images = []
        outfits_texts = []
        outfits_labels = []

        # Find the maximum number of items in any outfit in this batch
        max_items = max(len(outfit["outfit_images"]) for outfit in batch)

        for outfit in batch:
            # Pad or truncate the number of items to match max_items
            padded_images = torch.zeros(
                (max_items,) + outfit["outfit_images"].shape[1:]
            )
            padded_images[: outfit["outfit_images"].shape[0]] = outfit["outfit_images"]

            # Similarly, pad or truncate the number of texts
            padded_texts = outfit["outfit_texts"] + [""] * (
                max_items - len(outfit["outfit_texts"])
            )

            outfits_images.append(padded_images)
            outfits_texts.append(padded_texts)
            outfits_labels.append(outfit["outfit_label"])

        return {
            "outfit_images": torch.stack(outfits_images),
            "outfit_texts": outfits_texts,
            "outfit_labels": torch.tensor(outfits_labels, dtype=torch.float),
        }

    # Start Logging
    mlflow.start_run()

    # enable autologging
    mlflow.pytorch.autolog()

    train_dataset = OutfitDataset(data_dir, polyvore_split, split_train, transform)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=default_batch_size,
        shuffle=True,
        collate_fn=custom_collate,
    )

    valid_dataset = OutfitDataset(data_dir, polyvore_split, split_valid, transform)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=default_batch_size,
        shuffle=True,
        collate_fn=custom_collate,
    )

    total_train_num_outfits = len(train_dataloader.dataset)
    total_valid_num_outfits = len(valid_dataloader.dataset)

    mlflow.log_metric("total_train_num_outfits", total_train_num_outfits)
    mlflow.log_metric("total_valid_num_outfits", total_valid_num_outfits)

    print(
        f"total_num_outfits of splits:{split_train}: {total_train_num_outfits}; {split_valid}: {total_valid_num_outfits}"
    )

    # Instantiate the model and components for training (loss function, optimizer)
    model = OutfitCompatibilityModel()
    focal_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    num_epochs = 5

    train_losses = []  # to store training losses
    val_losses = []  # to store validation losses
    val_auc_scores = []  # to store AUC scores

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        for batch in train_dataloader:
            images = batch["outfit_images"]
            texts = batch["outfit_texts"]
            labels = batch["outfit_labels"]

            logging.info(f"BATCH - images.shape: {images.shape}")
            logging.debug(f"batch - texts: {texts}")
            logging.debug(f"batch - labels: {labels}")

            optimizer.zero_grad()
            outputs = model(images, texts)
            loss = focal_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            mlflow.log_metric("training loss", loss.item())

        # Print the average loss for the epoch
        train_losses.append(loss.item() / len(train_dataloader))

        # region Validation Phase
        print(f"[VALIDATION]")
        test(
            model=model,
            focal_loss=focal_loss,
            epoch=epoch,
            dataloader=valid_dataloader,
            losses=val_losses,
            auc_scores=val_auc_scores,
            is_validating=True,
        )

        # endregion

        save_checkpoint(
            model.state_dict(), polyvore_split, f"model_epoch_{epoch + 1}.pth"
        )

        # Adjust the learning rate as needed (reduce by half in steps of 10)
        if (epoch + 1) % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] / 2

    mlflow.pytorch.log_model(
        pytorch_model=model,
        registered_model_name="outfit-recommendation",
        artifact_path="outfit-recommendation",
    )

    mlflow.pytorch.save_model(
        pytorch_model=model,
        path=os.path.join("outfit-recommendation", "trained_model"),
    )

    show_plots(
        plots=[
            {"values": train_losses, "label": "Train Losses"},
            {"values": val_losses, "label": "Validation Losses"},
        ],
        label="Losses",
    )
    show_plots(
        plots=[{"values": val_auc_scores, "label": "Validation AUC"}], label="AUC"
    )

    test_dataset = OutfitDataset(data_dir, polyvore_split, split_test, transform)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=default_batch_size,
        shuffle=True,
        collate_fn=custom_collate,
    )

    total_test_num_outfits = len(test_dataloader.dataset)
    print(f"[TEST] total_test_num_outfits: {total_test_num_outfits}")

    mlflow.log_metric("total_test_num_outfits", total_test_num_outfits)

    test_losses = []  # to store validation losses
    test_auc_scores = []  # to store AUC scores
    test(
        model=model,
        focal_loss=focal_loss,
        epoch=0,
        dataloader=test_dataloader,
        losses=test_losses,
        auc_scores=test_auc_scores,
        is_validating=False,
    )
    show_plots(plots=[{"values": test_losses, "label": "Test Losses"}], label="Loss")
    show_plots(plots=[{"values": test_auc_scores, "label": "Test AUC"}], label="AUC")

    mlflow.end_run()


def show_plots(plots, label):
    for plot in plots:
        plt.plot(plot["values"], label=plot["label"])

    plt.xlabel("Epoch")
    plt.ylabel(label)
    plt.legend()
    plt.show()


def test(model, focal_loss, epoch, dataloader, losses, auc_scores, is_validating):
    model.eval()
    all_val_labels = []
    all_val_outputs = []

    with torch.no_grad():
        for val_batch in dataloader:
            val_images = val_batch["outfit_images"]
            val_texts = val_batch["outfit_texts"]
            val_labels = val_batch["outfit_labels"]

            val_outputs = model(val_images, val_texts)
            val_loss = focal_loss(val_outputs, val_labels)

            all_val_labels.extend(val_labels.cpu().numpy())
            all_val_outputs.extend(val_outputs.cpu().numpy())

            if is_validating:
                mlflow.log_metric("val_loss", val_loss.item())

    losses.append(val_loss.item() / len(dataloader))

    # Calculate AUC score
    auc = roc_auc_score(all_val_labels, all_val_outputs)

    if is_validating:
        mlflow.log_metric("val_auc_score", auc)
    else:
        mlflow.log_metric("test_auc_score", auc)

    auc_scores.append(auc)
    print(f"Epoch {epoch + 1}, AUC: {auc}")


if __name__ == "__main__":
    main()
