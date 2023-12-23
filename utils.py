import os
import torch


def save_checkpoint(state, dataset, filename="checkpoint.pth.tar"):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)


def get_dimensions_and_lengths(lst):
    dimensions = 0
    lengths = []

    while isinstance(lst, list):
        dimensions += 1
        lengths.append(len(lst))
        lst = lst[0] if len(lst) > 0 else None

        return dimensions, lengths


def get_dict_first_n_items(dict, n):
    first_n_dict_items = {key: dict[key] for key in list(dict)[:n]}
    return first_n_dict_items
