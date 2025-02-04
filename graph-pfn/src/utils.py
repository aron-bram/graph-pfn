import argparse
import traceback
from pathlib import Path
from typing import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from pfns.transformer import TransformerModel
from scipy.sparse.linalg._eigen.arpack.arpack import ArpackError
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.datasets import (Actor, Amazon, Coauthor,
                                      HeterophilousGraphDataset, Planetoid,
                                      Twitch, WebKB, WikiCS, WikipediaNetwork)
from torch_geometric.utils import degree



def data_split(
    data: Data, nsplit=10, val_size=0.2, test_size=0.2, stratify=False, seed=42
):
    """Create train/val/test splits on the {data} object."""
    labels = data.y.cpu().numpy()
    num_nodes = data.num_nodes

    # initialize empty masks for storing all splits (shape: [num_nodes, num_splits])
    train_masks = torch.zeros((num_nodes, nsplit), dtype=torch.bool)
    val_masks = torch.zeros((num_nodes, nsplit), dtype=torch.bool)
    test_masks = torch.zeros((num_nodes, nsplit), dtype=torch.bool)

    # loop over the number of splits and create train/val/test splits
    for split_num in range(nsplit):
        current_seed = seed + split_num  # Ensure each split has a different seed

        # perform (optinally stratified) train/test split
        train_indices, temp_indices, train_labels, temp_labels = train_test_split(
            range(num_nodes),
            labels,
            test_size=(val_size + test_size),
            random_state=current_seed,
            stratify=labels if stratify else None,
        )

        # split temp_indices into validation and test sets
        val_indices, test_indices, _, _ = train_test_split(
            temp_indices,
            temp_labels,
            test_size=test_size / (test_size + val_size),
            random_state=current_seed,
            stratify=temp_labels if stratify else None,
        )

        # create boolean masks for this split
        train_masks[train_indices, split_num] = True
        val_masks[val_indices, split_num] = True
        test_masks[test_indices, split_num] = True

    # add the masks to the data object
    data.train_mask = train_masks
    data.val_mask = val_masks
    data.test_mask = test_masks

    return data


def equalize_splits(data: Data):
    """Makes sure all splits on the data object have equal dimensions."""
    # if there is only a single train test split available, unsqueeze to allow iteration
    for mask in ["train_mask", "val_mask", "test_mask"]:
        if len(getattr(data, mask).shape) == 1:
            getattr(data, mask).unsqueeze_(dim=1)

    # determine the maximum size in the second dimension
    max_size = max(
        data.train_mask.size(1), data.val_mask.size(1), data.test_mask.size(1)
    )

    # expand each tensor to match max_size in the second dimension, if needed
    if data.train_mask.size(1) < max_size:
        data.train_mask = data.train_mask.repeat(1, max_size)

    if data.val_mask.size(1) < max_size:
        data.val_mask = data.val_mask.repeat(1, max_size)

    if data.test_mask.size(1) < max_size:
        data.test_mask = data.test_mask.repeat(1, max_size)

    return data