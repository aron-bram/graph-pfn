"""This file contains the training and evaluation of some GNNs and an MLP baseline.

To run this file with the default settings:

`python baseline_training.py`

example usage with custom settings

`python baseline_training.py --epochs 202 --supervised`

This script also accepts optional args, see bottom of the file for explanations for each

Most parameters and other settings for the grid search are taken from the original
GCN paper: https://arxiv.org/pdf/1609.02907v4
"""

import argparse
import itertools
import json
import time
import traceback
from copy import deepcopy

import numpy as np
import torch
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.datasets import (Actor, Amazon, HeterophilousGraphDataset,
                                      Planetoid, Twitch, WebKB, WikiCS,
                                      WikipediaNetwork)
from torch_geometric.nn import GCN, MLP

from utils import Output, data_split, equalize_splits

transform = T.Compose([T.ToSparseTensor()])

# Dataset loaders
dataset_loaders = {
    # Planetoid datasets
    "Cora": Planetoid(root="datasets/cora", name="Cora", transform=transform),
    "CiteSeer": Planetoid(
        root="datasets/citeseer", name="CiteSeer", transform=transform
    ),
    "PubMed": Planetoid(root="datasets/pubmed", name="PubMed", transform=transform),
    "WikiCS": WikiCS(root="datasets/wikics", transform=transform),
    # Amazon datasets
    "Computers": Amazon(
        root="datasets/computers", name="computers", transform=transform
    ),
    "Photo": Amazon(root="datasets/photo", name="photo", transform=transform),
    # WebKB datasets
    "Texas": WebKB(root="datasets/texas", name="texas", transform=transform),
    "Wisconsin": WebKB(
        root="datasets/wisconsin", name="wisconsin", transform=transform
    ),
    "Cornell": WebKB(root="datasets/cornell", name="cornell", transform=transform),
    # Wikipedia Network datasets
    "Chameleon": WikipediaNetwork(
        root="datasets/wikipedia", name="chameleon", transform=transform
    ),
    "Squirrel": WikipediaNetwork(
        root="datasets/wikipedia", name="squirrel", transform=transform
    ),
    # Twitch dataset
    "TwitchPT": Twitch(root="datasets/twitch-pt", name="PT", transform=transform),
    # Heterophilous Datasets
    "Minesweeper": HeterophilousGraphDataset(
        "datasets/minesweeper", name="Minesweeper", transform=transform
    ),
    "Tolokers": HeterophilousGraphDataset(
        "datasets/tolokers", name="Tolokers", transform=transform
    ),
    "Actor": Actor(root="datasets/actor", transform=transform),
}


def train(model, optimizer, scheduler, data, train_mask):
    model.train()
    optimizer.zero_grad()
    if isinstance(model, MLP):
        out = model(data.x)
    else:
        out = model(data.x, data.adj_t)
    loss = CrossEntropyLoss()(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    # scheduler.step()
    return loss.item()


def test(model, data, test_mask, metric):
    model.eval()
    if isinstance(model, MLP):
        logits = model(data.x)
    else:
        logits = model(data.x, data.adj_t)

    test_loss = CrossEntropyLoss()(logits[test_mask], data.y[test_mask])

    if metric == "acc":
        test_acc = (
            logits[test_mask].argmax(dim=1) == data.y[test_mask]
        ).sum() / test_mask.sum().float()
        score = test_acc.item()
    elif metric == "roc_auc":
        # Assuming binary classification
        y_true = data.y[test_mask].detach().cpu().numpy()
        y_scores = logits[test_mask].detach().cpu().numpy()[:, 1]
        score = roc_auc_score(y_true, y_scores)
    else:
        supp = ["roc_auc", "acc"]
        raise ValueError(f"metric {metric} is not supported. Use one from {supp}")

    return test_loss.item(), score


def run_experiment(args, data, model_class, metric="acc"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    incumbent = {}
    best_val_loss = float("inf")  # stores best val loss found on all splits

    if args.supervised or "train_mask" not in data:
        data = data_split(data, nsplit=args.nsplit, seed=args.validation_seed)

    data = equalize_splits(data)
    data.nsplit = min(args.nsplit, data.train_mask.size(1))

    epochs = args.epochs

    hps = {
        "nlayer": [int(v.strip()) for v in args.nlayer.split(",")],
        "hs": [int(v.strip()) for v in args.hs.split(",")],
        "drop": [float(v.strip()) for v in args.drop.split(",")],
        "lr": [float(v.strip()) for v in args.lr.split(",")],
        "l2reg": [float(v.strip()) for v in args.l2reg.split(",")],
    }

    # grid search over hyperparameters
    for nlayer, hs, drop, lr, l2reg in itertools.product(*hps.values()):
        print(
            f"Evaluating model with {dict({'nlayer': nlayer, 'hs': hs, 'drop': drop, 'lr': lr, 'l2reg': l2reg})}"
        )
        split_val_losses = []
        split_test_scores = []

        for split_idx in range(data.nsplit):
            print(f"Split {split_idx}")
            # initialize model and optimizer
            model = model_class(
                in_channels=data.num_features,
                hidden_channels=hs,
                out_channels=data.num_classes,
                num_layers=nlayer,
                dropout=drop,
            ).to(device)
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=l2reg)
            # scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
            # they do not use scheduler in the original paper
            scheduler = None
            # early stopping criteria
            patience = args.patience
            best_split = {"model": None, "epoch": 0, "val_loss": float("inf")}
            early_stop_counter = 0

            print(f"Started training for {epochs} epochs...")

            # training for 200 epochs with early stopping
            for epoch in range(epochs):
                train_loss = train(
                    model, optimizer, scheduler, data, data.train_mask[:, split_idx]
                )
                val_loss, val_acc = test(
                    model, data, data.val_mask[:, split_idx], metric
                )

                # early stopping logic
                if val_loss < best_split["val_loss"]:
                    # save best model so far up to this epoch on the current split
                    best_split["val_loss"] = val_loss
                    best_split["model_state_dict"] = deepcopy(model.state_dict())
                    best_split["epoch"] = epoch
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if early_stop_counter >= patience:
                    break

            model.load_state_dict(best_split["model_state_dict"])

            # val loss on one split with one hp setting
            split_val_losses.append(best_split["val_loss"])

            # optionally refit to the train + validation set
            if args.refit:
                train_val_mask = torch.logical_or(
                    data.train_mask[:, split_idx], data.val_mask[:, split_idx]
                )
                # we retrain until the epoch where we stopped
                for epoch in range(early_stop_counter):
                    train_loss = train(
                        model, optimizer, scheduler, data, train_val_mask
                    )

            _, split_test_score = test(
                model, data, data.test_mask[:, split_idx], metric
            )
            split_test_scores.append(split_test_score)

        print(f"Validation loss on each split: {split_val_losses}")

        avg_val_loss = np.mean(split_val_losses)
        avg_test_score = np.mean(split_test_scores)

        print(f"Mean validation loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            print("Above is the new incumbent HP setting found on all splits")
            best_val_loss = avg_val_loss
            incumbent_test_score = avg_test_score
            score_key = "acc" if metric == "acc" else "roc_auc"
            incumbent = {
                score_key: incumbent_test_score,
                "nlayer": nlayer,
                "hs": hs,
                "dropout": drop,
                "epochs": epoch,
                "lr": lr,
                "l2reg": l2reg,
            }

    return incumbent


def main(args):
    # iterate over all datasets except the excluded ones
    excluded_datasets = [
        "CS",
        "Amazon-Ratings",
        "Roman-Empire",
        "Twitch-DE",
        "Twitch-EN",
        "Twitch-ES",
        "Twitch-FR",
        "Twitch-RU",
    ]
    included_datasets = [b.lower() for b in args.benchmarks.split(",")]

    incumbent = {"gcn": {}, "mlp": {}}

    for dataset_name, dataset in dataset_loaders.items():
        if (
            dataset_name in excluded_datasets
            or dataset_name.lower() not in included_datasets
        ):
            continue
        incumbent["gcn"][dataset_name] = {}

        # train MLP and GCN on the dataset
        print(f"Running on {dataset_name}...")

        data = dataset[0]
        data.num_classes = dataset.num_classes
        metric = "acc" if data.num_classes != 2 else "roc_auc"
        try:
            print("With GCN:")
            start_time = time.time()
            incumbent["gcn"][dataset_name] = run_experiment(args, data, GCN, metric)
            incumbent["gcn"][dataset_name]["runtime"] = time.time() - start_time

            print("With MLP:")
            start_time = time.time()
            incumbent["mlp"][dataset_name] = run_experiment(args, data, MLP, metric)
            incumbent["mlp"][dataset_name]["runtime"] = time.time() - start_time

        except Exception:
            print(f"An exception occurred while evaluating models on {dataset_name}:")
            traceback.print_exc()

    print("\nFinal benchmark accuracies:\n")
    for model, datasets in incumbent.items():
        print(f"Results for {model.upper()}:")
        for dataset, result in datasets.items():
            metric = "acc" if "acc" in result.keys() else "roc_auc"
            print(f"{metric} on {dataset}: {result[metric]}")
            rest = {k: v for k, v in result.items() if k != metric}
            [print(f"{k}: {v} | ", end="") for k, v in rest.items()]
            print()
        print("\n")

    Output.output_folder.mkdir(parents=True, exist_ok=True)

    with open(Output.baseline_result_path, "w", encoding="utf-8") as f:
        json.dump(incumbent, f, ensure_ascii=False, indent=4)

    print(f"Done writing results to {Output.baseline_result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_name", type=str, default=None, help="Job name")
    parser.add_argument("--job_id", type=str, default=None, help="Job ID")
    parser.add_argument(
        "--nlayer",
        type=str,
        default=None,
        help="number of layer(s) for the models to include in grid search, e.g 1,2,3",
    )
    parser.add_argument(
        "--hs",
        type=str,
        default=None,
        help="number of hidden units for the models to include in grid search, e.g 16,32",
    )
    parser.add_argument(
        "--drop",
        type=str,
        default=None,
        help="dropout rate(s) for the models to include in grid search, e.g 0.1,0.2",
    )
    parser.add_argument(
        "--lr",
        type=str,
        default=None,
        help="learning rate(s) of the models to include in grid search, e.g 0.01",
    )
    parser.add_argument(
        "--l2reg",
        type=str,
        default=None,
        help="Weight decay value(s) for Adam optimizer to include in grid search, e.g 1e-5",
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed to use for the run")
    parser.add_argument(
        "--epochs", type=int, default=200, help="Epochs to train models for"
    )
    parser.add_argument("--patience", type=int, default=10, help="Early stop patience")
    parser.add_argument(
        "--nsplit", type=int, default=10, help="Number of splits to evaluate on"
    )
    parser.add_argument("--commit_url", type=str, default=None, help="current commit")
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="cora,citeseer,actor,minesweeper,tolokers,pubmed,wikics,chameleon,squirrel,texas,cornell,wisconsin,twitch-pt,computers,photo",
        help="name of benchmarks to models evaluate on, e.g. cora,citeseer",
    )
    parser.add_argument(
        "--validation_seed", type=int, default=1, help="seed used to sample splits"
    )
    parser.add_argument(
        "--supervised",
        action="store_true",
        default=False,
        help="whether to evaluate in a supervised setting",
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        default=False,
        help="whether to refit to train+val and then test",
    )
    args = parser.parse_args()

    args_dict = vars(args)

    # HP search space defaults
    hps = {
        "nlayer": "2",
        "hs": "8, 16, 32, 64, 128, 256",
        "drop": "0.2, 0.4, 0.5, 0.6, 0.8",
        "lr": "0.01, 0.03, 0.05",
        "l2reg": "5e-4, 1e-4, 1e-5",
    }
    # replace missing HP values in args with values from hps
    for key, default_value in hps.items():
        if args_dict[key] is None:
            args_dict[key] = default_value

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    main(args)
