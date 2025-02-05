"""This file contains the evaluation code for the PFN.

See installation instructions and other relevant doc in train.py.

To run this file, first train.py needs to be executed, or at least a model.pth has to
be present under the prior_fitted_model directory.

RUNNING INSTRUCTIONS:
python evaluate_pfn.py

OUTPUT:
The evaluation results as a json file, containing a dictionary mapping from benchmarks
to the loss and accuracy achieved on them by the model.
"""

import argparse
import json
import traceback
from typing import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from pfns.transformer import TransformerModel
from torch_geometric.data import Data
from torch_geometric.datasets import (Actor, Amazon, Coauthor,
                                      HeterophilousGraphDataset, Planetoid,
                                      Twitch, WebKB, WikiCS, WikipediaNetwork)
from train_pfn import Prior
from utils import Output, data_split, equalize_splits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_on_benchmarks(
    model: torch.nn.Module,
    benchmarks: list,
    neig: int | None = None,
    node_deg=False,
    validation_seed: int | None = None,
    nsplit: int | None = None,
    attention: str | None = None,
    supervised=False,
) -> list[OrderedDict]:
    """Evaluate the model on all benchmarks and return the accuracy on them as dict."""

    planetoid = ["cora", "citeseer", "pubmed"]  # undirected

    webkb = ["texas", "wisconsin", "cornell"]  # directed

    twitch = [
        "twitch-de",
        "twitch-en",
        "twitch-es",
        "twitch-fr",
        "twitch-pt",
        "twitch-ru",
    ]  # wandb fails to log it otherwise
    amazon = ["computers", "photo"]
    coauthor = ["cs", "physics"]
    wikipedia_network = ["chameleon", "squirrel"]
    heterophilous = [
        "roman-empire",
        "amazon-ratings",
        "minesweeper",
        "tolokers",
        "questions",
    ]

    bench_acc, bench_loss = OrderedDict(), OrderedDict()

    roc_auc_benchs = ["minesweeper", "tolokers"]

    benchs_lower = [s.lower() for s in benchmarks]

    for bench in sorted(benchs_lower):
        c_bench = bench.capitalize()
        if bench in benchs_lower:
            if bench in planetoid:
                dataset = Planetoid(root=f"datasets/{bench}", name=c_bench)
            elif bench in webkb:
                dataset = WebKB(root=f"datasets/{bench}", name=c_bench)
            elif bench in twitch:
                dataset = Twitch(
                    root=f"datasets/{bench}", name=c_bench.split("-")[1].upper()
                )
                c_bench = bench
            elif bench in amazon:
                dataset = Amazon(root=f"datasets/{bench}", name=c_bench)
            elif bench == "wikics":
                dataset = WikiCS(root=f"datasets/{bench}")
                dataset.name = "actor"
            elif bench in coauthor:
                dataset = Coauthor(root=f"datasets/{bench}", name=c_bench)
            elif bench in wikipedia_network:
                dataset = WikipediaNetwork(root=f"datasets/{bench}", name=c_bench)
            elif bench in heterophilous:
                dataset = HeterophilousGraphDataset(
                    root=f"datasets/{bench}", name=c_bench
                )
            elif bench == "actor":
                dataset = Actor(root=f"datasets/{c_bench}")
                dataset.name = "actor"
            else:
                print(f"Invalid benchmark: {bench}. Skipping...")
                continue
            try:
                (
                    bench_acc[bench + "_accuracy"],
                    bench_loss[bench + "_loss"],
                ) = evaluate_on_benchmark(
                    model=model,
                    dataset=dataset,
                    neig=neig,
                    node_deg=node_deg,
                    nsplit=nsplit,
                    validation_seed=validation_seed,
                    attention=attention,
                    supervised=supervised,
                )
            except Exception as e:
                print(f"An error occurred while evaluating on {dataset.name}\n{e}")
                traceback.print_exc()

        else:
            bench_acc[bench + "_accuracy"] = None
            bench_loss[bench + "_loss"] = None
    return bench_acc, bench_loss


def evaluate_on_benchmark(
    model: torch.nn.Module,
    dataset: Planetoid | WebKB | Twitch,
    neig: int | None = None,
    node_deg=False,
    validation_seed: int | None = None,
    nsplit: int | None = None,
    attention: str | None = None,
    supervised=False,
) -> tuple:
    """Evaluate the model on one of the benchmarks.

    It returns the accuracy, which is 0 if we can't feed the graph through the model."""
    data: Data = dataset[0].to(device)

    if "train_mask" not in data or supervised:
        data = data_split(data, nsplit=nsplit, seed=validation_seed)

    data = equalize_splits(data)

    # alleviate cuda memory fragmentation
    torch.cuda.empty_cache()

    bench_nfeat = data.x.shape[-1]
    model_nfeat = model.encoder.in_features
    train_eval_diff = model_nfeat - bench_nfeat

    if train_eval_diff < 0:
        print(
            f"Can't evaluate on {dataset.name}, as the transformer wasn't "
            f"trained with enough features. There is a feature difference of "
            f"{train_eval_diff}"
        )
        return -100, -100, -100

    if isinstance(model, TransformerModel):
        if not model_nfeat:
            raise AttributeError(
                "model_num_features has to be given when evaluating transformer."
            )
        model_nout = model.decoder_dict.standard[2].out_features
        data_nclass = len(data.y.unique())
        if model_nout < data_nclass:
            raise AttributeError(
                f"PFN model's output dimension is {data_nclass - model_nout} less than "
                "the number of classes in the benchmark."
            )

    accuracy, loss = [], []
    # iterate over possible train-test splits and return the mean accuracy and
    # loss; if there is only one split we just return that

    nsplit = data.train_mask.size(1)
    for i in range(nsplit):
        if isinstance(model, TransformerModel):
            # shuffle data, bring training nodes to the front of the dataset, and move
            # test ndoes after them
            if supervised:
                perm = torch.cat(
                    (
                        torch.nonzero(data.train_mask[:, i]),
                        torch.nonzero(data.val_mask[:, i]),
                        torch.nonzero(data.test_mask[:, i]),
                    )
                ).squeeze()
            else:
                # in the semi-supervised setting, most nodes are treated as unlabeled
                # in the benchmarks, so we treat the test set as all nodes except
                # the ones in the training set, but then we only measure accuracy on
                # the original test nodes and not the ones that are unlabeled to
                # make for a fairer comparison with other models
                unlabeled_mask = torch.logical_not(
                    torch.logical_or(
                        torch.logical_or(
                            data.train_mask[:, i],
                            data.val_mask[:, i],
                        ),
                        data.test_mask[:, i],
                    )
                )
                perm = torch.cat(
                    (
                        # train starts here
                        torch.nonzero(data.train_mask[:, i]),
                        torch.nonzero(data.val_mask[:, i]),
                        # train ends here, test starts
                        torch.nonzero(unlabeled_mask),
                        torch.nonzero(data.test_mask[:, i]),
                        # test ends
                    )
                ).squeeze()
            data.x = data.x[perm]
            data.y = data.y[perm]
            num_nodes = data.x.size(0)

            # shuffle edge_index as well accordingly, so that nodes that were originally
            # connected in the graph, remain connected after the node shuffling above
            if not model.full_attention:
                mapping = {perm[i].item(): i for i in range(num_nodes)}
                data.edge_index = torch.tensor(
                    [[mapping[node.item()] for node in row] for row in data.edge_index],
                    device=device,
                )

            data.split = 1  # dummy split
            # as we load the model with the already agumented features,
            # we need to undo the augmentation to calcuate the original num. of features
            original_nfeat = model_nfeat - neig - (1 if node_deg else 0)
            batch = Prior.graphs2batch(
                [data],
                max_nfeat=original_nfeat,
                neig=neig,
                node_deg=node_deg,
                attention=attention,
                split_name="random",
                attention_nhead=4,
            )
            # batch has dim (nnode, batch_size, nfeat)

            # define the dataset split between the train and test nodes
            # first n number of nodes are the training nodes, and the rest are the test
            train_mask = data.train_mask[
                :, i
            ]  # do not reveal labels for validation set
            split_pos = torch.sum(train_mask).item()
            test_start_pos = -data.test_mask[:, i].sum().item()

            with torch.no_grad(), torch.autocast("cuda"):
                # test the model on the data
                logits = model(
                    batch.x[:split_pos],
                    batch.y[:split_pos],
                    batch.x[split_pos:],
                    src_mask=batch.src_mask,
                )
                # only consider logits for the test set and ignore the nodes that
                # should be treated as unlabeled according to the benchmark split
                logits = logits[test_start_pos:]

                # cast based on the model loss function
                if isinstance(model.criterion, nn.BCEWithLogitsLoss):
                    batch.y = batch.y.to(logits.dtype)
                else:
                    batch.y = batch.y.to(torch.long)

                loss.append(
                    model.criterion(
                        logits.squeeze(), batch.y[test_start_pos:].squeeze()
                    )
                    .mean()
                    .item()
                )

            if isinstance(model.criterion, nn.BCEWithLogitsLoss):
                # binary classification
                probabilities = torch.sigmoid(logits)
                pred = (probabilities > 0.5).int()
            else:
                probabilities = torch.softmax(logits, dim=-1)
                pred = logits.argmax(dim=-1)

            accuracy.append((pred == batch.y[test_start_pos:]).float().mean().item())

    return np.mean(accuracy), np.mean(loss)


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="Seed to use for the run")
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="cora,citeseer,actor,minesweeper,tolokers,pubmed,wikics,chameleon,squirrel,texas,cornell,wisconsin,twitch-pt,computers,photo",
        help="name of benchmarks to models evaluate on, e.g. cora,citeseer",
    )
    parser.add_argument(
        "--nfeat",
        type=int,
        default=3703,
        help="The number of features the model was trained with",
    )
    parser.add_argument(
        "--nsplit",
        type=int,
        default=1,
        help="Max number of splits to use for each benchmark if it has any",
    )
    parser.add_argument(
        "--supervised",
        action="store_true",
        default=False,
        help="""evaluate pfn in supervised setting instead of semi-supervised setting on
         benchmarks where the default is semi-supervised""",
    )
    parser.add_argument(
        "--attention",
        type=str,
        default="adjecancy",
        help="adjecancy, full",
    )
    parser.add_argument(
        "--neig",
        type=int,
        default=0,
        help="Number of eigenvectors used for laplacian encoding that that are prepended"
        " to the node features.",
    )
    parser.add_argument(
        "--node_deg",
        action="store_true",
        default=True,
        help="Whether to prepend node degree to the features, set to True or False",
    )

    args = parser.parse_args()

    if args.benchmarks:
        if isinstance(args.benchmarks, list):
            args.benchmarks = ",".join(args.benchmarks)

        args.benchmarks = [b.lower() for b in args.benchmarks.split(",")]

    return args


if __name__ == "__main__":
    args = read_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(Output.model_path, map_location=device)

    result = evaluate_on_benchmarks(
        model,
        benchmarks=args.benchmarks,
        neig=args.neig,
        node_deg=args.node_deg,
        nsplit=args.nsplit,
        attention=args.attention,
        validation_seed=args.seed,
        supervised=args.supervised,
    )

    print(f"Losses and accuracies of {Output.model_path} on all benchmarks:\n{result}")

    with open(Output.pfn_result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"Done writing results to {Output.pfn_result_path}")
