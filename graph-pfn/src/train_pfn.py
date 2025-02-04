"""This file demonstrates how to train a PFN on a GNN prior with a simplified interface.

PFN is a Prior-data Fitted Network, and the goal of this script is to train it to solve
node classification problems on graphs. To train the PFN, a graph generating function
is created, which is essentially the prior distribution over graphs. This can be
thought of as an inductive bias that describes what types of graphs may appear in the
set of problems that you want your PFN to be evaluated on.

The graph generation process roughly looks as follows:
1) Sample features at random
2) Sample the graph connectivity
3) Generate the labels for each node in the graph based on the features and the 
connectivity, by initializing a Graph Neural Network (GNN), and feeding the features
along with the connectivity though it

Now that we are able to generate graphs, we can train the PFN using the following steps:
4) Define the attention mask based on the graph connectivity
5) Add optional positional or structural encodings to the nodes' features, and create
a tabular dataset from these features and the labels
6) Randomly sample a train/test split for this dataset
7) Train the PFN on this dataset by revealing the features for all data points and
the labels for only the data points present in the train set during training, and
use cross entropy loss to measure the error of the predicitions on the test set, finally
backpropegating the loss through the model weights.
8) Repeat from step 1) for (pfn_epochs * pfn_steps_per_epochs) number of times

The final number of datasets that the PFN is trained on can be calculated as 
(batch_size * pfn_epochs * pfn_steps_per_epochs)

Note that in order for the PFN to learn about the graph connectivity or its structure,
one needs to encode this information in the dataset or in the PFN's architecture.
The PFN demonstrated here uses an adjecacncy-based attention, which simply
implies that each data point can only attend to its neighbors in the original graph
that the data point stems from. This encodes the direct neighborhood information
in the PFN's architecture. Additionally, the prior can have positional encodings like
Laplacian Eigenvectors (LPE), or Random Walk Positional Encodings (RWPE), or simple
node structural information, like node degree. These encodings can be prepended to each
data point, essentially augmenting the features of each data point by properties
of the graph or its nodes.

The PFN's predictions can be thought of as the aggregated prediction of an ensemble
of GNNs that are weigthed by the probability of that GNN and that dataset occuring in
the prior.


### RUNNING INSTRUCTIONS:

# optionally create a virtual environment
python -m venv ~/venv/pfn_gnn
source ~/venv/pfn_gnn/bin/activate

# then install the project requirements in your environment:
pip install --upgrade pip setuptools wheel
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric==2.4.0
pip install torch_scatter torch-sparse==0.6.16 -f https://data.pyg.org/whl/torch-2.1.1+cpu.html
pip install git+https://github.com/aron-bram/PFNs@f875a21

# after successfully installing the packages above, make sure you are in the
# directory where this file called gnn_prior_fitting.py is located first

# run this file using the command
python gnn_prior_fitting.py


### OUTPUT:

1) The output of this script is the trained PFN model saved under the folder it was run 
from in a subfolder called prior_fitted_model.
2) Along with all logs generated during training under prior_fitted_model/logs
"""

import argparse
import itertools
from pathlib import Path
import random
from argparse import Namespace
import sys
import traceback
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from pfns import encoders
from pfns.priors.prior import Batch
from pfns.train import train
from scipy.sparse.linalg._eigen.arpack.arpack import ArpackError
from torch_geometric.data import Data
from torch_geometric.nn import GAT, GCN, GIN, GraphSAGE
from torch_geometric.nn.conv import GINConv, MessagePassing
from torch_geometric.nn.models import MLP
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
from typing import Optional
# from torch_cluster import knn_graph
from torch_geometric.utils import (
    barabasi_albert_graph,
    coalesce,
    degree,
    erdos_renyi_graph,
    is_undirected,
    to_undirected,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# define a few helper classes for sampling prior parameters
class DummySampler:
    """A prior parameter that is fixed to a single value. Sampling always returns the same
    value."""

    def __init__(self, value, name) -> None:
        self.value = value
        self.name = name
        print(f"Fixing the value for '{name}' to {value[0]}")

    def sample(self, n=1, weights: list[float] | None = None) -> np.ndarray:
        if n == 1:
            return self.value[0]
        return np.full((n,), self.value[0])


class Sampler:
    """A generic parameter which can be sampled from.
    It recognizes which specific Arg to use for sampling based on the given input value.
    """

    sep = ","
    w_sep = "__"
    r_sep = "--"

    def __init__(self, value: str, name: str) -> None:
        self.value = value
        self.name = name
        if value is not None and any(
            "." == sep or "_" == sep or "-" == sep
            for sep in [self.sep, self.w_sep, self.r_sep]
        ):
            raise ValueError("Separators can't be assigned to '.' or '_' or '-'")
        self.float_sep = "."
        self.sampler = self._create_sampler(value)
        self.value = self.sampler.value
        self.max_val = max(self.sampler.value)

    def _create_sampler(self, value: str):
        """Decide to fix the value, or to sample it from a set or range of values."""
        if value is None:
            return DummySampler([value], self.name)

        if (
            isinstance(value, int)
            or isinstance(value, float)
            or isinstance(value, bool)
        ):
            # value is fixed to a float or int
            return DummySampler([value], self.name)
        elif not any(sep in value for sep in [self.sep, self.w_sep, self.r_sep]):
            if value.isdigit():
                # value is fixed to an int
                return DummySampler([int(value)], self.name)
            elif value.replace(self.float_sep, "").isdigit():
                # value is fixed to a float
                return DummySampler([float(value)], self.name)
            elif value == "True" or value == "False":
                # value is fixed to a boolean
                return DummySampler([value == "True"], self.name)
            else:
                # value is fixed to an alfanumerical
                return DummySampler([value], self.name)
        elif self.r_sep in value:
            # value should be sampled from a range
            return RangeSampler(value, self.name, self.r_sep)
        else:
            # value should be sampled from a set
            return ListSampler(value, self.name, self.sep, self.w_sep)

    def sample(self, n=1) -> np.ndarray | Any:
        if isinstance(self.sampler, RangeSampler):
            return self.sampler.sample(n)
        elif isinstance(self.sampler, ListSampler):
            return self.sampler.sample(n, self.sampler.weights)
        else:
            return self.sampler.sample(n)


class ListSampler:
    """A parameter that can be sampled from a list of values with optional weights.

    ex. input: '1,2,5__0.1,0.8,0.1' or 'er,ba,rand' or 'True,False__0.1,0.9'
    """

    def __init__(
        self, value: str, name: str, sep: str = ",", w_sep: str = "__"
    ) -> None:
        self.value = value
        self.name = name
        self.float_sep = "."

        v_w = value.split(w_sep) if w_sep in value else [value, ""]
        values = v_w[0].split(sep)
        weights = v_w[1].split(sep)
        if any(char.isalpha() for v in values for char in v):
            # values contains strings such as 'aa' or 'a2'
            if all(v == "False" or v == "True" for v in values):
                # values contains booleans
                values = [v == "True" for v in values]
        elif all(not char.isalpha() for v in values for char in v):
            # values does not contain any alpha characters
            if any(self.float_sep in v for v in values):
                # values contain floats
                values = [float(v) for v in values]
            else:
                # assume values contain ints
                values = [int(v) for v in values]
        else:
            raise ValueError(
                f"""Can't process argument '{name}' with value {value}, as its value is 
                likely invalid.\nTo fix the argument to a single value, use 
                something like 'ba' or '1.0', or to sample it use 'er;ba;rand' 
                or '0.1;1.0' or to do weighted sampling provide 'er;ba__0.2;0.8' 
                or 2;4__0.1;0.9' depending on the argument."""
            )
        if len(weights) == 1 and weights[0] == "":
            weights = [1] * len(values)
        else:
            weights = [float(w) for w in weights]

        # remove duplicate values in values and their according weights
        dups = {v: [] for v in values}
        for i, v in enumerate(values):
            dups[v].append(i)
        values = [values[v[0]] for k, v in dups.items()]
        self.value = values
        weights = [weights[v[0]] for k, v in dups.items()]
        if any([len(v) > 1 for v in dups.values()]):
            print(f"Found and removed duplicate values from '{name}'")
        # normalize weights
        self.weights = np.array(weights) / np.sum(weights)
        print(
            f"Setting sampling set for '{name}' to {self.value} with weights {self.weights}."
        )

    def sample(self, n=1, weights: list[float] | None = None) -> np.ndarray | Any:
        weights = self.weights if weights is None else weights
        if len(weights) != len(self.value):
            raise RuntimeError(
                f"""Tried to sample '{self.name}' from {self.value} with weights 
                {weights}, but len(weights) for sampling does not match the length of 
                that array."""
            )
        sample = np.random.choice(self.value, p=weights, size=n)
        if n == 1:
            return sample.item()
        return sample


class RangeSampler:
    """A parameter that can be sampled from a range.

    example input: '1--5' or '0.2--0.6'
    """

    def __init__(self, value: str, name: str, sep: str = "--", a: Optional[str]=None,) -> None:
        self.value = value
        self.name = name
        values = [v for v in value.split(sep)]
        self.float_sep = "."
        likely_float = values[0] == "0" and values[1] == "1"
        if self.float_sep in value or likely_float:
            self.value = [float(v) for v in values]
            self.min, self.max = float(values[0]), float(values[1])
        else:
            self.value = [int(v) for v in values]
            self.min, self.max = int(values[0]), int(values[1])
        print(f"Setting sampling range for '{name}' to [{self.min}, {self.max}).")

    def sample(self, n=1) -> np.ndarray | Any:
        if self.min == self.max:
            return self.min
        likely_float = self.min == 0 and self.max == 1
        if isinstance(self.min, float) or isinstance(self.max, float) or likely_float:
            sample = np.random.uniform(self.min, self.max, size=n)
            if n == 1:
                return sample.item()
            else:
                return sample

        sample = np.random.randint(self.min, self.max, size=n)
        if n == 1:
            return sample.item()
        return sample


class Prior:
    """
    The class describing the graph generating distribution that one can sample from.
    """

    def __init__(self, args):
        self.nclass = Sampler(args.nclass, "nclass")
        self.nfeat = Sampler(args.nfeat, "nfeat")
        self.nnode = Sampler(args.nnode, "nnode")
        self.feat_name = Sampler(args.feat_name.lower(), "feat_name")
        self.split_name = Sampler(args.split_name.lower(), "split_name")
        self.class_name = Sampler(args.class_name.lower(), "class_name")

        self.model_name = Sampler(args.model_name.lower(), "model_name")
        self.model_hs = Sampler(args.model_hs, "model_hs")
        self.model_dropout = Sampler(args.model_dropout, "model_dropout")
        self.model_nlayer = Sampler(args.model_nlayer, "model_nlayer")
        self.model_act = Sampler(args.model_act.lower(), "model_act")
        self.model_bias = Sampler(args.model_bias, "model_bias")
        self.model_nn_nlayer = Sampler(args.model_nn_nlayer, "model_nn_nlayer")
        self.model_nn_hs = Sampler(args.model_nn_hs, "model_nn_hs")
        self.model_nn_dropout = Sampler(args.model_nn_dropout, "model_nn_dropout")
        self.gnn_jk = Sampler(args.gnn_jk, "gnn_jk")

        self.graph_name = Sampler(args.graph_name.lower(), "graph_name")
        self.ba_nedge = Sampler(args.ba_nedge, "ba_nedge")
        self.er_p = Sampler(args.er_p, "er_p")
        self.fake_avgdeg = Sampler(args.fake_avgdeg, "fake_avgdeg")
        self.sparsebin_p = Sampler(args.sparsebin_p, "sparsebin_p")
        self.verbose = args.verbose

    def generate_pe(self, n, graph, device=None):
        """Generate @n many positional encodings based on the @graph."""
        graph.num_nodes = graph.num_nodes
        if False:
            pos_encoding = AddRandomWalkPE(num_eig, "pe")
        else:
            pos_encoding = AddLaplacianEigenvectorPE(
                k=n, attr_name="pe", is_undirected=True
            )  # ncv for scipy ARPACK solver, hardcoded value
        graph = pos_encoding(graph)
        if device is not None:
            graph.pe = graph.pe.to(device)

    @staticmethod
    def add_pe(graph: Data, neig=0, node_deg=False):
        """Adds positional and structural encoding to the graph."""
        nnode = graph.num_nodes
        nfeat = graph.x.size(-1)
        if node_deg:
            graph["node_deg"] = degree(graph.edge_index[0], num_nodes=nnode).unsqueeze(
                1
            )

        try:
            # add the laplacian positional encoding of each node to its feature
            if neig != 0:
                Prior.generate_pe(neig, graph, device)

        except ArpackError:
            i -= 1
            print(f"ArpackErorr, recreating graph with {nnode} nodes, {nfeat} feats.")
            pass

    @staticmethod
    def get_pes(graph: Data):
        """
        Get the positional encodings as a dict mapping from their name to their value.
        """
        if hasattr(graph, "pe"):
            if graph[0].pe.dim() < 2:
                raise ValueError("graph.pe dimension should be > 1")
            neig = graph.pe.size(-1)
        else:
            neig = 0

        node_deg = hasattr(graph, "node_deg")

        return {
            "neig": neig,
            "node_deg": node_deg,
        }

    @staticmethod
    def augment_x_with_pe(graph: Data):
        """Prepend positional encodings to the features."""
        pes = Prior.get_pes(graph)
        neig = pes["neig"]
        node_deg = pes["node_deg"]
        # prepend positional encodings to x => pe - x - zeros
        # x will take shape (nnode, PEs and Struct.PE + nfeat)
        graph.x = torch.cat(
            (
                graph.pe if neig != 0 else torch.empty(0, device=device),
                graph.node_deg if node_deg else torch.empty(0, device=device),
                graph.x,
            ),
            dim=-1,
        )

    @staticmethod
    def pad_x(graph: Data, max_nfeat: int, constant=0):
        """Pad the features with {constant} up to {max_nfeat}."""
        pes = Prior.get_pes(graph)
        neig = pes["neig"]
        node_deg = pes["node_deg"]
        npe = neig + 1 if node_deg else 0
        # pad_before = 0, pad_after = max_num_features + num_eigenvectors - num_features
        p1d = (0, max_nfeat + npe - graph.x.size(-1))
        graph.x = F.pad(graph.x, p1d, "constant", constant)

    class MulticlassRankFix(nn.Module):
        """Convert from float to categorical."""

        def __init__(self, num_classes, ordered_p=0.5):
            super().__init__()
            self.num_classes = num_classes
            self.ordered_p = ordered_p

        def forward(self, x):
            # x has shape (T,B,H)

            # CAUTION:
            # This samples the same idx in sequence for each class boundary in a batch
            class_boundaries = torch.randint(0, x.shape[0], (self.num_classes - 1,))
            class_boundaries = x[class_boundaries].unsqueeze(1)

            d = (x > class_boundaries).sum(axis=0)

            randomized_classes = torch.rand((d.shape[1],)) > self.ordered_p
            d[:, randomized_classes] = self.randomize_classes(
                d[:, randomized_classes], self.num_classes
            )
            reverse_classes = torch.rand((d.shape[1],)) > 0.5
            d[:, reverse_classes] = self.num_classes - 1 - d[:, reverse_classes]
            return d

        def randomize_classes(self, x, num_classes):
            classes = torch.arange(0, num_classes, device=x.device)
            random_classes = torch.randperm(num_classes, device=x.device).type(x.type())
            x = ((x.unsqueeze(-1) == classes) * random_classes).sum(-1)
            return x

    class ER:
        """Erdos Renyi type graph."""

        NAME = "Erdos Renyi"

        def __init__(self, p_samp: Sampler, verbose: int = 0) -> None:
            self.p_samp = p_samp
            self.verbose = verbose

        def edge_index(self, num_nodes: int) -> torch.Tensor:
            p = self.p_samp.sample()
            if self.verbose == 2:
                print(f"Graph type is {self.NAME} with {p=}.")
            return erdos_renyi_graph(num_nodes, p, directed=False).to(device)

    class BA:
        """Barabasi Albert type graph."""

        NAME = "Barabasi Albert"

        def __init__(self, nedge_samp: Sampler, verbose: int = 0) -> None:
            self.nedge_samp = nedge_samp
            self.verbose = verbose

        def edge_index(self, nnode: int) -> torch.Tensor:
            nedge = self.nedge_samp.sample()

            if self.verbose == 2:
                print(f"Graph type is {self.NAME} with {nedge} 'edges'.")
            assert isinstance(nedge, int)  # just for type hint
            return barabasi_albert_graph(nnode, nedge).to(device)

    class FakeGraph:
        def edge_index(
            self,
            num_nodes: int,
            avg_degree: int,
            is_undirected: bool = True,
        ) -> torch.Tensor:
            # num_nodes = get_num_nodes(num_nodes, avg_degree)
            num_src_nodes, num_dst_nodes = num_nodes, num_nodes
            num_edges = num_src_nodes * avg_degree
            row = torch.randint(
                num_src_nodes, (num_edges,), dtype=torch.int64, device=device
            )
            col = torch.randint(
                num_dst_nodes, (num_edges,), dtype=torch.int64, device=device
            )
            edge_index = torch.stack([row, col], dim=0)

            num_nodes = max(num_src_nodes, num_dst_nodes)
            if is_undirected:
                edge_index = to_undirected(edge_index, num_nodes=num_nodes)
            else:
                edge_index = coalesce(edge_index, num_nodes=num_nodes)

            return edge_index

    def sample_graphs(self, n: int, graph_callback=None, **kwargs):
        """Graph generating distribution that one can sample from."""
        # This function corresponds to points 1) - 3) in the documentation at the top of
        # this file

        # generate a list of graphs and return them
        graphs: list[Data] = []

        # num_features, num_nodes, and model remain the same for all graphs in the batch
        nfeat = self.nfeat.sample()

        nnode = self.nnode.sample()

        # create the model, which will generate the floating point labels
        model = self.get_model(nfeat).to(device)
        model.train()  # needed for dropout

        i = 0
        while i < n:
            # build a graph step-by-step:
            graph = Data()

            # add edges to the graph
            graph["edge_index"] = self.generate_edge_index(nnode)

            # add node features to the graph
            graph["x"] = self.generate_features(nnode=nnode, nfeat=nfeat)

            # randomly initialize the model and generate a float label for each node
            model.reset_parameters()  # type: ignore
            with torch.no_grad(), torch.autocast("cuda"):
                try:
                    out = model(graph.x, graph.edge_index)
                except TypeError:
                    # for models that don't directly make use of graph connectivity
                    out = model(graph.x)

            graph["y"] = self.generate_classes(out)

            graph_callback(graph) if isinstance(graph_callback, Callable) else None

            graphs.append(graph)

            i += 1
        return graphs

    @staticmethod
    def graphs2batch(graphs: list[Data], max_nfeat: int, neig=0, node_deg=False, split_name="random", attention="adjacency", attention_nhead=4):
        """Convert from graphs to a batch of X y pairs, a tabular dataset."""
        # This function corresponds to points 4) - 6) in the documentation at the top of
        # this file

        # this is inefficient, but for sake of simplicity we augment features as follows
        for graph in graphs:
            # optionally add structural and positional encodings to the graph object
            Prior.add_pe(graph, neig, node_deg)

            # augment the features with the positional encodings on the graph object
            Prior.augment_x_with_pe(graph)

            # pad the features of each node with zeros at the end until the maximum
            # number of features that the PFN accepts
            Prior.pad_x(graph, max_nfeat=max_nfeat)

        # x shape (nnode, batch_size, nfeat)
        x = torch.stack([graph.x for graph in graphs], dim=1)
        # y shape (num_nodes, batch_size)
        y = torch.stack([graph.y.float() for graph in graphs], dim=1)

        # define the same train-test split position for the whole batch
        split_pos = Prior.generate_train_test_split(split_name=split_name, nnode=len(graphs[0].y))

        batch = Batch(
            x=x,
            y=y,
            target_y=y,
            single_eval_pos=split_pos,
        )

        # if not using full attention, we also have to populate the attention matrix
        # per graph
        if attention == "adjacency":
            # create the attention masks, shape: (batch_size, num_nodes, num_nodes)
            batch.src_mask = torch.zeros((len(graphs), len(y), len(y)), device=x.device)
            for i, g in enumerate(graphs):
                edge_index = g.edge_index
                if not is_undirected(edge_index):
                    # make graph undirected
                    reverse_edge_index = torch.stack(
                        [edge_index[1], edge_index[0]], dim=0
                    )
                    edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
                batch.src_mask[i, edge_index[0], edge_index[1]] = float("-inf")
                # allow self attention
                batch.src_mask[i].fill_diagonal_(float(0.0))
            if len(graphs) > 1:
                # shape: (batch_size * num_heads, num_nodes, num_nodes)
                batch.src_mask = torch.repeat_interleave(
                    batch.src_mask, attention_nhead, dim=0
                )
            else:
                batch.src_mask.squeeze_()
        else:
            batch.__dict__["src_mask"] = None

        return batch

    def sample_batch(
        self,
        batch_size,
        neig,
        node_deg,
        attention,
        attention_nhead,
        batch_callback: Callable | None = None,
        **kwargs,  # kwargs needed for train()
    ):
        """Return a batch of tabular data used to train the PFN."""
        # This function encompasses points 1) - 6) in the documentation at the top of
        # this file

        # sample a set of graphs with different graph connectivity
        graphs = self.sample_graphs(
            n=batch_size,
            neig=neig,
            node_deg=node_deg,
            graph_callback=None,
            **kwargs,
        )

        # convert graphs into a batch
        batch = self.graphs2batch(graphs, self.nfeat.max_val, neig=neig, node_deg=node_deg, split_name=self.split_name.sample(), attention=attention, attention_nhead=attention_nhead)

        batch_callback(batch) if isinstance(batch_callback, Callable) else None

        return batch

    def get_model(self, nfeat: int):
        choice = self.model_name.sample()

        if choice == "gcn":
            return GCN(
                in_channels=nfeat,
                hidden_channels=self.model_hs.sample(),
                num_layers=self.model_nlayer.sample(),
                out_channels=1,
                dropout=self.model_dropout.sample(),
                act=self.model_act.sample(),
                bias=self.model_bias.sample(),
                jk=self.gnn_jk.sample(),
            )
        elif choice == "sage":
            return GraphSAGE(
                in_channels=nfeat,
                hidden_channels=self.model_hs.sample(),
                num_layers=self.model_nlayer.sample(),
                out_channels=1,
                dropout=self.model_dropout.sample(),
                act=self.model_act.sample(),
                bias=self.model_bias.sample(),
                jk=self.gnn_jk.sample(),
            )
        elif choice == "gat":
            return GAT(
                in_channels=nfeat,
                hidden_channels=self.model_hs.sample(),
                num_layers=self.model_nlayer.sample(),
                out_channels=1,
                dropout=self.model_dropout.sample(),
                act=self.model_act.sample(),
                bias=self.model_bias.sample(),
                jk=self.gnn_jk.sample(),
            )
        elif choice == "mlp":
            act = self.model_act.sample()
            nlayer = self.model_nlayer.sample()
            hs = self.model_hs.sample()
            bias = self.model_bias.sample()
            dropout = self.model_dropout.sample()
            mlp = nn.Sequential(
                *[
                    item
                    for tup in itertools.zip_longest(
                        *[
                            # linears
                            [
                                # input layer
                                nn.Linear(
                                    in_features=nfeat,
                                    out_features=hs,
                                    bias=bias,
                                ),
                                # hidden layers
                                *[
                                    nn.Linear(
                                        in_features=hs,
                                        out_features=hs,
                                        bias=bias,
                                    )
                                    for _ in range(nlayer - 1)
                                ],
                                # final projection layer
                                nn.Linear(
                                    in_features=hs,
                                    out_features=1,
                                    bias=bias,
                                ),
                            ],
                            # acts
                            [activation_resolver(act) for _ in range(nlayer)],
                            # dropouts
                            [nn.Dropout(p=dropout) for _ in range(nlayer)],
                        ]
                    )
                    for item in tup
                    if item is not None
                ]
            )
            mlp.reset_parameters = lambda: [
                module.reset_parameters()
                for module in mlp
                if hasattr(module, "reset_parameters")
            ]
            return mlp
        elif choice == "sage_argmax":
            return GraphSAGE(
                in_channels=nfeat,
                hidden_channels=self.model_hs.sample(),
                num_layers=self.model_nlayer.sample(),
                out_channels=self.nclass.max_val,
                dropout=self.model_dropout.sample(),
                act=self.model_act.sample(),
                bias=self.model_bias.sample(),
                jk=self.gnn_jk.sample(),
            )
        elif choice == "gin_argmax":
            return GraphSAGE(
                in_channels=nfeat,
                hidden_channels=self.model_hs.sample(),
                num_layers=self.model_nlayer.sample(),
                out_channels=self.nclass,
                dropout=self.model_dropout.sample(),
                act=self.model_act.sample(),
                bias=self.model_bias.sample(),
                jk=self.gnn_jk.sample(),
            )
        elif choice == "gin":
            return GIN(
                in_channels=nfeat,
                hidden_channels=self.model_hs.sample(),
                num_layers=self.model_nlayer.sample(),
                out_channels=1,
                dropout=self.model_dropout.sample(),
                act=self.model_act.sample(),
                bias=self.model_bias.sample(),
                jk=self.gnn_jk.sample(),
            )
        elif choice == "gin_mlp":

            class GINMLP(BasicGNN):
                """A modification of GIN with a more complex mlp."""

                supports_edge_weight = False
                supports_edge_attr = False

                def init_conv(
                    self,
                    in_channels: int,
                    out_channels: int,
                    nn_num_hidden_layers: int,
                    nn_hidden_size: int,
                    nn_dropout: float,
                    **kwargs,
                ) -> MessagePassing:
                    self.mlp = MLP(
                        [
                            in_channels,
                            *[nn_hidden_size] * nn_num_hidden_layers,
                            out_channels,
                        ],
                        act=self.act,
                        norm=None,
                        dropout=nn_dropout,
                    )
                    return GINConv(self.mlp, **kwargs)

            return GINMLP(
                in_channels=nfeat,
                hidden_channels=self.model_hs.sample(),
                num_layers=self.model_nlayer.sample(),
                out_channels=1,
                dropout=self.model_dropout.sample(),
                act=self.model_act.sample(),
                bias=self.model_bias.sample(),
                nn_num_hidden_layers=self.model_nn_nlayer.sample(),
                nn_dropout=self.model_nn_dropout.sample(),
                nn_hidden_size=self.model_nn_hs.sample(),
                jk=self.gnn_jk.sample(),
            )
        else:
            raise AttributeError(f"Model '{choice}' does not exist.")

    def generate_features(self, nnode, nfeat, batch_size=1):
        choice = self.feat_name.sample()

        if choice == "randbin":
            return (
                torch.randint(0, 2, (batch_size, nnode, nfeat), device=device)
                .squeeze(0)
                .float()
            )
        elif choice == "randfloat":
            return (
                torch.rand((batch_size, nnode, nfeat), device=device).squeeze(0).float()
            )
        elif choice == "sparsebin":
            return (
                torch.bernoulli(
                    torch.full((batch_size, nnode, nfeat), self.sparsebin_p.sample())
                )
                .squeeze(0)
                .float()
                .to(device)
            )
        else:
            raise ValueError(f"Value {choice} is not supported for feat_name.")

    def generate_edge_index(self, nnode):
        choice = self.graph_name.sample()

        if choice == "ba":
            ba = self.BA(
                nedge_samp=self.ba_nedge,
                verbose=self.verbose,
            )
            return ba.edge_index(nnode)
        elif choice == "er":
            er = self.ER(p_samp=self.er_p, verbose=self.verbose)
            return er.edge_index(nnode)
        elif choice == "fake":
            fake = self.FakeGraph()
            return fake.edge_index(nnode, self.fake_avgdeg.sample())
        else:
            raise ValueError(f"Graph type '{choice}' is not supported.")

    def generate_classes(self, y: torch.Tensor):
        # y stores the raw floating point outputs
        choice = self.class_name.sample()

        # t has shape (nnode, batch_size, nfeat)
        t = y.transpose(0, 1) if len(y.shape) == 3 else y.unsqueeze(1)

        if choice == "multiclassrankfix":
            labels = (
                (self.MulticlassRankFix(self.nclass.max_val)(t))
                .transpose(0, 1)
                .squeeze()
            )
        else:
            raise AttributeError(f"Label generator '{choice}' is not supported.")

        return labels

    @staticmethod
    def generate_train_test_split(split_name, nnode, min_split_pos=0):
        if split_name == "random":
            return random.choices(range(min_split_pos, nnode))[0]
        else:
            raise AttributeError(f"Train-test split '{split_name}' is not supported.")


def epoch_callback(model, epochs, data_loader, scheduler):
    """A function called at each epoch while training the PFN."""
    pass


def read_args(defaults):
    """Parse arguments passed to this program."""
    parser = argparse.ArgumentParser(
        prog="gnn_prior_fitting.py",
    )

    sep, r_sep, w_sep = Sampler.sep, Sampler.r_sep, Sampler.w_sep

    parser.add_argument(
        "--nclass",
        type=str,
        default=None,
        help=("Number of classes the transformer is trained with, i.e. 10"),
    ),
    parser.add_argument(
        "--nnode",
        type=str,
        default=None,
        help=(
            "Number of nodes each graph should consist of. Set to\n"
            "1) '12' to fix it to 12\n"
            f"2) '1{r_sep}5' to sample from 1 to 4 or\n"
            f"3) '10{sep}20{sep}100' to sample either 10 or 20 or 100\n"
            f"4) '10{sep}20{sep}100{w_sep}0.1{sep}0.3{sep}0.6' to do weighted sampling"
        ),
    )
    parser.add_argument(
        "--nfeat",
        type=str,
        default=None,
        help=(
            "Number of features to use for each node. Set to\n"
            "1) '12' to fix it to 12\n"
            f"2) '1{r_sep}5' to sample from 1 to 4 or\n"
            f"3) '10{sep}20{sep}100' to sample either 10 or 20 or 100\n"
            f"4) '10{sep}20{sep}100{w_sep}0.1{sep}0.3{sep}0.6' to do weighted sampling"
        ),
    )
    parser.add_argument(
        "--graph_name",
        type=str,
        default=None,
        help=(
            "Type of graph model to use to generate the edge index (sbm, er, ba). Set to\n"
            "1) 'ba' to fix it to 'ba'\n"
            f"2) 'ba{sep}er' to sample either 'ba' or 'er\n"
            f"3) 'ba{sep}er{w_sep}0.9{sep}0.1' to do weighted sampling"
        ),
    )
    parser.add_argument(
        "--feat_name",
        type=str,
        default=None,
        help=(
            "Type of feature generation to use (randbin, sparsebin, randfloat). Set to\n"
            "1) 'randbin' to fix it to 'randbin'\n"
            f"2) 'randbin_sparsebin' to sample either 'randbin' or 'sparsebin'\n"
            f"3) 'randbin{sep}sparsebin{w_sep}0.1{sep}0.9' to do weighted sampling"
        ),
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default=None,
        help=(
            "Type of label generation to use (multiclassrank, multiclassrankrand). Set to\n"
            "1) 'multiclassrank' to fix it to 'multiclassrank'\n"
            f"2) 'multiclassrank{sep}argmax' to sample either 'multiclassrank' or 'argmax'\n"
            f"3) 'multiclassrank{sep}argmax{w_sep}0.1{sep}0.9' to do weighted sampling"
        ),
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default=None,
        help=(
            "Type of feature generation to use (random, fair). Set to\n"
            "1) 'random' to fix it to 'random'\n"
            f"2) 'random{sep}fair' to sample either 'random' or 'fair'\n"
            f"3) 'random{sep}fair{w_sep}0.1{sep}0.9' to do weighted sampling"
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=(
            "Type of model to use to generate the labels (GCN, GraphSAGE, Linear, MLP). Set to\n"
            "1) 'sage' to fix it to 'sage'\n"
            f"2) 'gcn{sep}sage' to sample either 'gcn' or 'sage'\n"
            f"3) 'gcn{sep}sage{w_sep}0.1{sep}0.9' to do weighted sampling"
        ),
    )
    parser.add_argument(
        "--model_nlayer",
        type=str,
        default=None,
        help=(
            "Number of layers to use for the label generating model. Set to\n"
            "1) '2' to fix it to 2\n"
            f"2) '1{r_sep}5' to sample from 1 to 4 or\n"
            f"3) '10{sep}8{sep}5' to sample either 10 or 8 or 5\n"
            f"4) '10{sep}8{sep}5{w_sep}0.1{sep}0.3{sep}0.6' to do weighted sampling"
        ),
    )
    parser.add_argument(
        "--model_act",
        type=str,
        default=None,
        help=(
            "Type of activation function to use for the label generating model \
            (ReLU, Tanh, GELU, ELU, Sigmoid). Set to\n"
            "1) 'relu' to fix it to 'relu'\n"
            f"2) 'relu{sep}tanh' to sample either 'relu' or 'tanh'\n"
            f"3) 'relu{sep}tanh{w_sep}0.1{sep}0.9' to do weighted sampling"
        ),
    )
    parser.add_argument(
        "--model_hs",
        type=str,
        default=None,
        help=(
            "Hidden size of the label generating model. Set to\n"
            "1) '12' to fix it to 12\n"
            f"2) '1{r_sep}5' to sample from 1 to 4 or\n"
            f"3) '10{sep}20{sep}100' to sample either 10 or 20 or 100\n"
            f"4) '10{sep}20{sep}100{w_sep}0.1{sep}0.3{sep}0.6' to do weighted sampling"
        ),
    )
    parser.add_argument(
        "--model_dropout",
        type=str,
        default=None,
        help=(
            "Dropout rate to use for the model. Set to\n"
            "1) '0.5' to fix it to 0.5\n"
            f"2) '0.1{r_sep}0.9' to sample from 0.1 to 0.9 or\n"
            f"3) '0.2{sep}0.4{sep}0.6' to sample either 0.2 or 0.4 or 0.6\n"
            f"4) '0.2{sep}0.4{sep}0.6{w_sep}0.1{sep}0.3{sep}0.6' to do weighted sampling"
        ),
    )
    parser.add_argument(
        "--model_bias",
        type=str,
        default=None,
        help=(
            "Wether the label generating model should have bias terms. Set to\n"
            "1) 'True' to fix it to True\n"
            f"2) 'True{sep}False' to sample either True or False\n"
            f"3) 'True{sep}False{w_sep}0.4{sep}0.6' to do weighted sampling"
        ),
    )
    parser.add_argument(
        "--model_nn_nlayer",
        type=str,
        default=None,
        help=(
            "Type of normalization to use for the model. Set to\n"
            "1) 'batchnorm' to fix it to batchnorm\n"
            f"2) 'batchnorm{sep}layernorm{sep}graphnorm' to sample either batchnorm or layernorm or graphnorm\n"
            f"3) 'batchnorm{sep}layernorm{sep}graphnorm{w_sep}0.1{sep}0.3{sep}0.6' to do weighted sampling"
        ),
    )
    parser.add_argument(
        "--model_nn_dropout",
        type=str,
        default=None,
        help=(
            "Dropout rate to use for the model. Set to\n"
            "1) '0.5' to fix it to 0.5\n"
            f"2) '0.1{r_sep}0.9' to sample from 0.1 to 0.9 or\n"
            f"3) '0.2{sep}0.4{sep}0.6' to sample either 0.2 or 0.4 or 0.6\n"
            f"4) '0.2{sep}0.4{sep}0.6{w_sep}0.1{sep}0.3{sep}0.6' to do weighted sampling"
        ),
    )
    parser.add_argument(
        "--model_nn_hs",
        type=str,
        default=None,
        help=(
            "Size of the hidden layers in the neural net that transforms the node "
            "embeddings in a GNN layer. Set to\n"
            "1) '12' to fix it to 12\n"
            f"2) '1{r_sep}5' to sample from 1 to 4 or\n"
            f"3) '10{sep}20{sep}100' to sample either 10 or 20 or 100\n"
            f"4) '10{sep}20{sep}100{w_sep}0.1{sep}0.3{sep}0.6' to do weighted sampling"
        ),
    )
    parser.add_argument(
        "--gnn_jk",
        type=str,
        default=None,
        help="The jumping-knowledge mode to use with gnn models (None, max, lstm, cat)",
    )
    parser.add_argument(
        "--er_p",
        type=str,
        default=None,
        help=(
            "Erdos Renyi p. Set to\n"
            "1) '0.5' to fix it to 0.5\n"
            f"2) '0.1-0.9' to sample from 0.1 to 0.9 or\n"
            f"3) '0.2{sep}0.4{sep}0.6' to sample either 0.2 or 0.4 or 0.6\n"
            f"4) '0.2{sep}0.4{sep}0.6{w_sep}0.1{sep}0.3{sep}0.6' to do weighted sampling"
        ),
    )
    parser.add_argument(
        "--ba_nedge",
        type=str,
        default=None,
        help=(
            "Number of edges to use for the Barabasi Albert graph. Set to\n"
            "1) '12' to fix it to 12\n"
            f"2) '1-5' to sample from 1 to 4 or\n"
            f"3) '10{sep}20{sep}100' to sample either 10 or 20 or 100\n"
            f"4) '10{sep}20{sep}100{w_sep}0.1{sep}0.3{sep}0.6' to do weighted sampling"
        ),
    )
    parser.add_argument(
        "--fake_avgdeg",
        type=str,
        default=None,
        help=(
            "The avarage degree in a fake graph dataset. Set to\n"
            "1) '12' to fix it to 12\n"
            f"2) '1-5' to sample from 1 to 4 or\n"
            f"3) '10{sep}20{sep}100' to sample either 10 or 20 or 100\n"
            f"4) '10{sep}20{sep}100{w_sep}0.1{sep}0.3{sep}0.6' to do weighted sampling"
        ),
    )
    parser.add_argument(
        "--sparsebin_p",
        type=str,
        default=None,
        help=(
            "Bernoulli probability of 1 for sparse binary features. Set to\n"
            "1) '0.5' to fix it to 0.5\n"
            f"2) '0.1-0.9' to sample from 0.1 to 0.9 or\n"
            f"3) '0.2{sep}0.4{sep}0.6' to sample either 0.2 or 0.4 or 0.6\n"
            f"4) '0.2{sep}0.4{sep}0.6{w_sep}0.1{sep}0.3{sep}0.6' to do weighted sampling"
        ),
    )
    parser.add_argument(
        "--neig",
        type=int,
        default=None,
        help="Number of eigenvectors used for laplacian encoding that that are prepended"
        " to the node features.",
    )
    parser.add_argument(
        "--node_deg",
        type=str,
        default=None,
        help="Whether to prepend node degree to the features, set to True or False",
    )
    parser.add_argument(
        "-e",
        "--pfn_epochs",
        type=int,
        default=None,
        help="Number of epochs",
    )
    parser.add_argument(
        "-se",
        "--pfn_steps_per_epoch",
        type=int,
        default=None,
        help="Number of steps per epoch",
    )
    parser.add_argument(
        "--pfn_lr",
        type=float,
        default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=None,
        help="Number of datasets in a batch",
    )
    parser.add_argument(
        "--pfn_num_layers",
        type=int,
        default=None,
        help="PFN number of layers",
    )
    parser.add_argument(
        "--pfn_nhead",
        type=int,
        default=None,
        help="PFN number of attention heads (nhead): The number of heads in the "
        "multi-head attention models (num_heads) in PyTorch MultiheadAttention "
        "layer. Default: 4",
    )
    parser.add_argument(
        "--pfn_nhid",
        type=int,
        default=None,
        help="PFN hidden size (dim_feedforward): The dimension of the feedforward "
        "network model in nn.TransformerEncoderLayer. Default: 1024",
    )
    parser.add_argument(
        "--pfn_emsize",
        type=int,
        default=None,
        help="PFN embedding size (ninp/d_model): The number of expected features in "
        "the input (embed_dim in PyTorch MultiheadAttention layer. Default: 512",
    )
    parser.add_argument(
        "-b",
        "--benchmarks",
        type=str,
        default=None,
        help=f"List the name of benchmarks to evaluate on e.g. 'cora{sep}citeseer{sep}pubmed'",
    )
    parser.add_argument(
        "--attention",
        type=str,
        default=None,
        help="adjecancy, full",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=None,
        help="Specify the amount of information to log (0, 1, 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for pytorch and numpy",
    )
    parser.add_argument(
        "--aggregate_k_gradients",
        type=int,
        default=None,
        help="aggregate k gradients before updating the model",
    )
    parser.add_argument(
        "--nsplit",
        type=int,
        default=None,
        help="Max number of splits to use for each benchmark if it has any",
    )
    parser.add_argument(
        "--supervised",
        action="store_true",
        default=None,
        help="""evaluate pfn in supervised setting instead of semi-supervised setting on
         benchmarks where the default is semi-supervised""",
    )
    args = parser.parse_args()

    print("Initializing arguments with their defaults.")

    def update_parser_args(parser_args, update_dict):
        for key, value in update_dict.items():
            arg_exists = hasattr(parser_args, key)
            arg_is_none = getattr(parser_args, key, None) is None
            if arg_exists and arg_is_none:
                setattr(parser_args, key, value)

    update_parser_args(args, defaults.__dict__)

    # have consitency in the ordering of the benchmarks over runs
    if args.benchmarks:
        if isinstance(args.benchmarks, list):
            args.benchmarks = ",".join(args.benchmarks)

        args.benchmarks = [b.lower() for b in args.benchmarks.split(",")]

        if "twitch" in args.benchmarks:
            # for convenience
            args.benchmarks.remove("twitch")
            d = [
                "twitch-de",
                "twitch-en",
                "twitch-es",
                "twitch-fr",
                "twitch-pt",
                "twitch-ru",
            ]
            args.benchmarks.extend(d)
        args.benchmarks.sort()

    # output all args
    print(
        "Starting training with:\n"
        + "\n".join([f"{k} = {v}" for k, v in args.__dict__.items()])
    )

    return args


def defaults_settings() -> Namespace:
    """This function defines the default settings used to run an expiremnt.

    Documentation on each parameter can be found in the read_args function.

    Each entry in the dictionary defines the value for a parameter in the Prior, or
    in the model or its training.

    The values present here are the ones used to train the prior that yielded the
    results demonstrated in the paper.
    """
    args = Namespace(
        **{
            "nclass": 10,
            "nnode": "347--649",
            "nfeat": "1--3704",
            "graph_name": "ba,er",
            "feat_name": "randbin,sparsebin",
            "class_name": "multiclassrankfix",
            "split_name": "random",
            "model_name": "sage,gin,gat,gin_mlp",
            "model_nlayer": "8--14",
            "model_act": "tanh",
            "model_hs": "248--512",
            "model_dropout": "0.4--0.8",
            "model_bias": "True",
            "model_nn_nlayer": "2",
            "model_nn_dropout": 0.0,
            "model_nn_hs": "64--256",
            "gnn_jk": None,
            "er_p": "0.005839684956818703--0.03357398067325377",
            "ba_nedge": "4--10",
            "fake_avgdeg": 4,
            "sparsebin_p": "0.01--0.02",
            "neig": 0,
            "node_deg": True,
            "pfn_emsize": 512,
            "pfn_lr": 1e-04,
            "pfn_nhead": 4,
            "pfn_nhid": 1024,
            "pfn_num_layers": 12,
            "pfn_epochs": 4000,
            "pfn_steps_per_epoch": 800,
            "batch_size": 1,
            "benchmarks": ["chameleon", "citeseer", "cora"],
            "plot": False,
            "attention": "adjacency",
            "verbose": 1,
            "seed": 3,
            "aggregate_k_gradients": 32,
            "nsplit": 1,
            "supervised": False,
        }
    )

    return args


def train_pfn_on_prior():
    # define default arguments
    defaults = defaults_settings()

    # optionally override default settings by passing in command line arguments
    args = read_args(defaults)

    # define the prior, which we can use to sample graphs from
    prior = Prior(args)

    # optionally sample some graphs from the prior and plot statistics
    # sampling can be done by calling prior.sample_graphs()

    # fit PFN on samples from the prior
    # This step corresponds to points 7) - 8) in the documentation at the top of this
    # file
    training_result = train(
        priordataloader_class_or_get_batch=prior.sample_batch,
        seq_len=prior.nnode.max_val,
        batch_size=args.batch_size,
        extra_prior_kwargs_dict={
            "num_features": prior.nfeat.max_val
            + args.neig
            + (1 if args.node_deg else 0),
            "neig": args.neig,
            "node_deg": args.node_deg,
            "batch_callback": None,
            "progress_bar": False,
            "attention": args.attention,
            "attention_nhead": args.pfn_nhead,
        },
        full_attention=(args.attention == "full"),
        train_mixed_precision=True,
        epochs=args.pfn_epochs,
        steps_per_epoch=args.pfn_steps_per_epoch,
        emsize=args.pfn_emsize,
        nhead=args.pfn_nhead,
        nhid=args.pfn_nhid,
        nlayers=args.pfn_num_layers,
        lr=args.pfn_lr,
        verbose=args.verbose,
        criterion=nn.CrossEntropyLoss(reduction="none", weight=torch.ones(args.nclass)),
        y_encoder_generator=encoders.Linear,
        encoder_generator=encoders.Linear,
        num_global_att_tokens=None,
        warmup_epochs=1,
        epoch_callback=epoch_callback,
    )

    # get the trained model
    model = training_result.model

    # save the trained model on disc
    torch.save(model, output_folder / "model.pth")
    print(f"Successfully trained PFN on the prior, and saved it under {output_folder}.")


def evaluate_pfn_on_benchmarks(benchmarks: list[str]):
    pass


if __name__ == "__main__":
    # define some global variables
    output_folder = Path("prior_fitted_model")

    logs_folder = output_folder / "logs"
    log_file_path = logs_folder / "out.txt"

    # create ouptut folder
    logs_folder.mkdir(parents=True, exist_ok=True)
    log_file_path.write_text("")  # Clears the file before appending

    # because pfns library uses print statements, we create a class to duplicate output
    # to both the terminal (default stdout) and a log file
    class Duo:
        def __init__(self, file, stream):
            self.file = file
            self.stream = stream  # terminal output (stdout)

        def write(self, message):
            self.stream.write(message)  # print to terminal
            self.file.write(message)  # write to file

        def flush(self):
            if not self.file.closed:
                self.file.flush()
            self.stream.flush()

        def close(self):
            if not self.file.closed:
                self.file.close()

    # open log file in append mode and redirect stdout
    log_file = open(log_file_path, "a", encoding="utf-8")
    sys.stdout = Duo(log_file, sys.__stdout__)

    # train and evaluate a PFN
    try:
        train_pfn_on_prior()

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

    finally:
        sys.stdout.close()
