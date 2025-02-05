"""This file contains utility functions used in multiple scripts."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


class Output:
    output_folder = Path("prior_fitted_model")
    logs_folder = output_folder / "logs"

    train_log_file_path = logs_folder / "pfn_train.log"
    model_path = output_folder / "model.pth"
    pfn_result_path = output_folder / "pfn_eval.json"
    baseline_result_path = output_folder / "baseline_eval.json"


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

    def __init__(self, value: str, name: str, sep: str = "--") -> None:
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


def setup_logging(log_file_name: str):
    """Set up logging that outputs to console and to a log file simultaneously."""
    # create ouptut folder
    Output.logs_folder.mkdir(parents=True, exist_ok=True)
    # clear log file before appending to it
    Output.train_log_file_path.write_text("")

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
    log_file = open(Output.train_log_file_path, "a", encoding="utf-8")
    sys.stdout = Duo(log_file, sys.__stdout__)
