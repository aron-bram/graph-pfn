# Graph-PFN

Graph-PFN is a Python project for graph-based PFNs (Prior-Fitted Networks).  
This repository provides scripts for training and evaluating PFNs and other models such as Graph Neural Networks (GNNs) using `torch`, `torch_geometric`, and other dependencies.

---

## 🚀 Installation Guide

### **1. Clone the Repository**
First, clone the repository to your local machine:
```sh
git clone git@github.com:aron-bram/graph-pfn.git
cd graph-pfn
```
### 2. Set Up a Virtual Environment

It is recommended to use a virtual environment to manage dependencies:
```sh
python3.10 -m venv venv  # NOTE that I only tested the installation with python version 3.10
source venv/bin/activate
```
### 3. Install Dependencies

Ensure you have the latest version of pip, setuptools, and wheel:
```sh
pip install --upgrade pip setuptools wheel
```
When running on a CPU, install the following dependencies:
```sh
pip install --upgrade pip setuptools wheel
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric==2.4.0
pip install torch_scatter==2.1.2 torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-2.1.1+cpu.html
pip install git+https://github.com/automl/PFNs
pip install matplotlib
```
Alternatively, when running on GPU with cuda 11.8 support, run these commands instead:
```sh
pip install --upgrade pip setuptools wheel
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric==2.4.0
pip install torch_scatter==2.1.2 torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-2.1.1+cu118.html
pip install git+https://github.com/automl/PFNs
pip install matplotlib
```
Optionally install dev tools:
```sh
pip install pytest ruff black isort
```

### 4. Verify Installation

To ensure everything is set up correctly, run:
```sh
cd scripts
python train_pfn.py --pfn_epochs 2 --pfn_steps_per_epoch 1
```
You should see the following output at the end of the run:
```sh
Successfully trained PFN on 2 samples from the prior, and saved it under prior_fitted_model
```
Using the command above, this output means that you successfully trained a pfn on 2 datasets, which is saved

## 📂 Project Structure:
```sh
graph-pfn/
│── scripts/            # Source code
│── tests/              # Test scripts
│── README.md           # Installation guide (this file)
│── requirements.txt    # Alternative dependency file
│── venv/               # (Optional) Virtual environment
```
## 🛠 Usage
### Train a Model

To start training a PFN model:
```sh
cd scripts
python scripts/train_pfn.py
```
### Evaluate a Model

To start evaluating a trained PFN model:
```sh
cd scripts
python evaluate_pfn.py
```
### Train and evaluate baselines

To train and evaluate other models on benchmarks:
```sh
cd scripts
python train_eval_baselines.py
```

### Understanding scripts
Refer to the respective script's documentation at the top of the .py file for more detail on requirements, output, and explanation of what each script does. For example, a comprehensive explanation on how the PFN is trained on the prior can be found in train_pfn.py.

Each script accepts its own arguments to customize the run, and each argument is documented in the code.

### Run Tests

To run all tests using pytest:
```sh
cd tests
pytest test_sampler.py
```
## ⚙️ Troubleshooting

If torch_sparse or torch_scatter fails to install, ensure you're using the correct index:

pip install torch_scatter==2.1.2 torch-sparse==0.6.16 -f https://data.pyg.org/whl/torch-2.1.1+cpu.html

If imports fail, check that src/graph_pfn/ contains an __init__.py file.
Ensure you're using Python 3.10+.

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome!
Please fork the repository and submit a pull request with your improvements.

For any issues, open an issue on GitHub.
