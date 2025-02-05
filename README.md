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
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```
### 3. Install Dependencies

Ensure you have the latest version of pip, setuptools, and wheel:
```sh
pip install --upgrade pip setuptools wheel
```
Then, install the required dependencies:
```sh
pip install -r requirements.txt
```

### 4. Verify Installation

To ensure everything is set up correctly, run:
```sh
pytest tests
```
If all tests pass, the installation was successful.
📂 Project Structure
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
python scripts/train_pfn.py
```
### Evaluate a Model

To start evaluating a trained PFN model:
```sh
python scripts/evaluate_pfn.py
```
### Train and evaluate baselines

To train and evaluate other models on benchmarks:
```sh
python scripts/train_eval_baselines.py
```
### Run Tests

To run all tests using pytest:
```sh
pytest tests
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
