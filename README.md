# AI Garbage Classification

This repository contains the source code and documentation for the "Garbage Classification" project, developed within the scope of the **Artificial Intelligence** elective course at **DHBW Stuttgart**.

The objective of this project is to implement and evaluate a Neural Network capable of classifying waste into different recycling categories.
We use a [pre-labeld dataset](https://www.kaggle.com/datasets/hassnainzaidi/garbage-classification) by Hassnain Zaidi featuring more than 2,500 images.

## Authors
* [Anton Engels](https://github.com/engelsanton)
* [Louis Muhler](https://github.com/Louis-Muhler)

## Project Overview
The system utilizes **PyTorch** to construct and train a deep learning model. The pipeline includes data preprocessing, model definition, training loops, and evaluation metrics. The codebase is designed to be hardware-agnostic, allowing for development on local CPUs and high-performance training on CUDA-enabled GPUs without code modification.

## Requirements

* **Python 3.10 or later**
* **PyTorch** (CPU or CUDA version)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Louis-Muhler/AI-Garbage-Classification.git
cd ai-garbage-classification
```

### 2. Dependency Installation
Install the necessary Python packages. The installation command for PyTorch varies depending on the hardware configuration.

**Option A: Standard Installation (Laptop / CPU only)**
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn

```

**Option B: GPU Installation (Desktop with NVIDIA GPU)**
Ensure CUDA drivers are installed. Use the command corresponding to your CUDA version (e.g., CUDA 11.8):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118

```

*Note: Consult the [official PyTorch website](https://pytorch.org/get-started/locally/) for the latest CUDA-compatible wheels.*

## Usage
The application automatically detects available hardware accelerators. It prioritizes `cuda` (NVIDIA), falls back to `mps` (Apple Silicon), and uses `cpu` if no accelerator is found.

**Training**
To execute the training script:
```bash
python 

```

**Data Exploration**
To run the data analysis:

```bash
python

```

## Project Structure

```text
ai-garbage-classification/
├── data/               # Dataset directory
├── src/                # Source code
│   ├── data_loader.py  # Data ingestion and transformation
│   ├── model.py        # Neural Network architecture
│   └── train.py        # Training and validation routines
├── models/             # Serialized model checkpoints (.pth)
├── README.md           # Project documentation
└── requirements.txt    # List of dependencies
```

## Disclaimer
This project was developed for academic purposes at the Baden-Wuerttemberg Cooperative State University (DHBW).
