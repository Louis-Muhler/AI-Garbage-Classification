# AI Garbage Classification

This repository contains the source code and documentation for the "Garbage Classification" project, developed within the scope of the **Artificial Intelligence** elective course at **DHBW Stuttgart**.

The objective of this project is to implement and evaluate a Neural Network capable of classifying waste into different recycling categories.
We composed a diverse dataset by aggregating multiple sources to ensure robustness and variety:

1. **Garcia Coding (2024)**: Waste Classification Dataset. Available at [Kaggle](https://www.kaggle.com/datasets/garciacoding/waste-classification)
2. **Challa, A. (2024)**: Waste Classification Dataset. Available at [Kaggle](https://www.kaggle.com/datasets/adithyachalla/waste-classification)
3. **King, A. (2024)**: Recyclable and Household Waste Classification. Available at [Kaggle](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification)
4. **Arkadiy Hacks (2024)**: Drinking Waste Classification. Available at [Kaggle](https://www.kaggle.com/datasets/arkadiyhacks/drinking-waste-classification)

## Authors

- [Anton Engels](https://github.com/engelsanton)
- [Louis Muhler](https://github.com/Louis-Muhler)

## Project Overview

The system utilizes **PyTorch** to construct and train a deep learning model. The pipeline includes data preprocessing, model definition, training loops, and evaluation metrics. The codebase is designed to be hardware-agnostic, allowing for development on local CPUs and high-performance training on CUDA-enabled GPUs without code modification.

## Requirements

- **Python 3.10 or later**
- **PyTorch** (CPU or CUDA version)

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
Ensure CUDA drivers are installed. Use the command corresponding to your CUDA version (e.g., CUDA 12.6):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

```

_Note: Consult the [official PyTorch website](https://pytorch.org/get-started/locally/) for the latest CUDA-compatible wheels._

## Usage

The application is controlled via a central CLI entry point `src/main.py`. It automatically detects available hardware accelerators (CUDA/MPS/CPU).

**Training (Default: Custom ResNet)**

```bash
python src/main.py --model custom_resnet
```

**Transfer Learning (e.g., MobileNetV3)**

```bash
python src/main.py --model mobilenet_v3_large
```

**Benchmark Suite**
To train and compare all available models (Logistic, SimpleCNN, CustomResNet, ResNet18, ResNet50, EfficientNetB0, MobileNetV3):

```bash
python src/run_benchmark.py
```

## Supported Models

- **custom_resnet**: Optimized custom architecture for this dataset (97% Acc).
- **simple_cnn**: Lightweight baseline CNN.
- **mobilenet_v3_large**: Efficient pre-trained model (good for mobile).
- **resnet18 / resnet50**: Standard industrial baselines.
- **efficientnet_b0**: Modern efficient architecture.
- **logistic**: Simple baseline.

## Project Structure

```text
AI-Garbage-Classification/
├── data_split/          # Processed dataset (train/val/test)
├── models/              # Checkpoints, history.json, and reports
│   ├── custom_resnet/
│   └── ...
├── src/                 # Source code
│   ├── main.py          # CLI Entry point & Argument Parsing
│   ├── model.py         # Model definitions & Model Factory
│   ├── train.py         # Training loop & Early Stopping logic
│   ├── utils.py         # Data loading, transforms & visualization
│   └── run_benchmark.py # Automated benchmark suite
├── README.md            # Project documentation
└── requirements.txt     # List of dependencies
```

## Disclaimer

This project was developed for academic purposes at the Baden-Wuerttemberg Cooperative State University (DHBW).
