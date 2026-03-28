# CIFAR-10 ML Training Pipeline

This project implements a complete Machine Learning training pipeline for image classification using the CIFAR-10 dataset and a Convolutional Neural Network (CNN).

## Features
- **Modular Architecture**: Separate modules for data, model, and pipeline.
- **Config Management**: All hyperparameters are stored in `configs/config.yaml`.
- **GPU Acceleration**: Fully compatible with Google Colab (CUDA).
- **Artifact Management**: Automatically saves logs, plots, and the best model to the `outputs/` folder.
- **Best Model Checkpointing**: Saves the model state with the highest F1-score.

## Project Structure
- `main.py`: Entry point of the application.
- `src/`: Source code (dataset, model, pipeline).
- `configs/`: YAML configuration files.
- `outputs/`: Training logs, loss plots, and saved models.

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/sophiakoroliova/mlops_lab1.git](https://github.com/sophiakoroliova/mlops_lab1.git)
   cd mlops_lab1