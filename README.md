# Neural Networks

PyTorch implementations of core deep learning architectures — autoencoders, RNNs, VAEs, and model calibration techniques.

## Projects

| Project | Focus | Key Techniques |
|---------|-------|----------------|
| **1. Autoencoders** | Dimensionality reduction & denoising | 3/5-layer architectures, regularization (dropout, Lasso, early stopping), Gaussian noise injection on MNIST & Fashion-MNIST |
| **2. Model Calibration** | Uncertainty quantification | Reliability diagrams, ECE metrics, temperature scaling (Platt's), fine-tuning pre-trained models |
| **3. RNNs & Attention** | Sentiment analysis | Vanilla RNN vs attention mechanisms for sequence modeling |
| **4. VAEs** | Generative modeling | Variational Autoencoders for face generation on CelebA dataset |
| **Kaggle** | Competition | SVM-based classification |

## Tech Stack

- **Framework**: PyTorch
- **Tools**: Jupyter, scikit-learn, Weights & Biases
- **Environment**: Nix flake for reproducibility

## Setup

```bash
nix develop  # Reproducible dev environment
```

## Repository

Each project directory contains Jupyter notebooks with implementations and trained model checkpoints (`.pth` files).