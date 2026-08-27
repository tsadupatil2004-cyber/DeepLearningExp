# Deep Learning Practicum Guide 🚀
*12 Complete Experiments with TensorFlow & Keras (Beginner to Intermediate Level)*

Welcome to the **Deep Learning Practicum** repository! This project contains clean, structured, and fully functional implementations of 12 foundational to advanced deep learning experiments built using **TensorFlow** and **Keras**.

---

## 👤 Author & Academic Details

* **Author:** Tanvi Sambhaji Sadupatil
* **Department:** Data Science Engineering
* **Roll no.:** 52
* **University:** Shivaji University, Kolhapur

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Prerequisites & Environment Setup](#-prerequisites--environment-setup)
- [Experiments Summary](#-experiments-summary)
- [Detailed List of Experiments](#-detailed-list-of-experiments)
  - [Exp 1: Anaconda/Miniconda Setup with TensorFlow & Keras](#experiment-1-anacondaminiconda-setup-with-tensorflow--keras)
  - [Exp 2: Google Colab Setup & GPU Acceleration](#experiment-2-google-colab-setup--gpu-acceleration)
  - [Exp 3: Simple Perceptron (Single-Layer Neural Network)](#experiment-3-simple-perceptron-single-layer-neural-network)
  - [Exp 4: Multilayer Neural Network (MLP)](#experiment-4-multilayer-neural-network-mlp)
  - [Exp 5: Basic Convolutional Neural Network (CNN)](#experiment-5-basic-convolutional-neural-network-cnn)
  - [Exp 6: Transfer Learning with Pre-trained Models](#experiment-6-transfer-learning-with-pre-trained-models)
  - [Exp 7: Recurrent Neural Network (Simple RNN)](#experiment-7-recurrent-neural-network-simple-rnn)
  - [Exp 8: Long Short-Term Memory Network (LSTM)](#experiment-8-long-short-term-memory-network-lstm)
  - [Exp 9: Gated Recurrent Unit (GRU)](#experiment-9-gated-recurrent-unit-gru)
  - [Exp 10: Character-Level Text Generation using LSTM](#experiment-10-character-level-text-generation-using-lstm)
  - [Exp 11: Neural Style Transfer](#experiment-11-neural-style-transfer)
  - [Exp 12: Generative Models (Variational Autoencoders - VAE)](#experiment-12-generative-models-variational-autoencoders---vae)
- [Best Practices & Workflow](#-best-practices--workflow)
- [Resources & References](#-resources--references)

---

## 📊 Overview

| Exp # | Experiment Name | Core Concept / Technique | Architecture / Model | Dataset |
| :---: | :--- | :--- | :--- | :--- |
| **01** | **Environment Setup** | Conda, Virtual Envs, Pip | N/A | Local / System |
| **02** | **Google Colab & GPU Setup** | Cloud GPU/TPU, Colab Integration | N/A | Google Drive / Cloud |
| **03** | **Simple Perceptron** | Binary Classification, Sigmoid | 1 Dense Layer | Iris (2 Classes, 2 Features) |
| **04** | **Multilayer Neural Network** | Multi-class Classification, Dropout | Dense + Dropout (64-32-16-3) | Iris (3 Classes, 4 Features) |
| **05** | **Basic CNN** | Image Classification, Convolutions | Conv2D + MaxPooling + Dense | MNIST Digits (28x28 Grayscale) |
| **06** | **Transfer Learning** | Feature Extraction & Fine-Tuning | VGG16 (Pre-trained) + Custom Top | CIFAR-10 / Custom Images |
| **07** | **Simple RNN** | Sequence Processing, Time Series | Stacked SimpleRNN + Dense | Synthetic Sine Wave |
| **08** | **RNN with LSTM** | Long-term Dependencies, Gates | Stacked LSTM + Dropout + Dense | Synthetic Stock Prices |
| **09** | **RNN with GRU** | Parameter Efficiency, Gated Units | Stacked GRU + Dropout + Dense | Synthetic Stock Prices |
| **10** | **Text Generation** | Character-level Language Modeling | Stacked LSTM + Softmax | Shakespeare Text Corpus |
| **11** | **Neural Style Transfer** | Gram Matrix, Feature Optimization | VGG19 (Deep Layer Extraction) | Content & Style Images |
| **12** | **Image Generation** | Generative Models, Latent Space | VAE (Conv2D + Conv2DTranspose) | MNIST Digits |

---

## 💻 Prerequisites & Environment Setup

### Option 1: Local Environment Setup (Experiment 1)
```bash
# 1. Create a conda environment
conda create -n dl_env python=3.10 -y
conda activate dl_env

# 2. Install TensorFlow and core libraries
pip install tensorflow numpy pandas matplotlib scikit-learn jupyter notebook pillow
```

### Option 2: Google Colab Setup (Experiment 2)
1. Open [Google Colab](https://colab.research.google.com).
2. Go to **Runtime** > **Change runtime type** > Select **GPU** (e.g., T4 GPU).
3. Verify GPU setup inside your notebook:
   ```python
   import tensorflow as tf
   print("GPU Available:", tf.config.list_physical_devices('GPU'))
   ```

---

## 🔬 Detailed List of Experiments

### Experiment 1: Anaconda/Miniconda Setup with TensorFlow & Keras
- **Objective:** Configure a isolated Python environment for deep learning on local hardware.
- **Key Concepts:** Virtual environments, package management (`pip`/`conda`), verification of CUDA/GPU bindings.

### Experiment 2: Google Colab Introduction & GPU/TPU Setup
- **Objective:** Harness cloud-based accelerators for faster deep learning training.
- **Key Concepts:** Cloud runtimes, Google Drive mounting (`google.colab.drive`), basic execution benchmark against CPU.

### Experiment 3: Simple Perceptron (Single-Layer Neural Network)
- **Objective:** Implement a single neuron model with Sigmoid activation for binary classification.
- **Key Concepts:** Decision boundary visualization, binary crossentropy, Stochastic Gradient Descent (SGD).

### Experiment 4: Multilayer Neural Network (MLP)
- **Objective:** Build a multi-layer deep network to solve multi-class classification tasks.
- **Key Concepts:** ReLU activation, One-Hot Encoding (`to_categorical`), Dropout regularization, Softmax output layer, Confusion Matrix analysis.

### Experiment 5: Basic Convolutional Neural Network (CNN)
- **Objective:** Build a CNN from scratch to classify handwritten digits from the MNIST dataset.
- **Key Concepts:** Convolutional filters (`Conv2D`), Spatial downsampling (`MaxPooling2D`), Feature map flattening, Dense layers.

### Experiment 6: Transfer Learning in CNN
- **Objective:** Leverage weights pre-trained on ImageNet to classify specialized target datasets.
- **Key Concepts:** Feature extraction using pre-trained `VGG16`, layer freezing (`trainable = False`), fine-tuning top classification layers vs. training from scratch.

### Experiment 7: Simple Recurrent Neural Network (RNN)
- **Objective:** Process sequential data and model temporal dynamics using simple recurrent layers.
- **Key Concepts:** Time-series sequence formatting `(samples, timesteps, features)`, recurrent states, sine wave prediction.

### Experiment 8: RNN with LSTM (Long Short-Term Memory)
- **Objective:** Solve vanishing/exploding gradient problems on longer temporal sequences using LSTM networks.
- **Key Concepts:** Forget gates, input gates, output gates, cell states, stock price trend modeling.

### Experiment 9: RNN with GRU (Gated Recurrent Unit)
- **Objective:** Build efficient gated sequence models and compare computational efficiency across RNN, LSTM, and GRU architectures.
- **Key Concepts:** Reset and update gates, parameter reduction, training runtime analysis.

### Experiment 10: Text Generation using LSTM
- **Objective:** Implement a character-level language model capable of predicting and generating new text.
- **Key Concepts:** Vocabulary building, sequence length sliding window, categorical crossentropy, text sampling.

### Experiment 11: Neural Style Transfer
- **Objective:** Synthesize an image preserving content from one image and style from another.
- **Key Concepts:** Feature representation in `VGG19`, Gram matrix calculation, content loss, style loss optimization.

### Experiment 12: Image Generation Models (Variational Autoencoders - VAE)
- **Objective:** Construct a generative model to synthesize brand new images from a continuous latent space.
- **Key Concepts:** Encoder-Decoder architecture, Reparameterization trick ($\mu, \sigma$), KL-Divergence loss, Reconstruction loss.

---

## 🛠️ Best Practices & Workflow

1. **Data Preprocessing:**
   - Always scale features (`StandardScaler`, `MinMaxScaler`, or division by `255.0` for images).
   - Maintain strict separation between Train, Validation, and Test datasets.
2. **Training & Regularization:**
   - Use `Dropout` and `EarlyStopping` callbacks to prevent overfitting.
   - Monitor both training and validation metrics for early detection of variance issues.
3. **Visualization:**
   - Plot loss curves (`loss` vs. `val_loss`) and accuracy curves across all training epochs.
   - Inspect model performance using confusion matrices and sample predictions.

---

## 📚 Recommended Resources

- **Books:**
  - *Deep Learning with Python* by François Chollet
  - *Deep Learning* by Goodfellow, Bengio, Courville
- **Documentation:**
  - [TensorFlow & Keras Official Guide](https://tensorflow.org/)
  - [PyTorch Documentation](https://pytorch.org/)

---
*Created as part of the Deep Learning Practicum Series | July 2026*
