## Deep Learning Experiments 
# Deep Learning Practicum Guide

A practical repository containing 12 complete experiments using TensorFlow and Keras, ranging from beginner setup to intermediate deep learning topics such as Computer Vision, Sequence Modeling, and Generative AI[cite: 1].

---

## 👤 Author & Academic Details

* **Author:** Tanvi Sambhaji Sadupatil
* **Department:** Data Science Engineering
* **Roll no.:** 52
* **University:** Shivaji University, Kolhapur

---

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites & Setup](#prerequisites--setup)
- [Experiments List](#experiments-list)
  - [Phase 1: Foundations & Setup](#phase-1-foundations--setup)
  - [Phase 2: Basic Neural Networks](#phase-2-basic-neural-networks)
  - [Phase 3: Convolutional Neural Networks (CNN)](#phase-3-convolutional-neural-networks-cnn)
  - [Phase 4: Recurrent Neural Networks (RNN & Sequence Modeling)](#phase-4-recurrent-neural-networks-rnn--sequence-modeling)
  - [Phase 5: Advanced & Generative Models](#phase-5-advanced--generative-models)
- [Best Practices](#best-practices)
- [Recommended Resources](#recommended-resources)

---

## ℹ️ Overview

This guide provides step-by-step implementations and code for 12 deep learning experiments designed to build core concepts in neural network design, training, optimization, and evaluation[cite: 1].

**Key Topics Covered:**
* Environment setup (Anaconda, Google Colab GPU/TPU)[cite: 1]
* Dense Networks (Perceptrons, Multilayer Perceptrons)[cite: 1]
* Computer Vision (CNNs, Transfer Learning with VGG16)[cite: 1]
* Sequential Data & Time Series (RNN, LSTM, GRU, Text Generation)[cite: 1]
* Generative Deep Learning (Neural Style Transfer, VAE/GAN)[cite: 1]

---

## 🛠️ Prerequisites & Setup

### Environment Requirements
* Python 3.8 – 3.10[cite: 1]
* TensorFlow & Keras[cite: 1]
* NumPy, Pandas, Matplotlib, Scikit-Learn[cite: 1]
* Google Colab (Optional, for GPU acceleration)[cite: 1]

### Local Installation (Anaconda / Miniconda)
```bash
# Create virtual environment
conda create -n dl_env python=3.10
conda activate dl_env

# Install dependencies
pip install tensorflow tensorflow-gpu numpy pandas matplotlib scikit-learn jupyter notebook
```[cite: 1]

---

## 🧪 Experiments List

### Phase 1: Foundations & Setup

#### Experiment 1: Anaconda/Miniconda Setup with TensorFlow & Keras
* **Objective:** Configure local virtual environment and verify TensorFlow/Keras installation[cite: 1].
* **Key Focus:** GPU device detection and environment setup[cite: 1].

#### Experiment 2: Google Colab Introduction & GPU/TPU Setup
* **Objective:** Configure cloud computing environments and mount Google Drive[cite: 1].
* **Key Focus:** Enabling GPU runtime acceleration and execution benchmarking[cite: 1].

---

### Phase 2: Basic Neural Networks

#### Experiment 3: Simple Perceptron (Single Layer Neural Network)
* **Objective:** Implement a single neuron for binary classification[cite: 1].
* **Dataset:** Iris (2 classes, 2 features)[cite: 1]
* **Architecture:** 1 Dense layer (Sigmoid activation)[cite: 1]
* **Outputs:** Decision boundary visualization, binary crossentropy loss curve[cite: 1].

#### Experiment 4: Multilayer Neural Network
* **Objective:** Build a deep feedforward network with multiple hidden layers and regularization[cite: 1].
* **Dataset:** Iris (3 classes, 4 features)[cite: 1]
* **Architecture:** Input (4) → Dense (64, ReLU) → Dropout (0.2) → Dense (32, ReLU) → Dropout (0.2) → Dense (16, ReLU) → Output (3, Softmax)[cite: 1]

---

### Phase 3: Convolutional Neural Networks (CNN)

#### Experiment 5: Basic CNN
* **Objective:** Build a CNN from scratch for image classification[cite: 1].
* **Dataset:** MNIST (28x28 grayscale digits)[cite: 1]
* **Architecture:** Conv2D (32) → MaxPool → Conv2D (64) → MaxPool → Conv2D (64) → Dense (64) → Dense (10, Softmax)[cite: 1]

#### Experiment 6: Transfer Learning in CNN
* **Objective:** Compare pre-trained feature extraction against training from scratch[cite: 1].
* **Dataset:** CIFAR-10[cite: 1]
* **Architecture:** Base VGG16 (ImageNet weights, frozen) + GlobalAveragePooling2D + Dense layers[cite: 1]

---

### Phase 4: Recurrent Neural Networks (RNN & Sequence Modeling)

#### Experiment 7: Simple RNN
* **Objective:** Implement sequential processing for time-series prediction[cite: 1].
* **Dataset:** Synthetic sine wave time series[cite: 1]
* **Architecture:** SimpleRNN (64, return sequences) → SimpleRNN (32) → Dropout → Dense (16) → Dense (1)[cite: 1]

#### Experiment 8: RNN with LSTM
* **Objective:** Mitigate vanishing gradients using Long Short-Term Memory units on longer sequences[cite: 1].
* **Dataset:** Synthetic stock price time-series[cite: 1]
* **Architecture:** LSTM (100) → Dropout → LSTM (50) → Dropout → Dense (25) → Dense (1)[cite: 1]

#### Experiment 9: RNN with GRU
* **Objective:** Compare performance, parameters, and computational efficiency across RNN, LSTM, and GRU[cite: 1].
* **Architecture:** Dual-layer GRU network[cite: 1]

#### Experiment 10: Text Generation with LSTM
* **Objective:** Character-level language modeling to predict next characters in text sequences[cite: 1].
* **Dataset:** Shakespeare text corpus[cite: 1]
* **Architecture:** LSTM (128) → Dropout → LSTM (64) → Dropout → Dense (Softmax over vocab)[cite: 1]

---

### Phase 5: Advanced & Generative Models

#### Experiment 11: Neural Style Transfer
* **Objective:** Separate and recombine content and style attributes of images using Gram matrices[cite: 1].
* **Architecture:** Pre-trained VGG19 feature extraction (`block5_conv2` for content; multiple conv layers for style)[cite: 1]

#### Experiment 12: Image Generation Models (VAE/GAN)
* **Objective:** Build a Variational Autoencoder (VAE) using the reparameterization trick to sample new data[cite: 1].
* **Dataset:** MNIST[cite: 1]
* **Architecture:** Conv2D Encoder → 20D Latent Space ($\mu, \sigma$) → Conv2DTranspose Decoder[cite: 1]
---
## 📌 Summary Matrix

| Exp # | Experiment Title | Key Concept | Network Architecture | Dataset Used |
| :---: | :--- | :--- | :--- | :--- |
| **1** | Anaconda/Miniconda Setup | Setup & Environment | N/A | N/A |
| **2** | Google Colab Intro | Cloud Computing & GPU | N/A | N/A |
| **3** | Simple Perceptron | Single Neuron | 1 Dense layer | Iris (binary) |
| **4** | Multilayer Network | Multi-class Classification | 3 Dense layers + Dropout | Iris (multiclass) |
| **5** | Basic CNN | Convolution & Feature Maps | Conv2D + MaxPooling2D | MNIST |
| **6** | Transfer Learning | Feature Extraction | Pre-trained VGG16 + Custom Top | CIFAR-10 |
| **7** | Simple RNN | Sequence Modeling | Dual SimpleRNN layers | Synthetic Sine Wave |
| **8** | RNN with LSTM | Gated Sequences | Dual LSTM layers | Synthetic Stock Prices |
| **9** | RNN with GRU | Efficiency vs Performance | Dual GRU layers | Synthetic Stock Prices |
| **10** | Text Generation | Language Modeling | Dual LSTM layers | Shakespeare Text |
| **11** | Neural Style Transfer | Gram Matrices & Content/Style | VGG19 Network | Custom Images |
| **12** | Image Generation | Generative Modeling / Latent Space | VAE (Conv2D / Transpose) | MNIST |

[cite: 1]
---
## 💡 Best Practices

1. **Data Handling:** Always scale/normalize inputs and use proper dataset splits (70/15/15)[cite: 1].
2. **Model Training:** Integrate callbacks like `EarlyStopping` and `ReduceLROnPlateau` to avoid overfitting[cite: 1].
3. **Debugging:** Validate pipelines on small datasets before scaling up model complexity[cite: 1].
4. **Visualization:** Plot loss/accuracy histories over training epochs and display predictions[cite: 1].

---

## 📚 Recommended Resources

* **Books:** *Deep Learning with Python* by François Chollet; *Deep Learning* by Goodfellow, Bengio, Courville[cite: 1]
* **Courses:** Fast.ai Practical Deep Learning, Andrew Ng's Deep Learning Specialization[cite: 1]
* **Documentation:** [TensorFlow / Keras Official Docs](https://tensorflow.org/)[cite: 1]
