# Google Colab Introduction & GPU/TPU Setup

## AIM

To study the **Google Colab environment** and understand how to use **GPU and TPU accelerators** for executing Python, Machine Learning, and Deep Learning programs efficiently.

---

## THEORY

### 1. Introduction to Google Colab

**Google Colaboratory (Google Colab)** is a cloud-based platform provided by Google that allows users to write and execute Python programs through a web browser. It is based on the **Jupyter Notebook** environment and is widely used for Data Science, Machine Learning, and Deep Learning experiments.

Google Colab eliminates the need for installing Python and many commonly used libraries locally. It provides a ready-to-use environment with libraries such as **NumPy, Pandas, Matplotlib, Scikit-learn, TensorFlow, and PyTorch**.

---

### 2. Colab Runtime

A **runtime** is the virtual computing environment in which the code written in a Colab notebook is executed.

The runtime can provide different hardware resources:

- **CPU** – General-purpose processing.
- **GPU** – Parallel processing and faster Deep Learning computations.
- **TPU** – Specialized hardware designed for Machine Learning workloads.

---

### 3. GPU in Google Colab

A **GPU (Graphics Processing Unit)** is a hardware accelerator capable of performing many calculations simultaneously. GPUs are particularly useful for computationally intensive operations such as neural network training, image processing, and matrix calculations.

GPU can be enabled in Google Colab through:

**Runtime → Change runtime type → Hardware accelerator → GPU**

GPU availability can be checked using Python:

```python
import torch

print("GPU Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

```
---

### 4. TPU in Google Colab

A **TPU (Tensor Processing Unit)** is a specialized processor developed by Google for accelerating Machine Learning and Deep Learning computations.

TPUs are particularly optimized for tensor operations and are commonly used with frameworks such as **TensorFlow and JAX**.

TPU can be selected through:

**Runtime → Change runtime type → Hardware accelerator → TPU**

TPU availability can be checked using TensorFlow:

```python
import tensorflow as tf

print("TensorFlow Version:", tf.__version__)

print("TPU Devices:", tf.config.list_logical_devices("TPU"))

```
### 6. Advantages of Google Colab

- Provides a cloud-based Python environment.
- No complex local setup is required.
- Provides access to GPU and TPU resources when available.
- Supports popular Machine Learning libraries
- Allows notebooks to be stored and shared easily.
- Useful for Data Science, Machine Learning, and Deep Learning experiments.

---

### CONCLUSION

Google Colab was studied as a cloud-based Python and Jupyter Notebook environment. The concepts of CPU, GPU, and TPU were understood, and the methods for enabling and verifying GPU/TPU hardware acceleration in Colab were studied.
Google Colab provides a convenient platform for executing computationally intensive Machine Learning and Deep Learning experiments without requiring high-end local hardware.

---

### screenshot

<img width="1915" height="832" alt="image" src="https://github.com/user-attachments/assets/4a6303da-2581-4ff7-953d-589ac0ccbf13" />

