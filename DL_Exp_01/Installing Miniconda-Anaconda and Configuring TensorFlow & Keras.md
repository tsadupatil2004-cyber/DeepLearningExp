# EXPERIMENT NO. 1
## Title : Installing Miniconda and Configuring TensorFlow & Keras 
---

---

## 🎯 Aim

To install and configure Miniconda on Windows, create a Python virtual environment, install TensorFlow and Keras, and verify the installation by running a basic TensorFlow program.

---

## 🎯 Objectives

- ✔ Install and configure Miniconda on Windows.
- ✔ Create a separate Python virtual environment for Deep Learning.
- ✔ Install TensorFlow, Keras, and supporting Python libraries.
- ✔ Verify the installation using a simple TensorFlow program.

---

## 📋 Prerequisites

- Windows 10/11 Operating System
- Stable Internet Connection
- Minimum 5 GB Free Disk Space
- Administrator Privileges (Recommended)

---

## 💻 Software Requirements

- Miniconda (Windows)
- Python 3.10
- TensorFlow
- Keras
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## 📖 Theory

Miniconda is a lightweight Python distribution that includes the Conda package manager. It allows users to install Python packages, create isolated environments, and manage dependencies efficiently.

A **Virtual Environment** provides an isolated workspace where project-specific libraries can be installed without affecting other Python projects on the system.

**TensorFlow** is an open-source Deep Learning framework developed by Google for designing, training, and deploying Machine Learning models.

**Keras** is a high-level Deep Learning API integrated with TensorFlow that simplifies the process of building and training neural networks.

---

# ⚙ Procedure / Implementation Steps

## Step 1: Download and Install Miniconda

1. Visit the official Miniconda website:

   https://docs.conda.io/en/latest/miniconda.html

2. Download the **Miniconda Windows 64-bit Installer**.

3. Double-click the downloaded `.exe` file.

4. Follow the installation wizard.

5. Select **Just Me** (Recommended).

6. Choose the installation directory.

7. Complete the installation.

8. Open **Anaconda Prompt** or **Command Prompt** after installation.

---

## Step 2: Verify Miniconda Installation

Open **Anaconda Prompt** and execute:

```bash
conda --version
```

**Expected Output**

```text
conda 25.x.x
```

---

## Step 3: Create a Virtual Environment

Create a new environment named **dl_env**.

```bash
conda create -n dl_env python=3.10
```

Type **y** when prompted.

Activate the environment.

```bash
conda activate dl_env
```

After activation, the prompt should appear similar to:

```text
(dl_env) C:\Users\YourName>
```

---

## Step 4: Install TensorFlow

Install TensorFlow (Keras is included).

```bash
pip install tensorflow
```

Verify TensorFlow installation.

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## Step 5: Install Required Libraries

```bash
pip install numpy pandas matplotlib scikit-learn notebook
```

---

## Step 6: Launch Jupyter Notebook (Optional)

```bash
jupyter notebook
```

A browser window will open where you can create a new Python notebook.

---

## Step 7: Verification Program

Create a Python file named **verify_tf.py** or run the following code inside Jupyter Notebook.

```python
import tensorflow as tf
import keras
import numpy as np

print("TensorFlow Version:", tf.__version__)
print("Keras Version:", keras.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Matrix Multiplication Test
matrix = tf.constant([[1., 2.],
                      [3., 4.]])

print("\nMatrix Multiplication Result:")
print(tf.matmul(matrix, matrix))
```

Run the program using:

```bash
python verify_tf.py
```

---

# 🖥 Output

```text
TensorFlow Version: 2.x.x
Keras Version: 3.x.x
GPU Available: []

Matrix Multiplication Result:

tf.Tensor(
[[ 7. 10.]
 [15. 22.]], shape=(2, 2), dtype=float32)
```

> **Note:** If your Windows system has a compatible NVIDIA GPU with CUDA installed, TensorFlow will display the available GPU(s) instead of an empty list (`[]`).

---

# ✅ Expected Outcomes

- ✔ Miniconda installed successfully on Windows.
- ✔ Virtual environment **dl_env** created successfully.
- ✔ TensorFlow and Keras installed successfully.
- ✔ NumPy, Pandas, Matplotlib, Scikit-learn, and Jupyter Notebook installed.
- ✔ TensorFlow verification program executed without errors.
- ✔ GPU detected if compatible hardware is available.

---

# ⚠ Common Issues and Solutions

| Issue | Solution |
|--------|----------|
| ❌ `conda` is not recognized | Restart Command Prompt or reinstall Miniconda and enable PATH settings. |
| ❌ Python version mismatch | Create the environment using Python 3.10. |
| ❌ Permission denied | Run Anaconda Prompt as Administrator. |
| ❌ `ModuleNotFoundError` | Install the missing package using `pip install <package_name>`. |
| ❌ TensorFlow installation failed | Upgrade pip using `python -m pip install --upgrade pip` and reinstall TensorFlow. |

---

# 🌟 Applications

- Deep Learning
- Machine Learning
- Artificial Intelligence
- Computer Vision
- Natural Language Processing (NLP)
- Image Classification
- Neural Network Development

---

# 🎓 Result

Miniconda was successfully installed on Windows, and a dedicated Python virtual environment was created. TensorFlow, Keras, and all required libraries were installed successfully. The verification program executed correctly, confirming that the Deep Learning environment is properly configured for developing, training, and testing TensorFlow and Keras applications.

---

# 📝 Conclusion

This experiment demonstrated the successful setup of a complete Deep Learning environment on Windows using Miniconda. The isolated Python environment ensured proper dependency management, while TensorFlow and Keras provided the necessary tools for developing neural network models. The successful execution of the verification program confirmed that the system is ready for Machine Learning, Deep Learning, and Artificial Intelligence development.
