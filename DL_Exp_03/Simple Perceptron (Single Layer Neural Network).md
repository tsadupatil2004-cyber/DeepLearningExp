# Simple Perceptron (Single Layer Neural Network)

## AIM

To study and understand the working of a **Simple Perceptron (Single Layer Neural Network)** and its application in solving binary classification problems.

---

## THEORY

### 1. Introduction to Perceptron

A **Perceptron** is the simplest form of an artificial neural network used for **binary classification**. It was introduced by **Frank Rosenblatt**.

A perceptron takes one or more input values, applies weights to those inputs, adds a bias, and passes the result through an activation function to produce an output.

The basic structure of a perceptron is:

```text
Input Values
     ↓
Weights
     ↓
Weighted Sum + Bias
     ↓
Activation Function
     ↓
Output
```

---

### 2. Components of a Perceptron

A perceptron consists of the following components:

- Input: Input values are the features provided to the perceptron.
- Weights: Weights determine the importance of each input.
- Bias: Bias is an additional value added to the weighted sum. It helps the perceptron shift the decision boundary.
- Weighted Sum: The weighted sum is calculated as:

```text
Z = W₁X₁ + W₂X₂ + b

where:

X₁, X₂ = input values
W₁, W₂ = weights
b = bias
Z = weighted sum

Activation Function: The activation function converts the weighted sum into the final output.
```

---

### 3. Step Activation Function

A basic perceptron commonly uses a Step Activation Function.

```text
If Z >= 0 → Output = 1
If Z < 0  → Output = 0
```
Mathematically:

```text
f(Z) = 1, if Z >= 0
f(Z) = 0, if Z < 0
```

Therefore, the output of the perceptron is either 0 or 1.

---

## 4. Perceptron Learning Rule

The perceptron learns by comparing its predicted output with the actual output.

The error is calculated as:
```text
Error = Actual Output - Predicted Output
```

The weights are updated using:
```text
Wᵢ = Wᵢ + η × Error × Xᵢ
```
The bias is updated using:
```text
b = b + η × Error
```
```text
where:

Wᵢ = weight
η = learning rate
Error = difference between actual and predicted output
Xᵢ = input value
b = bias
```

This process is repeated for multiple training examples and epochs until the perceptron learns the correct classification.

---

### 5. Learning Rate

The learning rate determines how much the weights and bias change during each update.

For example:

Learning Rate = 0.1

A smaller learning rate produces smaller updates, while a larger learning rate produces larger updates.

---

### 6. Binary Classification

A perceptron is mainly used for binary classification, where data is divided into two classes.

Class 0 → Negative Class
Class 1 → Positive Class

The perceptron learns a decision boundary that separates the two classes.

---

### 7. Decision Boundary

The perceptron creates a linear decision boundary between different classes.

For two input features, the decision boundary is:

```text
W₁X₁ + W₂X₂ + b = 0
```
Points on one side of the boundary are classified as one class, while points on the other side are classified as another class.


---

### 8. Linearly Separable Data

A dataset is called linearly separable when a straight line can separate its different classes.
A single-layer perceptron can successfully solve problems where the classes are linearly separable.

---

### 9. Limitations of Perceptron

The main limitation of a single-layer perceptron is that it can only solve linearly separable problems.
It cannot solve non-linearly separable problems such as the XOR problem using a single layer.
For complex problems, multi-layer neural networks are required.

---

### 10. Applications of Perceptron

- Binary classification
- Pattern recognition
- Logic gate implementation
- Simple decision-making systems
- Basic Machine Learning experiments

---

### CONCLUSION

The Simple Perceptron is a fundamental building block of neural networks. It performs binary classification by calculating a weighted sum of inputs, adding a bias, and applying an activation function. The perceptron learns by updating its weights and bias based on the classification error. It can successfully solve linearly separable problems, making it an important concept for understanding Artificial Neural Networks and Deep Learning.

---

### screenshot

<img width="1918" height="881" alt="image" src="https://github.com/user-attachments/assets/eb3f3bea-3bf7-4f66-b8d9-1965b48e92ab" />

---

<img width="1910" height="911" alt="image" src="https://github.com/user-attachments/assets/0bfa4b22-d71d-4964-9725-4412f25cca00" />

---

<img width="1799" height="790" alt="image" src="https://github.com/user-attachments/assets/972f04cf-c4b2-4757-9d58-3d7ab3c26aa9" />
