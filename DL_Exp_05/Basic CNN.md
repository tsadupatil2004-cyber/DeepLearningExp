# Basic CNN

## AIM
To study and implement a basic Convolutional Neural Network (CNN) for image classification and to understand its architecture, working principle, and performance evaluation.

## THEORY

### Introduction
A **Convolutional Neural Network (CNN)** is a class of deep learning algorithms specifically designed to process data with a grid-like topology, such as images. Unlike traditional fully connected neural networks, CNNs exploit the spatial structure of images by using local connections and shared weights, which drastically reduces the number of parameters while preserving important spatial and hierarchical patterns.

CNNs are widely used in computer vision tasks such as image classification, object detection, facial recognition, and medical image analysis due to their ability to automatically learn spatial hierarchies of features from raw pixel data.

---

### Architecture of a Basic CNN
A typical CNN consists of the following layers:

1. **Input Layer**
   Accepts the raw image data (e.g., an image of size 28x28x1 for grayscale or 32x32x3 for RGB).

2. **Convolutional Layer**
   This layer applies a set of learnable filters (kernels) that slide over the input image to produce feature maps. Each filter detects specific features such as edges, textures, or patterns.
   - **Kernel/Filter**: A small matrix (e.g., 3x3, 5x5) that performs element-wise multiplication and summation (convolution operation) with the input.
   - **Stride**: The number of pixels the filter moves at each step.
   - **Padding**: Adding extra pixels around the input to control the spatial size of the output (Valid or Same padding).

3. **Activation Function (ReLU)**
   The Rectified Linear Unit (ReLU) introduces non-linearity into the network by converting all negative values to zero, allowing the network to learn complex patterns.
   ```
   f(x) = max(0, x)
   ```

4. **Pooling Layer (Subsampling)**
   Reduces the spatial dimensions (width and height) of the feature maps, decreasing computation and controlling overfitting.
   - **Max Pooling**: Selects the maximum value from each patch of the feature map.
   - **Average Pooling**: Computes the average value from each patch.

5. **Flattening Layer**
   Converts the 2D feature maps into a 1D vector to be fed into the fully connected layers.

6. **Fully Connected (Dense) Layer**
   Every neuron is connected to every neuron in the previous layer. This layer performs the final classification based on the extracted features.

7. **Output Layer**
   Produces the final prediction using an activation function such as **Softmax** (for multi-class classification) or **Sigmoid** (for binary classification).

---

### Working Principle
1. The input image is passed through convolutional layers where filters extract low-level features (edges, corners) in the initial layers and high-level features (shapes, objects) in the deeper layers.
2. Activation functions introduce non-linearity so the network can learn complex relationships.
3. Pooling layers reduce dimensionality while retaining important information, making the network computationally efficient and less prone to overfitting.
4. The flattened feature vector is passed to fully connected layers, which combine the extracted features to make a final decision.
5. The output layer generates class probabilities, and the class with the highest probability is selected as the prediction.
6. During training, the network uses **backpropagation** and an optimization algorithm (e.g., Adam, SGD) to minimize the **loss function** (e.g., Cross-Entropy Loss) by updating the weights of filters and neurons.

---

### Advantages of CNN
- Automatic feature extraction without manual intervention.
- Parameter sharing reduces the number of trainable parameters.
- Translation invariance due to pooling operations.
- High accuracy in image-related tasks compared to traditional machine learning methods.

---

## CONCLUSION
The basic Convolutional Neural Network was successfully studied and implemented. It was observed that CNNs are highly effective for image classification tasks because they automatically learn hierarchical spatial features through convolutional and pooling operations, reducing the need for manual feature engineering. The model demonstrated good performance in classifying images by leveraging convolution, activation, pooling, and fully connected layers, confirming CNN's suitability for computer vision applications.

---

### screenshot

<img width="1411" height="746" alt="Screenshot 2026-08-15 084926" src="https://github.com/user-attachments/assets/cbc88475-245b-4779-a93d-84bff7a92ef7" />
