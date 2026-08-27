import numpy as np
import matplotlib.pyplot as plt


# AND Gate Dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])


# Initialize weights and bias
weights = np.zeros(2)
bias = 0

learning_rate = 0.1
epochs = 10


# Step Activation Function
def activation(z):
    if z >= 0:
        return 1
    else:
        return 0


# Train the Perceptron
for epoch in range(epochs):

    for i in range(len(X)):

        # Calculate weighted sum
        z = np.dot(X[i], weights) + bias

        # Predict output
        prediction = activation(z)

        # Calculate error
        error = y[i] - prediction

        # Update weights
        weights = weights + learning_rate * error * X[i]

        # Update bias
        bias = bias + learning_rate * error


# Display trained parameters
print("Final Weights:", weights)
print("Final Bias:", bias)


# Test the Perceptron
print("\nPredictions:")

for i in range(len(X)):

    z = np.dot(X[i], weights) + bias

    prediction = activation(z)

    print(
        "Input:", X[i],
        "Actual:", y[i],
        "Predicted:", prediction
    )


# Plot the data points
plt.figure(figsize=(7, 5))

for i in range(len(X)):

    if y[i] == 0:
        plt.scatter(
            X[i][0],
            X[i][1],
            marker="o",
            s=100,
            label="Class 0" if i == 0 else ""
        )
    else:
        plt.scatter(
            X[i][0],
            X[i][1],
            marker="x",
            s=100,
            label="Class 1"
        )


# Plot Decision Boundary
x_values = np.linspace(-0.2, 1.2, 100)

if weights[1] != 0:

    y_values = -(weights[0] * x_values + bias) / weights[1]

    plt.plot(
        x_values,
        y_values,
        label="Decision Boundary"
    )


# Graph labels
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Simple Perceptron - AND Gate")

plt.legend()
plt.grid(True)

plt.show()
