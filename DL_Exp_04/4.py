import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier


# XOR Dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])


# Create Multilayer Neural Network
model = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation="relu",
    solver="adam",
    learning_rate_init=0.1,
    max_iter=5000,
    random_state=42
)


# Train the model
model.fit(X, y)


# Make predictions
predictions = model.predict(X)


print("Predictions:")
for i in range(len(X)):
    print(
        "Input:", X[i],
        "Actual:", y[i],
        "Predicted:", predictions[i]
    )


# Calculate accuracy
accuracy = model.score(X, y)

print("\nAccuracy:", accuracy)


# Create graph
plt.figure(figsize=(7, 5))

for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(
            X[i][0],
            X[i][1],
            marker="o",
            s=150,
            label="Class 0" if i == 0 else ""
        )
    else:
        plt.scatter(
            X[i][0],
            X[i][1],
            marker="x",
            s=150,
            label="Class 1" if i == 1 else ""
        )


plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Multilayer Neural Network - XOR Problem")
plt.legend()
plt.grid(True)

plt.show()
