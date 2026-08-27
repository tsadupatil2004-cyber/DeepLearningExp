import tensorflow as tf
import matplotlib.pyplot as plt


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0


# Add channel dimension
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# Create CNN model
model = tf.keras.Sequential([
    
    # Convolution Layer
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    # Pooling Layer
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Second Convolution Layer
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    # Pooling Layer
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Flatten
    tf.keras.layers.Flatten(),

    # Fully Connected Layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Output Layer
    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])


# Display model structure
model.summary()


# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Train CNN
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.1
)


# Evaluate model
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\nTest Accuracy:", test_accuracy)


# Make predictions
predictions = model.predict(x_test)


# Display sample predictions
plt.figure(figsize=(10, 5))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(
        x_test[i].reshape(28, 28),
        cmap="gray"
    )

    predicted_class = predictions[i].argmax()

    plt.title(
        f"Actual: {y_test[i]}\nPredicted: {predicted_class}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()


# Plot training accuracy
plt.figure(figsize=(7, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training and Validation Accuracy")

plt.legend()
plt.grid(True)

plt.show()
