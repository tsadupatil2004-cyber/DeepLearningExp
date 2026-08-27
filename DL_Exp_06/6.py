import tensorflow as tf
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Use a smaller dataset for faster execution
x_train = x_train[:10000]
y_train = y_train[:10000]

x_test = x_test[:2000]
y_test = y_test[:2000]

# Convert labels to 1D
y_train = y_train.flatten()
y_test = y_test.flatten()

# Resize images to MobileNetV2 input size
x_train = tf.image.resize(x_train, (96, 96))
x_test = tf.image.resize(x_test, (96, 96))

# Convert pixel values to range 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0


# Load pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False


# Create Transfer Learning model
model = tf.keras.Sequential([
    base_model,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])


# Display model architecture
model.summary()


# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Train the model
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.1,
    batch_size=32
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


# Display predictions
plt.figure(figsize=(10, 5))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(x_test[i])

    predicted_class = predictions[i].argmax()

    plt.title(
        f"Actual: {y_test[i]}\n"
        f"Predicted: {predicted_class}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()


# Plot accuracy
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

plt.title("Transfer Learning - Training Accuracy")

plt.legend()
plt.grid(True)

plt.show()
