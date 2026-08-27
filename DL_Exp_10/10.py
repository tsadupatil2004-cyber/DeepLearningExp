import tensorflow as tf
import numpy as np

# Small text dataset
text = """
machine learning is a branch of artificial intelligence.
deep learning uses neural networks to learn patterns.
lstm is useful for sequence and text generation.
neural networks learn from data.
"""

# Convert text to lowercase
text = text.lower()

# Create character vocabulary
chars = sorted(set(text))

char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}

# Convert text into numbers
encoded_text = np.array(
    [char_to_index[char] for char in text]
)

# Sequence length
seq_length = 20

X = []
y = []

# Create training sequences
for i in range(len(encoded_text) - seq_length):
    X.append(encoded_text[i:i + seq_length])
    y.append(encoded_text[i + seq_length])

X = np.array(X)
y = np.array(y)

# Create LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=len(chars),
        output_dim=32
    ),

    tf.keras.layers.LSTM(64),

    tf.keras.layers.Dense(
        len(chars),
        activation="softmax"
    )
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy"
)

# Display model
model.summary()

# Train model
model.fit(
    X,
    y,
    epochs=30,
    batch_size=16,
    verbose=1
)


# Text generation function
def generate_text(seed_text, length=100):

    seed_text = seed_text.lower()

    generated_text = seed_text

    for _ in range(length):

        # Convert seed text to numbers
        input_text = generated_text[-seq_length:]

        input_sequence = np.array([
            char_to_index.get(char, 0)
            for char in input_text
        ])

        input_sequence = input_sequence.reshape(
            1, -1
        )

        # Predict next character
        prediction = model.predict(
            input_sequence,
            verbose=0
        )

        next_index = np.argmax(prediction[0])

        next_char = index_to_char[next_index]

        generated_text += next_char

    return generated_text


# Generate text
seed = "machine learning"

result = generate_text(
    seed,
    length=100
)

print("\nGenerated Text:")
print(result)
