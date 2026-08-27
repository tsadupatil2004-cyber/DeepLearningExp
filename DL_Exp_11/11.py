import tensorflow as tf
import matplotlib.pyplot as plt
from google.colab import files

# Upload content and style images
uploaded = files.upload()

# Get uploaded file names
file_names = list(uploaded.keys())

print("Uploaded files:", file_names)

# First image = Content Image
# Second image = Style Image
content_path = file_names[0]
style_path = file_names[1]


# Load images
content_image = tf.keras.utils.load_img(
    content_path,
    target_size=(224, 224)
)

style_image = tf.keras.utils.load_img(
    style_path,
    target_size=(224, 224)
)


# Convert images to arrays
content_image = tf.keras.utils.img_to_array(content_image)
style_image = tf.keras.utils.img_to_array(style_image)


# Add batch dimension
content_image = tf.expand_dims(content_image, axis=0)
style_image = tf.expand_dims(style_image, axis=0)


# Display images
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(content_image[0] / 255.0)
plt.title("Content Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(style_image[0] / 255.0)
plt.title("Style Image")
plt.axis("off")

plt.show()


# Load pretrained VGG19
vgg = tf.keras.applications.VGG19(
    include_top=False,
    weights="imagenet"
)

vgg.trainable = False


# Content and style layers
content_layer = "block5_conv2"

style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1"
]


# Create feature extraction model
outputs = [
    vgg.get_layer(content_layer).output
]

outputs += [
    vgg.get_layer(layer).output
    for layer in style_layers
]

feature_model = tf.keras.Model(
    vgg.input,
    outputs
)


# Preprocess images for VGG19
content_processed = tf.keras.applications.vgg19.preprocess_input(
    content_image
)

style_processed = tf.keras.applications.vgg19.preprocess_input(
    style_image
)


# Extract features
content_features = feature_model(content_processed)
style_features = feature_model(style_processed)


print("Content features extracted successfully!")
print("Style features extracted successfully!")
print("Neural Style Transfer setup completed!")
