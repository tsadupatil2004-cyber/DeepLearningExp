# Install required libraries
!pip install -q diffusers transformers accelerate

# Import libraries
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt


# Load Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)


# Text prompt
prompt = "A beautiful futuristic city at sunset, digital art"


# Generate image
image = pipe(prompt).images[0]


# Display image
plt.figure(figsize=(8, 8))

plt.imshow(image)
plt.title("Generated Image")

plt.axis("off")
plt.show()


# Save image
image.save("generated_image.png")

print("Image generated successfully!")
print("Saved as generated_image.png")
