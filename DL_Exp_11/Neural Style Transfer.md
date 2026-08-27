# Neural Style Transfer

## AIM
To study and implement Neural Style Transfer (NST) using a pre-trained Convolutional Neural Network to generate a new image that combines the content of one image with the artistic style of another.

## THEORY

### Introduction
**Neural Style Transfer (NST)** is a deep learning technique that uses Convolutional Neural Networks (CNNs) to blend two images — a **content image** and a **style image** — to produce a new, generated image that preserves the structural content of the content image while adopting the artistic style (colors, textures, brush strokes) of the style image. This technique was introduced by Gatys et al. in their paper "A Neural Algorithm of Artistic Style," and it demonstrated that CNNs trained for image classification (such as VGG networks) implicitly learn representations that can separate and recombine content and style from images.

NST works by leveraging the fact that different layers of a CNN capture different types of information: shallower layers capture low-level features like edges and colors, while deeper layers capture high-level content and object structure.

### Key Concepts

1. **Content Image**
   The image whose overall structure, objects, and spatial layout are to be preserved in the final output.

2. **Style Image**
   The image whose artistic style — such as color patterns, textures, and brush strokes (e.g., a famous painting) — is to be transferred onto the content image.

3. **Generated (Target) Image**
   The output image, initialized either as a copy of the content image or as random noise, which is iteratively updated to match the content of the content image and the style of the style image.

### Working Principle
NST uses a **pre-trained CNN** (commonly **VGG19**, trained on ImageNet) purely as a fixed feature extractor — the network's weights are not updated during the process. Instead, the pixel values of the generated image itself are treated as trainable parameters and are iteratively optimized.

The core idea is to define two separate loss functions — a **content loss** and a **style loss** — and combine them into a total loss that is minimized using gradient descent to update the generated image.

#### 1. Content Loss
Content loss measures how different the content of the generated image is from the content image. It is computed by comparing the feature map activations at a chosen deeper convolutional layer of the CNN for both the content image and the generated image.
```
L_content(C, G) = (1/2) * Σ ( F_ij^C - F_ij^G )²
```
Where:
- `F^C` = feature map of the content image at the chosen layer
- `F^G` = feature map of the generated image at the same layer

Deeper layers are used for content loss because they capture high-level structural and object information while discarding fine-grained pixel details.

#### 2. Style Loss
Style loss measures how different the style (textures, colors, patterns) of the generated image is from the style image. Rather than comparing raw feature maps, style is captured using the **Gram Matrix**, which represents the correlations between different feature channels at a given layer, capturing texture information while discarding spatial arrangement.

**Gram Matrix:**
```
G_ij^l = Σ_k F_ik^l * F_jk^l
```
Where `F^l` is the feature map at layer `l`, and `i, j` index different filter channels.

**Style Loss at a single layer:**
```
E_l = (1 / (4 * N_l² * M_l²)) * Σ ( G_ij^S - G_ij^G )²
```

**Total Style Loss** (summed over multiple layers, each with a weight `w_l`):
```
L_style(S, G) = Σ w_l * E_l
```
Style loss is typically computed across multiple layers (both shallow and deep) to capture style information at different scales — from fine textures to larger patterns.

#### 3. Total Loss
The total loss is a weighted combination of the content loss and style loss:
```
L_total = α * L_content + β * L_style
```
Where `α` and `β` are weighting hyperparameters that control the relative importance of content preservation versus style transfer. Increasing `β` relative to `α` results in a more stylized output, while increasing `α` preserves more of the original content structure.

### Optimization Process
1. Initialize the generated image (often as a copy of the content image, or random noise).
2. Pass the content image, style image, and generated image through the pre-trained CNN (e.g., VGG19) to extract feature maps at selected layers.
3. Compute the content loss and style loss using the extracted feature maps.
4. Compute the total loss as a weighted sum of content and style losses.
5. Use gradient descent (or an optimizer like L-BFGS or Adam) to update the **pixel values of the generated image** so as to minimize the total loss — note that the CNN's weights remain frozen throughout.
6. Repeat this process for a number of iterations until the generated image sufficiently blends the content of the content image with the style of the style image.

### Choice of Layers
- **Content Representation**: Usually extracted from a single deeper convolutional layer (e.g., `conv4_2` in VGG19), which captures high-level content while ignoring fine pixel-level detail.
- **Style Representation**: Usually extracted from multiple layers across the network (e.g., `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`), to capture style information at multiple scales.

### Applications of Neural Style Transfer
- Digital art and creative content generation
- Photo and video stylization applications/filters
- Design and advertising (generating stylized visuals)
- Assisting artists in exploring different artistic styles

### Advantages
- Produces visually compelling artistic results by combining content and style from two different images.
- Leverages pre-trained CNNs without requiring task-specific training data.
- Highly flexible — style and content weighting can be adjusted to control the output.

### Limitations
- Computationally expensive, as it requires iterative optimization for each new image pair.
- Sensitive to the choice of layers and loss weights (α, β), which require tuning for good results.
- Slower compared to feed-forward style transfer methods that use a separately trained generator network for real-time stylization.

## CONCLUSION
Neural Style Transfer was successfully studied and implemented using a pre-trained CNN. It was observed that by separately capturing and optimizing content representation (from deeper layers) and style representation (via Gram matrices across multiple layers), it is possible to generate a new image that preserves the structural content of one image while adopting the artistic style of another. This demonstrates the powerful ability of CNNs to disentangle and recombine content and style information, making Neural Style Transfer an effective technique for creative and artistic image generation applications.

---

### Screenshot

<img width="614" height="301" alt="image" src="https://github.com/user-attachments/assets/9fe56711-7d1b-41e6-b61f-383d10a2f8d5" />
