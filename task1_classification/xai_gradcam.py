import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# 1. Define the reshape transform for ViT
# This function converts (Batch, Sequence_Length, Hidden_Size) -> (Batch, Hidden_Size, H, W)
def reshape_transform(tensor, height=14, width=14):
    # For ViT tiny/base with patch 16 and image size 224, the grid is 14x14
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the second dimension
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# 2. Select the target layer
# We target the last norm layer in the last block
target_layers = [model.blocks[-1].norm1]

# 3. Initialize Grad-CAM with the reshape_transform
cam = GradCAM(model=model,
              target_layers=target_layers,
              reshape_transform=reshape_transform)

# 4. Prepare a sample from the test set
image_tensor, label = test_dataset[10] # You can change the index
input_tensor = image_tensor.unsqueeze(0).to(device)

# 5. Generate Heatmap
targets = [ClassifierOutputTarget(1)] # Focus on 'Pneumonia' class
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]

# 6. Visualization
# Prepare the background image (RGB [0, 1])
rgb_img = image_tensor.permute(1, 2, 0).cpu().numpy()
rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())

# Combine heatmap and image
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# 7. Plotting
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(rgb_img.squeeze(), cmap='gray')
plt.title(f"Original X-ray (Label: {label[0]})")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(visualization)
plt.title("ViT Attention (Grad-CAM)")
plt.axis('off')

plt.show()
