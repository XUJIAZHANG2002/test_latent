import torch
from torch.nn.functional import conv3d, conv2d
import numpy as np

def power_of_2(n: int) -> bool:
    """Checks if an integer is a power of 2."""
    return (n & (n - 1) == 0) and n != 0

def reduction(original_mask: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """
    Performs sum pooling on a mask using convolution. This effectively counts
    the number of masked pixels in each `scale x scale` window.
    """
    if not isinstance(scale, int) or not power_of_2(scale) or scale == 0:
        raise ValueError("Scale must be a positive integer power of two")

    dims = len(original_mask.shape)
    # Add batch and channel dimensions for convolution: [H, W] -> [1, 1, H, W]
    mask_unsqueezed = original_mask.unsqueeze(0).unsqueeze(0).float()

    if dims == 2:
        # Create a 1x1 kernel of ones to sum values in a sliding window
        kernel = torch.ones((1, 1, scale, scale), dtype=mask_unsqueezed.dtype, device=mask_unsqueezed.device)
        # Convolve with a stride equal to the scale to get non-overlapping sums
        reduced = conv2d(mask_unsqueezed, kernel, stride=scale, padding=0)
    elif dims == 3:
        kernel = torch.ones((1, 1, scale, scale, scale), dtype=mask_unsqueezed.dtype, device=mask_unsqueezed.device)
        reduced = conv3d(mask_unsqueezed, kernel, stride=scale, padding=0)
    else:
        raise ValueError("Dims must be 2 or 3")

    # Remove batch and channel dimensions before returning
    return reduced.squeeze(0).squeeze(0)

def exists_mask(mask: torch.Tensor, scale: int = 4, dims: int = 3) -> torch.Tensor:
    """Mask exists if at least one pixel was masked in the original window."""
    return (mask > 0).to(torch.float32)

def forall_mask(mask: torch.Tensor, scale: int = 4, dims: int = 3) -> torch.Tensor:
    """Mask exists only if all pixels were masked in the original window."""
    return (mask == scale ** dims).to(torch.float32)

def rounded_mask(mask: torch.Tensor, scale: int = 4, dims: int = 3) -> torch.Tensor:
    """Mask exists if more than 50% of pixels were masked."""
    return (mask / (scale ** dims) > 0.5).to(torch.float32)

def weighted_mask(mask: torch.Tensor, scale: int = 4, dims: int = 3) -> torch.Tensor:
    """The new mask value is the proportion of masked pixels."""
    return (mask / (scale ** dims)).to(torch.float32)


if __name__ == "__main__":
    # --- 1. Configuration ---
    import matplotlib.pyplot as plt
    H, W = 256, 256 # Original dimensions
    depth = 5
    scale = 2 ** int(np.log2(H) - depth)
    # scale =16       # Downsampling scale factor
    dims = 2        # Working in 2D

    # --- 2. Create Random Data ---
    # Create a random noise image (values from 0 to 1)
    original_image = torch.ones((3, H, W)) * 0.5
    # Create a random binary mask (0s and 1s) where ~30% of pixels are 1
    # original_mask = (torch.rand((H, W)) > 0.7).to(torch.float32)
    original_mask = torch.zeros((H, W)) # For demonstration, use a constant mask
    original_mask[10:30, 10:30] = 1  # Create a small area of zeros for visualization
    original_mask[100:140, 100:140] = 1  # Another area of zeros

    # --- 3. Process Image and Masks ---
    # Downsample the image using average pooling for reference
    downsampled_image = torch.nn.functional.avg_pool2d(
        original_image.unsqueeze(0), kernel_size=scale, stride=scale
    ).squeeze(0).numpy().transpose(1, 2, 0)  # Convert to HWC format for plotting

    # Apply the sum reduction to the original mask once
    reduced_sum_mask = reduction(original_mask, scale=scale)

    # Generate the final downsampled mask for each technique
    results = {
        "exists": exists_mask(reduced_sum_mask, scale=scale, dims=dims),
        "forall": forall_mask(reduced_sum_mask, scale=scale, dims=dims),
        "rounded": rounded_mask(reduced_sum_mask, scale=scale, dims=dims),
        "weighted": weighted_mask(reduced_sum_mask, scale=scale, dims=dims),
    }

    # --- 4. Visualization ---
    fig, axes = plt.subplots(3, 2, figsize=(8, 11), dpi=120)
    fig.suptitle(f"Masking Techniques Demonstration (Scale={scale})", fontsize=16)
    
    
    plot_image = original_image.clone().numpy().transpose(1, 2, 0)  # Convert to HWC format for plotting
    # Plot Original Image
    axes[0, 0].imshow(plot_image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    masked_image = plot_image.copy()
    masked_image[original_mask == 0] = 1
    # Plot Original Mask
    axes[0, 1].imshow(masked_image, vmin=0, vmax=1)
    axes[0, 1].set_title("Original Mask")
    axes[0, 1].axis('off')

    # Plot results for each technique
    plot_config = {
        "exists": axes[1, 0], "forall": axes[1, 1],
        "rounded": axes[2, 0], "weighted": axes[2, 1],
    }

    for mode, ax in plot_config.items():
        downsampled_mask = results[mode]
        ax.imshow(downsampled_image) # Base image

        # Create a colored overlay (viridis) where alpha is determined by the mask
        overlay = plt.get_cmap('viridis')(downsampled_mask.numpy())
        overlay[..., 3] = np.clip(downsampled_mask.numpy(), 0, 1) # Use mask value for alpha
        
        ax.imshow(overlay)
        ax.set_title(f'"{mode.capitalize()}" Mask')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()