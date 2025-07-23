import matplotlib.pyplot as plt
import torch

from data_loader import get_binarized_mnist_dataloader
from diffusion_utils import ALPHAS_CUMPROD, BETAS, T, binarize_tensor, forward_diffusion
from unet_model import DeepUNet
from vae_model import VAE


def latent_inpaint(z_0, unet, mask, steps=T - 1):
    """
    Performs inpainting in the latent space using a blended diffusion process (RePaint).

    Args:
        z_0 (torch.Tensor): The original latent tensor.
        unet (nn.Module): The U-Net model.
        mask (torch.Tensor): The mask indicating the region to inpaint (1 for inpainting, 0 for known).
        steps (int): The number of diffusion steps.

    Returns:
        torch.Tensor: The inpainted latent tensor.
    """
    z = torch.randn_like(z_0)

    for i in range(steps, 0, -1):
        t = torch.full((z.size(0),), i - 1, device=z.device, dtype=torch.long)

        # 1. Denoise one step unconditionally to get a candidate z_{t-1}
        pred_noise = unet(z, t)

        alphas_cumprod_t = ALPHAS_CUMPROD[t].view(-1, 1, 1, 1)
        alphas_cumprod_t_prev = (
            ALPHAS_CUMPROD[t - 1].view(-1, 1, 1, 1)
            if i > 1
            else torch.ones_like(alphas_cumprod_t)
        )
        betas_t = BETAS[t].view(-1, 1, 1, 1)

        # Using the same reverse step logic as in DDPM for consistency
        # First, predict the mean of the distribution for x_{t-1}
        model_mean = (1 / torch.sqrt(1.0 - betas_t)) * (
            z - (betas_t / torch.sqrt(1 - alphas_cumprod_t)) * pred_noise
        )

        # Then, add noise if not the last step
        z_uncond = model_mean
        if i > 1:
            variance = (
                betas_t * (1.0 - alphas_cumprod_t_prev) / (1.0 - alphas_cumprod_t)
            )
            z_uncond += torch.sqrt(variance) * torch.randn_like(z)

        # 2. Get the forward-diffused original image (for the known parts)
        z_known_t, _ = forward_diffusion(z_0, t)

        # 3. Combine them using the mask
        z = z_known_t * (1 - mask) + z_uncond * mask

        if i % 100 == 0:
            print(
                f"Inpainting Step {i}: z mean {z.mean().item():.4f}, std {z.std().item():.4f}"
            )

    return z


def inpaint_image(
    unet_path: str = "deep_unet_model_v6.pth",
    vae_path: str = "vae_model.pth",
    mask: torch.Tensor = None,
    latent_channels: int = 2,
    in_channels: int = 2,
) -> None:
    """
    Performs inpainting on an image from the dataset.

    Args:
        unet_path (str): Path to the trained U-Net model.
        vae_path (str): Path to the trained VAE model.
        latent_channels (int): Number of latent channels in the VAE.
        in_channels (int): Number of input channels for the U-Net.
    """
    unet = DeepUNet(in_channels=in_channels, T=T).cuda()
    vae = VAE(latent_channels=latent_channels).cuda()

    unet.load_state_dict(torch.load(unet_path))
    vae.load_state_dict(torch.load(vae_path))

    unet.eval()
    vae.eval()

    dataloader = get_binarized_mnist_dataloader(batch_size=1, shuffle=True)

    with torch.no_grad():
        x, _ = next(iter(dataloader))
        x = x.cuda()

        # Encode the image and create a mask for the bottom half
        z_0, _ = vae.encode(x)
        if mask is None:
            mask = torch.zeros_like(z_0)
            mask[:, :, z_0.size(2) // 2 :, :] = 1

        z_inpainted = latent_inpaint(z_0, unet, mask)

        x_reconstructed = vae.decode(z_0)
        x_inpainted = vae.decode(z_inpainted)

        x_reconstructed_bin = binarize_tensor(x_reconstructed)
        x_inpainted_bin = binarize_tensor(x_inpainted)

        # Create a masked version of the original for visualization
        x_masked = x.clone()
        x_masked_binned = binarize_tensor(x_masked)
        # We need to approximate the mask in the image space for visualization
        # A simple way is to downsample the image-space mask
        im_mask = torch.nn.functional.interpolate(mask, size=x.shape[2:])
        x_masked_binned[im_mask > 0.5] = 0.5  # Gray out the masked area

        plt.figure(figsize=(16, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(x[0].cpu().squeeze(), cmap="gray")
        plt.title("Original")

        plt.subplot(1, 4, 2)
        plt.imshow(x_masked_binned[0].cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
        plt.title("Masked")

        plt.subplot(1, 4, 3)
        plt.imshow(x_reconstructed_bin[0].cpu().squeeze(), cmap="gray")
        plt.title("Reconstructed")

        plt.subplot(1, 4, 4)
        plt.imshow(x_inpainted_bin[0].cpu().squeeze(), cmap="gray")
        plt.title("Inpainted")
        plt.show()


if __name__ == "__main__":
    inpaint_image()
