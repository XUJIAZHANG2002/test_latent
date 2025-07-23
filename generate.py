import matplotlib.pyplot as plt
import torch

from diffusion_utils import T, binarize_tensor, reverse_diffusion
from unet_model import DeepUNet
from vae_model import VAE


def generate_image(
    unet_path="deep_unet_model_v6.pth",
    vae_path="vae_model.pth",
    latent_channels=2,
    in_channels=2,
):
    """
    Generates an image from random noise using the trained U-Net and VAE.

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

    with torch.no_grad():
        z_t = torch.randn(1, in_channels, 7, 7, device="cuda")
        print(f"Initial z_t: mean {z_t.mean().item():.4f}, std {z_t.std().item():.4f}")

        z_0 = reverse_diffusion(z_t, unet)
        print(f"Final z_0: mean {z_0.mean().item():.4f}, std {z_0.std().item():.4f}")

        x_generated = vae.decode(z_0)
        x_generated_binarized = binarize_tensor(x_generated, threshold=0.5)

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(x_generated[0].cpu().squeeze(), cmap="gray")
        plt.title("Generated (Continuous)")
        plt.subplot(1, 2, 2)
        plt.imshow(x_generated_binarized[0].cpu().squeeze(), cmap="gray")
        plt.title("Generated (Binarized)")
        plt.show()


if __name__ == "__main__":
    generate_image()
