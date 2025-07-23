import matplotlib.pyplot as plt
import torch

from data_loader import binarize_tensor, get_binarized_mnist_dataloader
from vae_model import VAE


def evaluate_vae(
    model_path: str = "vae_model_binarized_optimized.pth",
    latent_channels: int = 2,
    binarized_data_path: str = "binarized_mnist.pt",
) -> None:
    """
    Evaluates the VAE model by visualizing reconstructed images.

    Args:
        model_path (str): Path to the trained VAE model.
        latent_channels (int): Number of latent channels in the VAE.
        binarized_data_path (str): Path to the binarized data.
    """
    vae = VAE(latent_channels=latent_channels)
    vae.load_state_dict(torch.load(model_path))
    vae.cuda().eval()

    dataloader = get_binarized_mnist_dataloader(
        binarized_data_path, batch_size=20, shuffle=True
    )

    with torch.no_grad():
        x, _ = next(iter(dataloader))
        x = x.cuda()

        recon_x, _, _ = vae(x)
        recon_x_binarized = binarize_tensor(recon_x, threshold=0.5)

        plt.figure(figsize=(15, 6))

        for i in range(20):
            plt.subplot(4, 10, i + 1)
            plt.imshow(x[i].cpu().squeeze(), cmap="gray")
            plt.title(f"Original {i+1}")
            plt.axis("off")

            plt.subplot(4, 10, i + 21)
            plt.imshow(recon_x_binarized[i].cpu().squeeze(), cmap="gray")
            plt.title(f"Recon {i+1}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    evaluate_vae()
