import os

import torch
import torch.optim as optim

from data_loader import get_binarized_mnist_dataloader, prepare_mnist_data
from vae_model import VAE, vae_loss


def train_vae(
    epochs: int = 25,
    lr: float = 1e-3,
    batch_size: int = 1024,
    latent_channels: int = 2,
    model_path: str = "vae_model_binarized_optimized.pth",
    binarized_data_path: str = "binarized_mnist.pt",
) -> None:
    """
    Trains the VAE model.

    Args:
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        latent_channels (int): Number of latent channels in the VAE.
        model_path (str): Path to save the trained model.
        binarized_data_path (str): Path to the binarized data.
    """
    if not os.path.exists(binarized_data_path):
        prepare_mnist_data(binarized_data_path, visualize=False)

    dataloader = get_binarized_mnist_dataloader(binarized_data_path, batch_size)

    vae = VAE(latent_channels=latent_channels).cuda()
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.cuda()
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(x)
            loss = vae_loss(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss {loss.item():.4f}")

    torch.save(vae.state_dict(), model_path)
    print(f"Optimized VAE trained and saved to '{model_path}'!")


if __name__ == "__main__":
    train_vae()
