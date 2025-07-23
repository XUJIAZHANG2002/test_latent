import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Enhanced Variational Autoencoder (VAE) model with a deeper architecture.
    """

    def __init__(self, latent_channels: int = 2) -> None:
        """
        Initializes the VAE model.

        Args:
            latent_channels (int): Number of latent channels in the VAE.
        """
        super(VAE, self).__init__()
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Conv2d(8, latent_channels, kernel_size=1)
        self.fc_logvar = nn.Conv2d(8, latent_channels, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 8, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input data into the latent space.

        Args:
            x: The input data.
            """

        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterizes the latent space.

        Args:
            mu: The mean from the latent space.
            logvar: The log variance from the latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent space into the original data.

        Args:
            z: The latent space.

        Returns:
            The decoded data.
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Forward pass of the VAE.

        Args:
            x: The input data.

        Returns:
            The decoded data.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.2,
) -> torch.Tensor:
    """
    Calculates the VAE loss, which consists of reconstruction loss and KL divergence.
    
    Args:
        recon_x: The reconstructed data.
        x: The original data.
        mu: The mean from the latent space.
        logvar: The log variance from the latent space.
        beta: A weighting factor for the KL divergence.

    Returns:
        The total VAE loss.
    """
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div
