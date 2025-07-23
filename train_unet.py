import torch
import torch.nn.functional as F
import torch.optim as optim

from diffusion_utils import T, forward_diffusion
from unet_model import DeepUNet


def train_unet(
    epochs=160,
    lr=1e-3,
    batch_size=2048,
    model_path="deep_unet_model_v6.pth",
    latent_data_path="latent_mnist_binarized.pt",
    in_channels=2,
):
    """
    Trains the U-Net model for the diffusion process.

    Args:
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Training batch size.
        model_path (str): Path to save the trained model.
        latent_data_path (str): Path to the latent space dataset.
        in_channels (int): Number of input channels for the U-Net.
    """
    latent_dataset = torch.load(latent_data_path)

    unet = DeepUNet(in_channels=in_channels, T=T).cuda()
    optimizer = optim.Adam(unet.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(epochs):
        for i in range(0, len(latent_dataset), batch_size):
            z_0 = latent_dataset[i : i + batch_size].cuda()
            t = torch.randint(0, T, (z_0.size(0),), device="cuda")
            z_t, noise = forward_diffusion(z_0, t)

            optimizer.zero_grad()
            pred_noise = unet(z_t, t)
            loss = F.mse_loss(pred_noise, noise)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()

            if i % 100 == 0:
                print(
                    f"Epoch {epoch}, Step {i}, Loss {loss.item():.4f}, LR {scheduler.get_last_lr()[0]:.6f}"
                )

        scheduler.step()

    torch.save(unet.state_dict(), model_path)
    print(f"Deep U-Net trained and saved to '{model_path}'!")


if __name__ == "__main__":
    train_unet()
