import torch

# Diffusion parameters
T = 500
DEVICE = "cuda"
BETAS = torch.linspace(1e-4, 0.01, T, device=DEVICE)
ALPHAS = 1.0 - BETAS
ALPHAS_CUMPROD = torch.cumprod(ALPHAS, dim=0)


def forward_diffusion(z_0, t):
    """
    Applies forward diffusion to a tensor.

    Args:
        z_0: The initial tensor.
        t: The timestep.

    Returns:
        A tuple containing the diffused tensor and the noise that was added.
    """
    noise = torch.randn_like(z_0)
    alpha_t = ALPHAS_CUMPROD[t].view(-1, 1, 1, 1)
    z_t = torch.sqrt(alpha_t) * z_0 + torch.sqrt(1 - alpha_t) * noise
    return z_t, noise


def reverse_diffusion(z_t, unet, steps=T - 1):
    """
    Applies reverse diffusion to a tensor to generate an image.

    Args:
        z_t: The diffused tensor (noise).
        unet: The U-Net model to use for noise prediction.
        steps: The number of reverse diffusion steps.

    Returns:
        The generated tensor.
    """
    z = z_t
    for step in range(steps, 0, -1):
        t = torch.full((z.size(0),), step - 1, device=DEVICE)
        pred_noise = unet(z, t)

        alpha_t = ALPHAS_CUMPROD[step - 1].view(1).to(DEVICE)
        alpha_t_prev = (
            ALPHAS_CUMPROD[step - 2].view(1).to(DEVICE)
            if step > 1
            else torch.tensor(1.0, device=DEVICE)
        )
        beta_t = BETAS[step - 1].view(1).to(DEVICE)

        z = (z - (1 - alpha_t) / torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(
            alpha_t_prev
        )

        if step > 1:
            z = z + torch.sqrt(beta_t) * torch.randn_like(z)

        if step % 100 == 0:
            print(
                f"Step {step}: z mean {z.mean().item():.4f}, std {z.std().item():.4f}"
            )

    return z


def binarize_tensor(tensor, threshold=0.5):
    """Binarizes a tensor."""
    return (tensor >= threshold).float()
