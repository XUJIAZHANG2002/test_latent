import torch

def get_mask(
    original_mask: torch.Tensor,
    scale: float = 0.25,
    mask_mode: str = "exists",
) -> torch.Tensor:
    """
    Get the mask for the original image.

    Args:
        original_mask: The original mask.
        mask_mode: The mode of the mask.
    """
    if mask_mode == "exists":
        mask = original_mask
    elif mask_mode == "forall":
        mask = torch.ones_like(original_mask)
    elif mask_mode == "none":
        mask = torch.zeros_like(original_mask)
    elif mask_mode == "random":
        mask = torch.rand_like(original_mask)
    else:
        raise ValueError(f"Invalid mask mode: {mask_mode}")
    return mask


def exists_mask(original_mask: torch.Tensor, scale: float = 0.25) -> torch.Tensor:
    """
    Downsample mask to fit scaling size. If scale is 0.25, the mask will be downsampled by 4.
    If the exists a 1 in the original mask, the mask will be 1 in the downsampled mask.

    Args:
        original_mask: The original mask.
        scale: The scale of the mask.
    """
    return original_mask.view(1, 1, -1, -1).repeat(1, 1, 4, 4)[:, :, : scale * original_mask.size(2), : scale * original_mask.size(3)]