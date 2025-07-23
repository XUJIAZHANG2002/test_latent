import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def binarize_tensor(tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Binarizes a tensor by a given threshold."""
    return (tensor >= threshold).float()


def prepare_mnist_data(binarized_data_path: str = "binarized_mnist.pt", visualize: bool = True) -> None:
    """
    Prepares the MNIST dataset by binarizing it and saving it to a file.

    Args:
        binarized_data_path (str): The path to save the binarized data.
        visualize (bool): Whether to visualize the binarization effect.
    """
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    dataloader = DataLoader(mnist, batch_size=64, shuffle=False)

    binarized_images = []
    labels = []
    for x, y in dataloader:
        x_binarized = binarize_tensor(x, threshold=0.5)
        binarized_images.append(x_binarized)
        labels.append(y)

    binarized_images = torch.cat(binarized_images, dim=0)
    labels = torch.cat(labels, dim=0)
    torch.save({"images": binarized_images, "labels": labels}, binarized_data_path)
    print(f"Binarized MNIST dataset saved to '{binarized_data_path}'!")

    if visualize:
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(mnist.data[0].float() / 255, cmap="gray")
        plt.title("Original Grayscale")
        plt.subplot(1, 2, 2)
        plt.imshow(binarized_images[0].squeeze(), cmap="gray")
        plt.title("Binarized")
        plt.show()


def get_binarized_mnist_dataloader(
    binarized_data_path: str = "binarized_mnist.pt",
    batch_size: int = 1024,
    shuffle: bool = True,
) -> DataLoader:
    """
    Loads the binarized MNIST dataset and returns a DataLoader.

    Args:
        binarized_data_path (str): The path to the binarized data.
        batch_size (int): The batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: A DataLoader for the binarized MNIST dataset.
    """
    binarized_data = torch.load(binarized_data_path)
    binarized_dataset = TensorDataset(
        binarized_data["images"], binarized_data["labels"]
    )
    dataloader = DataLoader(binarized_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    prepare_mnist_data()
