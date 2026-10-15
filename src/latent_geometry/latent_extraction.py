import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_latents(model, loader, device):
    """Return deterministic VAE mean codes and labels for a data loader."""
    model.eval()
    latents = []
    labels = []

    for x, y in loader:
        x = x.to(device)
        mu, _ = model.encode(x)
        latents.append(mu.cpu())
        labels.append(y.cpu())

    return torch.cat(latents, dim=0), torch.cat(labels, dim=0)


@torch.no_grad()
def reconstruction_mse(model, loader, device):
    """Average summed reconstruction MSE per image using the latent mean."""
    model.eval()
    total_mse = 0.0
    total_images = 0

    for x, _ in loader:
        x = x.to(device)
        mu, _ = model.encode(x)
        x_hat = model.decode(mu)
        total_mse += F.mse_loss(x_hat, x, reduction="sum").item()
        total_images += x.size(0)

    return total_mse / total_images
