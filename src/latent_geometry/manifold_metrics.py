import torch


def class_centroids(latents, labels):
    """Compute one latent centroid per observed class."""
    latents = latents.detach().cpu().float()
    labels = labels.detach().cpu().long().view(-1)
    return {
        int(c): latents[labels == c].mean(dim=0)
        for c in torch.unique(labels)
    }


def class_radii(latents, labels):
    """Return mean and std of distance to the class centroid for each class."""
    latents = latents.detach().cpu().float()
    labels = labels.detach().cpu().long().view(-1)
    result = {}

    for c in torch.unique(labels):
        points = latents[labels == c]
        center = points.mean(dim=0, keepdim=True)
        distances = torch.norm(points - center, dim=1)
        result[int(c)] = {
            "mean": distances.mean().item(),
            "std": distances.std().item(),
        }

    return result


def typical_latent_scale(latents):
    """Median distance from the global latent center, used to normalize perturbation size."""
    latents = latents.detach().cpu().float()
    center = latents.mean(dim=0, keepdim=True)
    return torch.norm(latents - center, dim=1).median().item()
