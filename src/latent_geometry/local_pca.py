import torch


def estimate_local_tangent(
    z0,
    train_z,
    train_y,
    source_class: int,
    k_neighbors: int = 200,
    variance_threshold: float = 0.80,
    max_tangent_dim: int = 8,
):
    """Estimate a same-class local tangent basis around z0 using nearest-neighbor PCA/SVD.

    Returns a dictionary containing the orthonormal basis B with shape
    [latent_dim, tangent_dim], selected neighbors, singular values, and
    cumulative explained variance.
    """
    z0_cpu = z0.detach().cpu().float()
    z_all = train_z.detach().cpu().float()
    y_all = train_y.detach().cpu().long().view(-1)

    z_same = z_all[y_all == source_class]
    if len(z_same) == 0:
        raise ValueError(f"No training latents found for class {source_class}.")

    distances = torch.norm(z_same - z0_cpu, dim=1)
    k_local = min(k_neighbors, len(z_same))
    nearest = torch.topk(distances, k=k_local, largest=False).indices
    z_local = z_same[nearest]

    local_mean = z_local.mean(dim=0, keepdim=True)
    x_local = z_local - local_mean
    _, singular_values, vh = torch.linalg.svd(x_local, full_matrices=False)

    variance = singular_values.square()
    explained = variance / variance.sum().clamp_min(1e-12)
    cumulative = torch.cumsum(explained, dim=0)

    over = torch.where(cumulative >= variance_threshold)[0]
    tangent_dim = int(over[0].item() + 1) if len(over) else z_all.shape[1]
    tangent_dim = max(2, tangent_dim)
    tangent_dim = min(tangent_dim, max_tangent_dim, z_all.shape[1])

    basis = vh[:tangent_dim].T.contiguous()

    return {
        "basis": basis,
        "tangent_dim": tangent_dim,
        "neighbors": z_local,
        "neighbor_distances": distances[nearest],
        "local_mean": local_mean,
        "singular_values": singular_values,
        "explained_variance": explained,
        "cumulative_variance": cumulative,
    }


def orthogonality_error(basis):
    """Compute ||B^T B - I||_F for a candidate orthonormal basis."""
    eye = torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device)
    return torch.norm(basis.T @ basis - eye).item()
