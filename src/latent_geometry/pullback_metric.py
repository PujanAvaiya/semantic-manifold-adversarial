from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class PullbackResult:
    """Local stochastic pullback geometry for one input sample."""

    metric: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    jacobian_mu: torch.Tensor
    jacobian_sigma: torch.Tensor

    @property
    def spectral_radius(self) -> torch.Tensor:
        return self.eigenvalues[0]

    @property
    def dominant_direction(self) -> torch.Tensor:
        return self.eigenvectors[:, 0]


def _require_single_sample(x: torch.Tensor) -> None:
    if x.ndim < 2 or x.shape[0] != 1:
        raise ValueError(
            "This reference implementation expects one sample at a time "
            "(batch dimension must equal 1)."
        )


def encoder_mean_std(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return VAE encoder mean and standard deviation.

    The repository ConvVAE exposes ``encode(x) -> (mu, logvar)``.  The paper
    models the stochastic encoder as ``f(x)=mu(x)+eps*sigma(x)``, therefore
    ``sigma(x)=exp(0.5*logvar(x))``.
    """

    mu, logvar = model.encode(x)
    sigma = torch.exp(0.5 * logvar)
    return mu, sigma


def encoder_jacobians(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute J_mu(x) and J_sigma(x) for one sample.

    Returns tensors of shape ``[latent_dim, input_dim]``.
    """

    _require_single_sample(x)
    x_req = x.detach().clone().requires_grad_(True)
    mu, sigma = encoder_mean_std(model, x_req)

    j_mu = []
    j_sigma = []

    for i in range(mu.shape[1]):
        grad_mu = torch.autograd.grad(
            mu[0, i],
            x_req,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        grad_sigma = torch.autograd.grad(
            sigma[0, i],
            x_req,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]

        j_mu.append(grad_mu.reshape(-1))
        j_sigma.append(grad_sigma.reshape(-1))

    return torch.stack(j_mu, dim=0), torch.stack(j_sigma, dim=0)


def stochastic_pullback_metric(
    model,
    x: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float64,
) -> PullbackResult:
    r"""Compute the expected stochastic pullback metric from Eq. (4).

    Implements

        G_hat_x = J_mu(x)^T J_mu(x) + J_sigma(x)^T J_sigma(x)

    for one input sample. Eigenpairs are returned in descending eigenvalue
    order. For MNIST this produces a 784 x 784 matrix.
    """

    j_mu, j_sigma = encoder_jacobians(model, x)
    j_mu = j_mu.to(dtype=dtype)
    j_sigma = j_sigma.to(dtype=dtype)

    metric = j_mu.T @ j_mu + j_sigma.T @ j_sigma
    metric = 0.5 * (metric + metric.T)  # numerical symmetrisation

    eigenvalues, eigenvectors = torch.linalg.eigh(metric)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    eigenvectors = eigenvectors[:, order]

    return PullbackResult(
        metric=metric,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        jacobian_mu=j_mu,
        jacobian_sigma=j_sigma,
    )


def paper_anisotropy_score(
    eigenvalues: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Reproduce the public-code robustness/anisotropy score.

    The authors' public implementation normalises the eigenvalue spectrum by
    its maximum before computing ``-sum(s * log(s))``. We preserve that
    convention instead of silently replacing it with trace-normalised entropy.
    """

    values = eigenvalues.abs()
    if values.numel() == 0:
        raise ValueError("eigenvalues must not be empty")

    vmax = values.max().clamp_min(eps)
    scaled = values / vmax
    return -(scaled * torch.log(scaled + eps)).sum()


def perturb_along_eigendirection(
    x: torch.Tensor,
    eigenvalue: torch.Tensor | float,
    eigenvector: torch.Tensor,
    delta: float,
    *,
    scale_by_eigenvalue: bool = True,
    clamp: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> torch.Tensor:
    r"""Apply the paper's one-step corruption.

    The paper describes

        x_c = x + delta * lambda * eta

    where ``eta`` is an eigenvector and ``lambda`` is its eigenvalue. Setting
    ``scale_by_eigenvalue=False`` is useful only as a diagnostic that isolates
    direction from the spectrum magnitude.
    """

    _require_single_sample(x)

    direction = eigenvector.to(device=x.device, dtype=x.dtype)
    direction = direction / direction.norm().clamp_min(torch.finfo(x.dtype).eps)

    scale = float(delta)
    if scale_by_eigenvalue:
        if isinstance(eigenvalue, torch.Tensor):
            eigenvalue = float(eigenvalue.detach().cpu())
        scale *= float(eigenvalue)

    x_corrupt = (x.reshape(-1) + scale * direction).reshape_as(x)
    if clamp is not None:
        x_corrupt = x_corrupt.clamp(*clamp)
    return x_corrupt


@torch.no_grad()
def mean_reconstruction(model, x: torch.Tensor) -> torch.Tensor:
    """Deterministic reconstruction using the encoder mean."""

    mu, _ = model.encode(x)
    return model.decode(mu)


@torch.no_grad()
def latent_mean_shift(model, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
    """Euclidean distance between the mean latent codes of two inputs."""

    mu_a, _ = model.encode(x_a)
    mu_b, _ = model.encode(x_b)
    return torch.linalg.vector_norm(mu_b - mu_a, dim=1).mean()


def matched_random_direction(
    reference: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample a random unit vector with the same shape as ``reference``."""

    direction = torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )
    return direction / direction.norm().clamp_min(torch.finfo(direction.dtype).eps)


def direction_sweep(
    model,
    x: torch.Tensor,
    result: PullbackResult,
    deltas: Sequence[float],
    *,
    direction_index: int = 0,
    scale_by_eigenvalue: bool = True,
    clamp: Optional[Tuple[float, float]] = (0.0, 1.0),
):
    """Evaluate latent and reconstruction changes across corruption steps."""

    lam = result.eigenvalues[direction_index]
    eta = result.eigenvectors[:, direction_index]

    rows = []
    clean_recon = mean_reconstruction(model, x)
    clean_recon_mse = F.mse_loss(clean_recon, x, reduction="mean").item()

    for delta in deltas:
        x_corrupt = perturb_along_eigendirection(
            x,
            lam,
            eta,
            float(delta),
            scale_by_eigenvalue=scale_by_eigenvalue,
            clamp=clamp,
        )
        recon = mean_reconstruction(model, x_corrupt)

        rows.append(
            {
                "delta": float(delta),
                "direction_index": int(direction_index),
                "eigenvalue": float(lam.detach().cpu()),
                "input_l2": float(
                    torch.linalg.vector_norm((x_corrupt - x).reshape(-1)).cpu()
                ),
                "latent_shift": float(latent_mean_shift(model, x, x_corrupt).cpu()),
                "recon_mse_to_original": float(
                    F.mse_loss(recon, x, reduction="mean").cpu()
                ),
                "clean_recon_mse": float(clean_recon_mse),
            }
        )

    return rows
