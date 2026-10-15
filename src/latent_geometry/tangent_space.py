import torch


def tangent_perturbation(coefficients, basis):
    """Map tangent coefficients a to a latent perturbation delta = a B^T."""
    return coefficients @ basis.T


def project_onto_tangent(delta, basis):
    """Orthogonally project a latent vector onto the span of an orthonormal basis."""
    return (delta @ basis) @ basis.T


def tangent_residual(delta, basis):
    """Return the component of delta orthogonal to the estimated tangent space."""
    return delta - project_onto_tangent(delta, basis)


def tangent_membership_error(delta, basis):
    """L2 magnitude of the component outside the estimated tangent space."""
    return torch.norm(tangent_residual(delta, basis), p=2).item()
