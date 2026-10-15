"""Sparse tangent-space optimization utilities.

The original exploratory implementation is preserved in the archived notebook.
A cleaned reusable implementation will be migrated here after the baseline
experiment is reproduced from the modular code path.
"""

import torch


def hard_threshold(x, k_active: int):
    """Keep only the k largest-magnitude entries and zero the rest."""
    if k_active >= x.numel():
        return x.clone()
    flat = x.flatten()
    keep = torch.topk(flat.abs(), k=k_active).indices
    out = torch.zeros_like(flat)
    out[keep] = flat[keep]
    return out.view_as(x)
