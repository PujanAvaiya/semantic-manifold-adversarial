import torch
import torch.nn.functional as F


def perturbation_norms(delta, thresholds=(1e-8, 1 / 255, 0.01, 0.05, 0.10)):
    """Return common L0-threshold counts plus L1/L2/Linf norms."""
    flat = delta.detach().flatten()
    result = {f"l0_gt_{threshold:g}": int((flat.abs() > threshold).sum().item()) for threshold in thresholds}
    result.update(
        {
            "l1": flat.abs().sum().item(),
            "l2": torch.norm(flat, p=2).item(),
            "linf": flat.abs().max().item(),
        }
    )
    return result


@torch.no_grad()
def prediction_summary(model, x):
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    return {
        "prediction": int(logits.argmax(dim=1).item()),
        "logits": logits.detach().cpu(),
        "probabilities": probs.detach().cpu(),
    }


def image_mse(x_a, x_b):
    return F.mse_loss(x_a, x_b).item()
