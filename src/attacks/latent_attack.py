import torch
import torch.nn.functional as F


def targeted_latent_margin_attack(
    model,
    victim,
    z,
    x_reference,
    target_class: int,
    latent_scale: float,
    steps: int = 500,
    lr: float = 0.03,
    lambda_norm: float = 0.10,
    beta_sem: float = 20.0,
    kappa: float = 0.10,
    p: int = 2,
):
    """Optimize an additive perturbation directly in latent space."""
    model.eval()
    victim.eval()

    delta = torch.zeros_like(z, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    history = {
        "total": [],
        "margin": [],
        "norm": [],
        "semantic": [],
        "prediction": [],
    }

    best_delta = None
    best_image = None
    best_norm = float("inf")

    for _ in range(steps):
        optimizer.zero_grad()
        x_adv = model.decode(z + delta)
        logits = victim(x_adv)

        target_logit = logits[:, target_class]
        masked = logits.clone()
        masked[:, target_class] = -1e9
        max_other = masked.max(dim=1).values
        margin_loss = F.relu(max_other - target_logit + kappa).mean()

        if p == 1:
            delta_norm = delta.abs().sum()
        elif p == 2:
            delta_norm = torch.norm(delta, p=2)
        else:
            raise ValueError("p must be 1 or 2")

        norm_loss = delta_norm / float(latent_scale)
        semantic_loss = F.mse_loss(x_adv, x_reference)
        total_loss = margin_loss + lambda_norm * norm_loss + beta_sem * semantic_loss

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = victim(x_adv).argmax(dim=1).item()
            current_norm = torch.norm(delta, p=2).item()
            if pred == target_class and current_norm < best_norm:
                best_norm = current_norm
                best_delta = delta.detach().clone()
                best_image = x_adv.detach().clone()

        history["total"].append(total_loss.item())
        history["margin"].append(margin_loss.item())
        history["norm"].append(current_norm)
        history["semantic"].append(semantic_loss.item())
        history["prediction"].append(pred)

    if best_delta is None:
        best_delta = delta.detach().clone()
        with torch.no_grad():
            best_image = model.decode(z + best_delta)

    return best_delta, best_image, history
