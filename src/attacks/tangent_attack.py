import torch
import torch.nn.functional as F

from src.latent_geometry.tangent_space import tangent_perturbation


def targeted_tangent_attack(
    model,
    victim,
    z0,
    x_reference,
    basis,
    target_class: int,
    latent_scale: float,
    steps: int = 700,
    lr: float = 0.03,
    lambda_norm: float = 0.15,
    beta_sem: float = 10.0,
    kappa: float = 0.05,
):
    """Optimize only tangent coefficients a, with delta = a B^T."""
    device = z0.device
    tangent_dim = basis.shape[1]
    basis = basis.to(device=device, dtype=z0.dtype)

    model.eval()
    victim.eval()

    a = torch.zeros((1, tangent_dim), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([a], lr=lr)

    history = {
        "loss": [],
        "margin": [],
        "latent_norm": [],
        "image_mse": [],
        "prediction": [],
    }
    best = None
    best_norm = float("inf")

    for step in range(steps):
        optimizer.zero_grad()

        delta = tangent_perturbation(a, basis)
        z_adv = z0 + delta
        x_adv = model.decode(z_adv)
        logits = victim(x_adv)

        target_logit = logits[:, target_class]
        masked = logits.clone()
        masked[:, target_class] = -1e9
        max_other = masked.max(dim=1).values
        margin_loss = F.relu(max_other - target_logit + kappa).mean()

        latent_norm = torch.norm(delta, p=2)
        normalized_norm = latent_norm / float(latent_scale)
        image_mse = F.mse_loss(x_adv, x_reference)
        loss = margin_loss + lambda_norm * normalized_norm + beta_sem * image_mse

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta_eval = tangent_perturbation(a, basis)
            x_eval = model.decode(z0 + delta_eval)
            logits_eval = victim(x_eval)
            pred = logits_eval.argmax(dim=1).item()
            current_norm = torch.norm(delta_eval, p=2).item()

            if pred == target_class and current_norm < best_norm:
                best_norm = current_norm
                best = {
                    "step": step,
                    "coefficients": a.detach().clone(),
                    "delta": delta_eval.detach().clone(),
                    "image": x_eval.detach().clone(),
                    "prediction": pred,
                    "norm": current_norm,
                }

        history["loss"].append(loss.item())
        history["margin"].append(margin_loss.item())
        history["latent_norm"].append(current_norm)
        history["image_mse"].append(image_mse.item())
        history["prediction"].append(pred)

    if best is None:
        with torch.no_grad():
            delta_eval = tangent_perturbation(a, basis)
            x_eval = model.decode(z0 + delta_eval)
            pred = victim(x_eval).argmax(dim=1).item()
        best = {
            "step": steps - 1,
            "coefficients": a.detach().clone(),
            "delta": delta_eval.detach().clone(),
            "image": x_eval.detach().clone(),
            "prediction": pred,
            "norm": torch.norm(delta_eval, p=2).item(),
        }

    return best, history
