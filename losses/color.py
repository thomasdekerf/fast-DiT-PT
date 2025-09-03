import torch

try:
    from kornia.color import rgb_to_lab
    from kornia.metrics import delta_e
except Exception:  # pragma: no cover
    rgb_to_lab = None
    delta_e = None


def deltaE2000(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Approximate DeltaE2000 color difference.
    If kornia is not available, returns zeros to allow training to continue."""
    if rgb_to_lab is None or delta_e is None:
        return torch.zeros(x.size(0), device=x.device)
    x_lab = rgb_to_lab((x + 1) / 2)
    y_lab = rgb_to_lab((y + 1) / 2)
    return delta_e(x_lab, y_lab, method='2000').view(x.size(0), -1).mean(dim=1)
