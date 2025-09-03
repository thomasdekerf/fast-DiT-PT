import torch

try:
    import kornia
except Exception:  # pragma: no cover
    kornia = None


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Structural similarity index. Returns ones if kornia missing."""
    if kornia is None:
        return torch.ones(x.size(0), device=x.device)
    return kornia.metrics.ssim(x, y, 11)
