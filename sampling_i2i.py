import torch
from diffusion import create_diffusion
from typing import Optional


def i2i_sample(dit_pt, vae, src_img: torch.Tensor, steps: int = 20,
               direction_id: Optional[torch.Tensor] = None,
               time_on: str = 'both', scheduler: str = 'dpm_solver'):
    """Simple i2i sampling loop."""
    device = next(dit_pt.parameters()).device
    diffusion = create_diffusion(str(steps))
    with torch.no_grad():
        if src_img.dim() == 3:
            src_img = src_img.unsqueeze(0)
        z_src = vae.encode(src_img).latent_dist.sample() * 0.18215
        z = torch.randn_like(z_src)

        def model_fn(x, t):
            return dit_pt(z_src, x, t, direction_id, time_on=time_on)

        samples = diffusion.p_sample_loop(model_fn, z.shape, z, progress=False, device=device)
        imgs = vae.decode(samples / 0.18215).sample
        return imgs
