import os
import csv
from typing import Optional, Dict

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class PairedI2I(Dataset):
    """Dataset returning paired source/target images for I2I translation."""

    def __init__(self, root: str, split_csv: str = "split.csv", size: int = 256,
                 augment: bool = True, vae_cache_dir: Optional[str] = None):
        self.root = root
        self.size = size
        self.augment = augment
        self.vae_cache_dir = vae_cache_dir

        csv_path = os.path.join(root, split_csv)
        self.pairs = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    self.pairs.append((row[0], row[1]))

        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        if augment:
            self.augment_transform = transforms.RandomHorizontalFlip(p=0.5)
        else:
            self.augment_transform = lambda x: x

        if self.vae_cache_dir is not None:
            os.makedirs(self.vae_cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_image(self, rel_path: str) -> Image.Image:
        path = os.path.join(self.root, rel_path)
        return Image.open(path).convert("RGB")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        src_rel, tgt_rel = self.pairs[index]
        src_img = self._load_image(src_rel)
        tgt_img = self._load_image(tgt_rel)

        src_img = self.augment_transform(src_img)
        tgt_img = self.augment_transform(tgt_img)

        src_img = self.transform(src_img)
        tgt_img = self.transform(tgt_img)
        sample_id = os.path.splitext(os.path.basename(src_rel))[0]
        return {"src_img": src_img, "tgt_img": tgt_img, "id": sample_id}

    # ----------------- VAE utilities -----------------
    def encode_to_latents(self, vae, src_img: torch.Tensor, tgt_img: torch.Tensor, sample_id: str):
        """Encode images to latents using a provided VAE.
        Optionally cache the result to disk for faster subsequent loads."""
        z_src = z_tgt = None
        cache_path = None
        if self.vae_cache_dir is not None:
            cache_path = os.path.join(self.vae_cache_dir, f"{sample_id}.npz")
            if os.path.exists(cache_path):
                data = np.load(cache_path)
                z_src = torch.from_numpy(data["z_src"])  # type: ignore[arg-type]
                z_tgt = torch.from_numpy(data["z_tgt"])  # type: ignore[arg-type]
        if z_src is None or z_tgt is None:
            src_img = src_img.unsqueeze(0)
            tgt_img = tgt_img.unsqueeze(0)
            with torch.no_grad():
                z_src = vae.encode(src_img).latent_dist.sample() * 0.18215
                z_tgt = vae.encode(tgt_img).latent_dist.sample() * 0.18215
            z_src = z_src.squeeze(0)
            z_tgt = z_tgt.squeeze(0)
            if cache_path is not None:
                np.savez(cache_path, z_src=z_src.cpu().numpy(), z_tgt=z_tgt.cpu().numpy())
        return z_src, z_tgt
