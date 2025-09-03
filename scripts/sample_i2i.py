import argparse
import torch
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

from models import DiT_models
from models.dit_pt import DiTPT
from sampling_i2i import i2i_sample


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    latent_size = args.image_size // 8
    base = DiT_models[args.model](input_size=latent_size, learn_sigma=False, num_classes=1)
    model = DiTPT(base, hidden_size=base.x_embedder.proj.out_channels,
                  use_direction_token=args.direction_token).to(device)
    state_dict = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    src_img = Image.open(args.src).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    src_tensor = transform(src_img).unsqueeze(0).to(device)
    direction_id = torch.tensor([0], device=device) if args.direction_token else None
    imgs = i2i_sample(model, vae, src_tensor, steps=args.steps,
                      direction_id=direction_id, time_on=args.time_on)
    save_image(imgs, args.out, normalize=True, value_range=(-1, 1))


if __name__ == '__main__':
    from torchvision import transforms
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/i2i_dit_pt.yaml')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--out', type=str, default='out.png')
    parser.add_argument('--model', type=str, default='DiT-S/8')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--vae', type=str, choices=['ema', 'sd_vae', 'mse'], default='sd_vae')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--direction-token', action='store_true')
    parser.add_argument('--time_on', type=str, choices=['tgt', 'both'], default='both')
    args = parser.parse_args()
    main(args)
