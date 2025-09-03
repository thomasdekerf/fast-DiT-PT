import torch
import torch.nn as nn
from typing import Optional


class DiTPT(nn.Module):
    """Wrapper adding paired token conditioning to a base DiT model."""

    def __init__(self, base_dit, hidden_size: int, use_direction_token: bool = False):
        super().__init__()
        self.dit = base_dit
        self.type_embed_src = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.type_embed_tgt = nn.Parameter(torch.zeros(1, 1, hidden_size))
        if use_direction_token:
            self.dir_embed = nn.Embedding(8, hidden_size)
        else:
            self.dir_embed = None
        nn.init.normal_(self.type_embed_src, std=0.02)
        nn.init.normal_(self.type_embed_tgt, std=0.02)
        if self.dir_embed is not None:
            nn.init.normal_((self.dir_embed.weight), std=0.02)

    def forward(self, z_src_clean: torch.Tensor, z_tgt_noisy: torch.Tensor,
                t: torch.Tensor, direction_id: Optional[torch.Tensor] = None,
                time_on: str = 'both') -> torch.Tensor:
        """Forward pass producing epsilon prediction for target tokens."""
        # patchify
        src_tok = self.dit.patchify(z_src_clean)
        tgt_tok = self.dit.patchify(z_tgt_noisy)

        src_tok = src_tok + self.type_embed_src
        tgt_tok = tgt_tok + self.type_embed_tgt

        te = self.dit.t_embedder(t)
        if time_on == 'both':
            src_tok = src_tok + te.unsqueeze(1)
        tgt_tok = tgt_tok + te.unsqueeze(1)

        prefix = []
        if getattr(self.dit, 'cls_token', None) is not None:
            prefix.append(self.dit.cls_token.expand(src_tok.size(0), -1, -1))
        if self.dir_embed is not None:
            assert direction_id is not None, "direction_id required when using direction token"
            prefix.append(self.dir_embed(direction_id).unsqueeze(1))

        seq = torch.cat(prefix + [src_tok, tgt_tok], dim=1)
        eps_all = self.dit.forward_tokens(seq, t)

        n_prefix = sum(p.size(1) for p in prefix)
        n_src = src_tok.size(1)
        eps_tgt = eps_all[:, n_prefix + n_src:, :]
        eps_tgt = self.dit.unpatchify_eps(eps_tgt, z_tgt_noisy.shape)
        return eps_tgt
