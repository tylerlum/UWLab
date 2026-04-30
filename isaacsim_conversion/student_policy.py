from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StudentOutput:
    action: torch.Tensor
    aux: dict[str, torch.Tensor]


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.proj(image)
        return x.flatten(2).transpose(1, 2)


class MonoTransformerRecurrentPolicy(nn.Module):
    def __init__(
        self,
        image_channels: int,
        proprio_dim: int,
        action_dim: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        hidden_dim: int = 256,
        aux_heads: dict[str, int] | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(image_channels, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.rnn = nn.GRUCell(embed_dim * 2, hidden_dim)
        self.actor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.aux_heads = nn.ModuleDict()
        for name, size in (aux_heads or {}).items():
            self.aux_heads[name] = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, size),
            )
        self.hidden_dim = hidden_dim

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[StudentOutput, torch.Tensor]:
        patches = self.patch_embed(image)
        cls = self.cls_token.expand(image.shape[0], -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = self.transformer(tokens)
        img_feat = tokens[:, 0]
        proprio_feat = self.proprio_proj(proprio)
        fused = torch.cat([img_feat, proprio_feat], dim=-1)
        next_hidden = self.rnn(fused, hidden_state)
        action = self.actor(next_hidden)
        aux = {name: head(next_hidden) for name, head in self.aux_heads.items()}
        return StudentOutput(action=action, aux=aux), next_hidden


class MLPRecurrentPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        aux_heads: dict[str, int] | None = None,
    ):
        super().__init__()
        self.obs_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.actor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.aux_heads = nn.ModuleDict()
        for name, size in (aux_heads or {}).items():
            self.aux_heads[name] = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, size),
            )
        self.hidden_dim = hidden_dim

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[StudentOutput, torch.Tensor]:
        feat = self.obs_proj(obs)
        next_hidden = self.rnn(feat, hidden_state)
        action = self.actor(next_hidden)
        aux = {name: head(next_hidden) for name, head in self.aux_heads.items()}
        return StudentOutput(action=action, aux=aux), next_hidden


class MLPPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        aux_heads: dict[str, int] | None = None,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.actor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.aux_heads = nn.ModuleDict()
        for name, size in (aux_heads or {}).items():
            self.aux_heads[name] = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, size),
            )
        self.hidden_dim = hidden_dim

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor | None:
        return None

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[StudentOutput, torch.Tensor | None]:
        feat = self.backbone(obs)
        action = self.actor(feat)
        aux = {name: head(feat) for name, head in self.aux_heads.items()}
        return StudentOutput(action=action, aux=aux), None


def preprocess_image(image: torch.Tensor, modality: str) -> torch.Tensor:
    if modality == "depth":
        if image.dim() == 3:
            image = image.unsqueeze(1)
        return image
    if modality == "rgb":
        return image
    if modality == "rgbd":
        return image
    raise ValueError(f"Unsupported modality: {modality}")


def resize_image(image: torch.Tensor, height: int = 224, width: int = 224) -> torch.Tensor:
    _, _, src_h, src_w = image.shape
    if src_h == height and src_w == width:
        return image
    scale = min(height / src_h, width / src_w)
    resized_h = max(1, int(round(src_h * scale)))
    resized_w = max(1, int(round(src_w * scale)))
    resized = F.interpolate(image, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    pad_h = height - resized_h
    pad_w = width - resized_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom))
