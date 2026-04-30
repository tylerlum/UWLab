"""Student policy modules used by Isaac Lab-native distillation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class StudentOutput:
    action: torch.Tensor
    aux: dict[str, torch.Tensor]


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 10):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.proj(image)
        return x.flatten(2).transpose(1, 2)


class MonoTransformerRecurrentPolicy(nn.Module):
    """Small image+proprio recurrent policy.

    Input image tensors are expected to already be normalized by the environment
    camera path, so the model has no hidden image preprocessing.
    """

    def __init__(
        self,
        *,
        image_channels: int,
        proprio_dim: int,
        action_dim: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        hidden_dim: int = 512,
        patch_size: int = 10,
        aux_heads: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(image_channels, embed_dim, patch_size=patch_size)
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
        tokens = self.transformer(torch.cat([cls, patches], dim=1))
        image_feat = tokens[:, 0]
        proprio_feat = self.proprio_proj(proprio)
        next_hidden = self.rnn(torch.cat([image_feat, proprio_feat], dim=-1), hidden_state)
        action = self.actor(next_hidden)
        aux = {name: head(next_hidden) for name, head in self.aux_heads.items()}
        return StudentOutput(action=action, aux=aux), next_hidden


class MLPRecurrentPolicy(nn.Module):
    """Recurrent student for privileged/teacher observations."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        aux_heads: dict[str, int] | None = None,
    ) -> None:
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
