from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.metadata import META_DIM


class MetadataEncoder(nn.Module):


    def __init__(self, in_dim: int = META_DIM, hidden: int = 128, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.net(m)


class PrototypeNet(nn.Module):


    def __init__(self, image_encoder: nn.Module, feat_dim: int, use_meta: bool, protos: str, num_classes: int,
                 embed_dim: int = 256, meta_hidden: int = 128, fusion_hidden: int = 512, dropout: float = 0.1,
                 text_H: torch.Tensor | None = None, logit_scale_init: float = 1 / 0.07,
                 logit_scale_min: float = 1.0, logit_scale_max: float = 100.0):
        super().__init__()
        if protos not in ("text", "learned"):
            raise ValueError(protos)
        self.use_meta, self.protos, self.num_classes = use_meta, protos, num_classes
        self.image_encoder = image_encoder
        self.img_proj = nn.Linear(feat_dim, embed_dim)
        self.meta_enc = MetadataEncoder(META_DIM, meta_hidden, embed_dim, dropout) if use_meta else None
        in_f = 2 * embed_dim if use_meta else embed_dim
        self.fusion = nn.Sequential(nn.Linear(in_f, fusion_hidden), nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(fusion_hidden, embed_dim))
        if protos == "text":
            if text_H is None or text_H.shape[0] != num_classes:
                raise ValueError("text prototypes need text_H with one row per class")
            self.register_buffer("text_H", text_H.detach().float().clone())
            self.text_proj = nn.Linear(text_H.shape[1], embed_dim)
        else:
            self.V = nn.Parameter(torch.randn(num_classes, embed_dim))
        self.theta = nn.Parameter(torch.tensor(math.log(logit_scale_init)))
        self.s_min, self.s_max = logit_scale_min, logit_scale_max

    def prototypes(self) -> torch.Tensor:
        p = self.text_proj(self.text_H) if self.protos == "text" else self.V
        return F.normalize(p, dim=-1)

    def scale(self) -> torch.Tensor:
        return self.theta.exp().clamp(self.s_min, self.s_max)

    def embed(self, image: torch.Tensor, meta: torch.Tensor | None = None) -> torch.Tensor:
        z_i = F.normalize(self.img_proj(self.image_encoder(image)), dim=-1)
        if self.use_meta:
            z_m = F.normalize(self.meta_enc(meta), dim=-1)
            h = torch.cat([z_i, z_m], dim=-1)
        else:
            h = z_i
        return F.normalize(self.fusion(h), dim=-1)

    def forward(self, image: torch.Tensor, meta: torch.Tensor | None = None, features=None) -> dict:
        q = self.embed(image, meta)
        p = self.prototypes()
        logits = self.scale() * q.float() @ p.float().t()
        return {"logits": logits, "q": q}


class MetaBlock(nn.Module):


    def __init__(self, V: int, U: int):
        super().__init__()
        self.fb = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))
        self.gb = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))

    def forward(self, V: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        t1 = self.fb(U)
        t2 = self.gb(U)
        return torch.sigmoid(torch.tanh(V * t1.unsqueeze(-1)) + t2.unsqueeze(-1))


class MetaBlockNet(nn.Module):


    def __init__(self, resnet: nn.Module, num_classes: int, n_meta: int = META_DIM, feat_maps: int = 64,
                 n_feat_conv: int = 2048):
        super().__init__()
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feat_maps, self.n_feat_conv = feat_maps, n_feat_conv
        self.comb = MetaBlock(feat_maps, n_meta)
        self.classifier = nn.Linear(n_feat_conv, num_classes)

    def forward(self, image: torch.Tensor, meta: torch.Tensor, features=None) -> dict:
        x = self.features(image)
        x = x.view(x.size(0), self.feat_maps, self.n_feat_conv // self.feat_maps, -1).squeeze(-1)
        x = self.comb(x, meta.float())
        return {"logits": self.classifier(x.view(x.size(0), -1))}


class MetaOnlyNet(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 256, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.enc = MetadataEncoder(META_DIM, hidden, embed_dim, dropout)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, image=None, meta: torch.Tensor = None, features=None) -> dict:
        return {"logits": self.head(F.normalize(self.enc(meta), dim=-1))}


class LinearProbe(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int, use_meta: bool = True):
        super().__init__()
        self.use_meta = use_meta
        self.head = nn.Linear(feat_dim + (META_DIM if use_meta else 0), num_classes)

    def forward(self, image=None, meta: torch.Tensor = None, features: torch.Tensor = None) -> dict:
        x = torch.cat([features, meta], dim=-1) if self.use_meta else features
        return {"logits": self.head(x)}
