from __future__ import annotations

import math

import numpy as np

import torch
import torch.nn.functional as F


def class_weights(labels, num_classes: int) -> torch.Tensor:

    counts = torch.bincount(torch.as_tensor(np.array(labels, copy=True), dtype=torch.long), minlength=num_classes).float()
    if (counts == 0).any():
        raise ValueError(f"class with zero training samples: counts={counts.tolist()}")
    w = counts.sum() / (num_classes * counts)
    return w / w.mean()


def focal_loss(logits: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, gamma: float = 2.0, alpha: float = 1.0,
               smoothing: float = 0.1) -> torch.Tensor:

    K = logits.shape[-1]
    logp = F.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    q = torch.full_like(logp, smoothing / K)
    q.scatter_(1, y[:, None], 1.0 - smoothing + smoothing / K)
    per = -(alpha * weights[None, :] * q * (1.0 - p).pow(gamma) * logp).sum(dim=1)
    return per.mean()


def i2t_loss(logits: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:

    nll = F.cross_entropy(logits.float(), y, reduction="none")
    wy = weights[y]
    return (wy * nll).sum() / wy.sum()


def t2i_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:


    a = logits.float()
    B, K = a.shape
    logrho = a - torch.logsumexp(a, dim=0, keepdim=True)
    onehot = F.one_hot(y, K).float()
    n = onehot.sum(dim=0)
    present = n > 0
    sum_pos = (onehot * logrho).sum(dim=0)
    Lc = -sum_pos[present] / n[present] - torch.log(n[present])
    return Lc.mean()


def clip_class_prototype_loss(logits: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:

    return 0.5 * (i2t_loss(logits, y, weights) + t2i_loss(logits, y))


def total_loss(logits, y, weights, lam: float, gamma: float, alpha: float, smoothing: float) -> tuple[torch.Tensor, dict]:
    l_cls = focal_loss(logits, y, weights, gamma, alpha, smoothing)
    if lam > 0:
        l_clip = clip_class_prototype_loss(logits, y, weights)
        return l_cls + lam * l_clip, {"loss_cls": float(l_cls.detach()), "loss_clip": float(l_clip.detach())}
    return l_cls, {"loss_cls": float(l_cls.detach()), "loss_clip": math.nan}
