from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .factory import CLIP_MEAN, CLIP_STD


def load_biomedclip(model_id: str, device: str):
    import open_clip

    model, _ = open_clip.create_model_from_pretrained(model_id)
    tok = open_clip.get_tokenizer(model_id)
    return model.eval().to(device), tok


@torch.no_grad()
def image_features(model, images_uint8: np.ndarray, device: str, batch: int = 128) -> np.ndarray:
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
    out = []
    for s in range(0, len(images_uint8), batch):
        x = torch.from_numpy(np.asarray(images_uint8[s:s + batch])).permute(0, 3, 1, 2).float().div(255).to(device)
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        f = model.encode_image((x - mean) / std)
        out.append(F.normalize(f.float(), dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def zero_shot_logits(model, tokenizer, img_feats: np.ndarray, prompts: list[str], device: str) -> np.ndarray:
    t = tokenizer(prompts, context_length=256).to(device)
    tf = F.normalize(model.encode_text(t).float(), dim=-1)
    s = model.logit_scale.exp().float()
    return (s * torch.from_numpy(img_feats).to(device) @ tf.t()).cpu().numpy()
