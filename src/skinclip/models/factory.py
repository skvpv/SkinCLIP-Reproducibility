from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .. import CLASSES, NUM_CLASSES
from ..common import load_yaml, sha256_file
from .core import LinearProbe, MetaBlockNet, MetaOnlyNet, PrototypeNet

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def build_vit(name: str, pretrained: bool, img_size: int):
    import timm

    kw = {} if img_size == 224 else {"img_size": img_size}
    m = timm.create_model(name, pretrained=pretrained, num_classes=0, **kw)
    cfg = timm.data.resolve_data_config({}, model=m)
    info = {"timm_name": name, "pretrained": pretrained,
            "pretrained_tag": getattr(m, "pretrained_cfg", {}).get("tag"), "hf_hub_id": getattr(m, "pretrained_cfg", {}).get("hf_hub_id")}
    return m, m.num_features, tuple(cfg["mean"]), tuple(cfg["std"]), info


def build_resnet50(pretrained: bool):
    import torchvision

    w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    return torchvision.models.resnet50(weights=w), {"torchvision": "resnet50", "weights": str(w)}


def prompts_from_config(path: str | Path) -> tuple[list[str], str]:
    cfg = load_yaml(path)
    if cfg["order"] != CLASSES:
        raise ValueError(f"prompt order {cfg['order']} != CLASSES {CLASSES}")
    return [cfg["prompts"][c] for c in CLASSES], sha256_file(path)


def encode_prompts(model_name: str, prompts: list[str], max_len: int) -> tuple[torch.Tensor, dict]:

    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    lens = [len(tok(p)["input_ids"]) for p in prompts]
    if max(lens) > max_len:
        raise ValueError(f"prompt token lengths {lens} exceed text_max_len={max_len}")
    model = AutoModel.from_pretrained(model_name).eval()
    with torch.no_grad():
        enc = tok(prompts, padding=True, truncation=False, return_tensors="pt")
        H = model(**enc).last_hidden_state[:, 0].float()
    info = {"text_encoder": model_name, "token_lengths": lens, "dim": int(H.shape[1]),
            "commit": getattr(model.config, "_commit_hash", None)}
    return H, info


def synthetic_text_bank(dim: int = 64, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(NUM_CLASSES, dim, generator=g)


def text_bank_path(work_dir: str | Path) -> Path:
    return Path(work_dir) / "text_bank" / "H.pt"


def save_text_bank(work_dir, H: torch.Tensor, info: dict, prompts_sha: str) -> Path:
    p = text_bank_path(work_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"H": H, "info": info, "prompts_sha256": prompts_sha}, p)
    return p


def load_text_bank(work_dir, prompts_sha: str) -> tuple[torch.Tensor, dict]:
    obj = torch.load(text_bank_path(work_dir), map_location="cpu", weights_only=False)
    if obj["prompts_sha256"] != prompts_sha:
        raise RuntimeError("text bank was encoded from a different prompts.yaml; re-run the text-bank job")
    return obj["H"], obj["info"]


def build_model(arm: str, exp: dict, text_H: torch.Tensor | None = None):

    a, m = exp["arms"][arm], exp["model"]
    kind = a["kind"]
    if kind == "prototype":
        enc, fdim, mean, std, info = build_vit(m["image_backbone"], m["image_pretrained"], m["img_size"])
        model = PrototypeNet(enc, fdim, a["use_meta"], a["protos"], NUM_CLASSES, m["embed_dim"], m["meta_hidden"],
                             m["fusion_hidden"], m["dropout"], text_H if a["protos"] == "text" else None,
                             m["logit_scale_init"], m["logit_scale_min"], m["logit_scale_max"])
        return model, (mean, std), info
    if kind == "metablock":
        r, info = build_resnet50(m["image_pretrained"])
        return MetaBlockNet(r, NUM_CLASSES, feat_maps=m["metablock"]["feat_maps"]), (IMAGENET_MEAN, IMAGENET_STD), info
    if kind == "meta_only":
        return MetaOnlyNet(NUM_CLASSES, m["embed_dim"], m["meta_hidden"], m["dropout"]), (IMAGENET_MEAN, IMAGENET_STD), {}
    if kind == "biomedclip_probe":
        return LinearProbe(exp["model"].get("biomedclip_feat_dim", 512), NUM_CLASSES, True), (CLIP_MEAN, CLIP_STD), {}
    raise ValueError(f"arm {arm} kind {kind} is not trainable")


def needs_images(arm: str, exp: dict) -> bool:
    return exp["arms"][arm]["kind"] in ("prototype", "metablock")


def needs_features(arm: str, exp: dict) -> bool:
    return exp["arms"][arm]["kind"] == "biomedclip_probe"
