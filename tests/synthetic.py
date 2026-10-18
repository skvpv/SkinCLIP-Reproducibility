from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

HAM_DX = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
ISIC_OF = {"akiec": "AK", "bcc": "BCC", "bkl": "BKL", "df": "DF", "mel": "MEL", "nv": "NV", "vasc": "VASC"}
HAM_SITES = ["abdomen", "back", "chest", "face", "foot", "hand", "lower extremity", "neck", "scalp", "trunk",
             "upper extremity", "unknown", "acral", "ear", "genital"]
ISIC_SITES = ["anterior torso", "posterior torso", "lateral torso", "head/neck", "upper extremity", "lower extremity",
              "palms/soles", "oral/genital", None]
COLORS = {"akiec": (200, 80, 80), "bcc": (220, 170, 170), "bkl": (140, 100, 60), "df": (190, 150, 120),
          "mel": (40, 30, 40), "nv": (120, 80, 50), "vasc": (170, 30, 60), "SCC": (210, 120, 100), "UNK": (128, 128, 128)}


def lesion_image(rng, cls: str, w=64, h=48) -> Image.Image:

    yy, xx = np.mgrid[:h, :w].astype(float)
    ang = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(ang) * xx / w + np.sin(ang) * yy / h)[..., None] * rng.uniform(20, 70)
    img = np.array([225, 190, 170]) + rng.normal(0, 10, 3) + grad + rng.normal(0, 12, (h, w, 3))
    for k in range(rng.integers(1, 4)):
        cy, cx = rng.uniform(8, h - 8), rng.uniform(8, w - 8)
        a, b, th = rng.uniform(4, 16), rng.uniform(3, 12), rng.uniform(0, np.pi)
        u_ = ((xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)) / a
        v_ = (-(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)) / b
        m = u_ ** 2 + v_ ** 2 < 1
        img[m] = np.array(COLORS[cls]) * rng.uniform(0.7, 1.2) + rng.normal(0, 15, (m.sum(), 3))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def make(root: Path, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    root = Path(root)
    ham_dir, isic_dir = root / "raw/HAM10000/images", root / "raw/ISIC2019/ISIC_2019_Training_Input"
    ham_dir.mkdir(parents=True)
    isic_dir.mkdir(parents=True)
    rows, nid, les_no = [], 24306, 0
    for cls, n_les in zip(HAM_DX, [16, 18, 22, 14, 26, 40, 14]):
        for _ in range(n_les):
            les_no += 1
            lid = f"HAM_{les_no:07d}"
            k = rng.choice([1, 1, 2, 3])
            base_rng = np.random.default_rng(rng.integers(1 << 30))
            for _ in range(k):
                iid = f"ISIC_{nid:07d}"
                nid += 1
                lesion_image(np.random.default_rng(base_rng.integers(1 << 30)), cls).save(ham_dir / f"{iid}.jpg", quality=95)
                rows.append({"lesion_id": lid, "image_id": iid, "dx": cls, "dx_type": "histo",
                             "age": float(rng.choice([np.nan, *range(20, 85, 5)])), "sex": rng.choice(["male", "female", "unknown"]),
                             "localization": rng.choice(HAM_SITES)})
    ham = pd.DataFrame(rows)
    ham.to_csv(root / "raw/HAM10000/HAM10000_metadata.csv", index=False)

    gt, meta = [], []

    def add(iid, label, lesion, img: Image.Image, site=None, quality=95):
        img.save(isic_dir / f"{iid}.jpg", quality=quality)
        gt.append({"image": iid, **{c: float(c == label) for c in ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]}})
        meta.append({"image": iid, "age_approx": float(rng.choice([np.nan, *range(30, 90, 5)])),
                     "anatom_site_general": site if site is not None else rng.choice(ISIC_SITES),
                     "lesion_id": lesion, "sex": rng.choice(["male", "female", None])})

    ham_sample = ham.sample(90, random_state=seed).reset_index(drop=True)
    exact = ham_sample.iloc[:60]
    for r in exact.itertuples():
        add(r.image_id, ISIC_OF[r.dx], r.lesion_id, Image.open(ham_dir / f"{r.image_id}.jpg").copy())
    traps = {"exact_id": exact["image_id"].tolist(), "near_dup": [], "lesion_only": [], "near_dup_source": []}
    nd = ham_sample.iloc[60:80]
    for i, r in enumerate(nd.itertuples()):
        im = Image.open(ham_dir / f"{r.image_id}.jpg").convert("RGB")
        im = im.crop((1, 1, im.width - 1, im.height - 1)).resize((int(im.width * 1.25), int(im.height * 1.25)))
        iid = f"ISIC_{70000 + i:07d}"
        add(iid, ISIC_OF[r.dx], f"BCN_{9000 + i:07d}", im, quality=55)
        traps["near_dup"].append(iid)
        traps["near_dup_source"].append(r.image_id)
    lr = ham_sample.iloc[80]
    add("ISIC_0080000", ISIC_OF[lr.dx], lr.lesion_id, lesion_image(rng, lr.dx))
    traps["lesion_only"].append("ISIC_0080000")
    k = 0
    for lab, n in [("MEL", 30), ("NV", 40), ("BCC", 20), ("AK", 12), ("BKL", 20), ("DF", 10), ("VASC", 10), ("SCC", 14)]:
        cls = {v: kk for kk, v in ISIC_OF.items()}.get(lab, lab)
        for j in range(n):
            les = f"BCN_{k // 2:07d}" if j % 3 else (None if j % 2 else f"MSK4_{k:05d}")
            iid = f"ISIC_{60000 + k:07d}" + ("_downsampled" if k == 5 else "")
            add(iid, lab, les, lesion_image(rng, cls))
            k += 1
    pd.DataFrame(gt).to_csv(root / "raw/ISIC2019/ISIC_2019_Training_GroundTruth.csv", index=False)
    pd.DataFrame(meta).to_csv(root / "raw/ISIC2019/ISIC_2019_Training_Metadata.csv", index=False)
    return traps


def make_paths(root: Path) -> Path:
    root = Path(root)
    p = {"project_root": str(Path(__file__).resolve().parents[1]), "work_dir": str(root / "work"),
         "raw": {"ham_metadata_csv": str(root / "raw/HAM10000/HAM10000_metadata.csv"),
                 "ham_image_dirs": [str(root / "raw/HAM10000/images")],
                 "isic2019_train_gt_csv": str(root / "raw/ISIC2019/ISIC_2019_Training_GroundTruth.csv"),
                 "isic2019_train_meta_csv": str(root / "raw/ISIC2019/ISIC_2019_Training_Metadata.csv"),
                 "isic2019_train_image_dirs": [str(root / "raw/ISIC2019/ISIC_2019_Training_Input")],
                 "isic2019_test_gt_csv": None},
         "weights": {"hf_home": str(root / "hf"), "torch_home": str(root / "torch")}}
    f = root / "paths.yaml"
    f.write_text(yaml.safe_dump(p))
    return f


def make_config(root: Path, approve: bool = True) -> Path:

    src = Path(__file__).resolve().parents[1] / "config"
    dst = Path(root) / "config"
    shutil.copytree(src, dst)
    exp = yaml.safe_load((dst / "experiment.yaml").read_text())
    exp["synthetic"] = True
    exp["image"]["cache_size"] = 32
    exp["model"].update(image_backbone="vit_tiny_patch16_224", image_pretrained=False, img_size=32, biomedclip_feat_dim=16)
    exp["train"].update(batch_size=8, max_epochs=2, patience=1, num_workers=0, amp="none")
    exp["probe"].update(max_epochs=3, patience=2, batch_size=16)
    exp["audit"].update(embedding_model="vit_tiny_patch16_224", embedding_pretrained=False, embedding_img_size=32,
                        n_negative_pairs=0, fpr_strict=1e-2, fpr_review=5e-2)
    exp["stats"].update(n_boot=40, auroc_ci=False)
    exp["min_external_lesions_per_class"] = 1
    exp["interpret"].update(ig_steps=4, n_correct_per_class=1, n_incorrect_per_class=1, tsne_max_points=50)
    (dst / "experiment.yaml").write_text(yaml.safe_dump(exp, sort_keys=False))
    if approve:
        for f in ("taxonomy.yaml", "site_harmonisation.yaml", "source_attribution.yaml", "prompts.yaml"):
            d = yaml.safe_load((dst / f).read_text())
            d["status"] = "APPROVED"
            (dst / f).write_text(yaml.safe_dump(d, sort_keys=False))
    return dst


def sign_cohort(config_dir: Path, mode: str = "precautionary") -> None:
    dec = {"taxonomy": "APPROVED (test)", "site_table": "APPROVED (test)", "source_attribution": "APPROVED (test)",
           "prompts": "APPROVED (test)", "small_class_rule": "n/a (test)", "label_informed_calibration": "ACKNOWLEDGED (test)"}
    if mode == "precautionary":
        dec["near_duplicate_policy"] = "precautionary"
    else:
        dec["near_duplicate_adjudication_complete"] = True
    (Path(config_dir) / "integrity" / "cohort_approval.yaml").write_text(yaml.safe_dump(
        {"approved": True, "approver": "TEST SUITE", "date": "2026-10-16", "decisions": dec}))


def approve_precautionary(config_dir: Path, draft_summary: dict) -> None:

    pol = yaml.safe_load((Path(config_dir) / "near_duplicate_policy.yaml").read_text())
    pc = draft_summary["claims"]["e1_precautionary_exclusions"]["counts"]
    pol["status"] = "APPROVED"
    pol["expected"] = {**{k: int(v) for k, v in pc.items()}, "e1": draft_summary["n"]["e1"], "e1_oot": draft_summary["n"]["e1_oot"]}
    (Path(config_dir) / "near_duplicate_policy.yaml").write_text(yaml.safe_dump(pol, sort_keys=False))


def approve_configs(config_dir: Path) -> None:
    for f in ("taxonomy.yaml", "site_harmonisation.yaml", "source_attribution.yaml", "prompts.yaml"):
        d = yaml.safe_load((Path(config_dir) / f).read_text())
        d["status"] = "APPROVED"
        (Path(config_dir) / f).write_text(yaml.safe_dump(d, sort_keys=False))
