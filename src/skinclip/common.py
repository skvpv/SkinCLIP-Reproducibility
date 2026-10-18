from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import random
import socket
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PKG_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict:

    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    return yaml.safe_load(os.path.expandvars(raw)) or {}


def load_paths(path: str | Path | None = None) -> dict:

    path = Path(path or os.environ.get("SKINCLIP_PATHS", PKG_ROOT / "config" / "paths.yaml"))
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; copy config/paths.yaml.template and set absolute paths.")
    cfg = load_yaml(path)
    for k in ("work_dir",):
        if k not in cfg:
            raise KeyError(f"paths config missing required key '{k}'")
    return cfg


def work_dir(paths: dict) -> Path:
    p = Path(paths["work_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:

    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


class IntegrityError(RuntimeError):
    pass


class FirewallError(RuntimeError):
    pass


TRAIN_SPLITS = ("train", "val")
EVAL_SPLITS = ("internal_test", "e1", "e1_oot", "e2", "e2_oot")


def integrity_dir(paths: dict) -> Path:
    d = work_dir(paths) / "integrity"
    d.mkdir(parents=True, exist_ok=True)
    return d


def require_cohort(paths: dict) -> dict:

    m = integrity_dir(paths) / "cohort_manifest.json"
    if not m.exists():
        raise IntegrityError("cohort manifest missing: data check not passed (run freeze_cohort after researcher confirmation).")
    man = read_json(m)
    for rel, digest in man["artifacts"].items():
        f = Path(rel)
        if not f.exists() or sha256_file(f) != digest:
            raise IntegrityError(f"cohort artifact changed or missing since freeze: {rel}")
    return man


def require_checkpoint(paths: dict) -> dict:
    m = integrity_dir(paths) / "checkpoint_manifest.json"
    if not m.exists():
        raise FirewallError("checkpoint manifest missing: evaluation sets are locked until all runs are frozen.")
    man = read_json(m)
    if man.get("status") != "FROZEN":
        raise FirewallError("checkpoint manifest not in FROZEN state.")
    for rel, digest in man["checkpoints"].items():
        if not Path(rel).exists() or sha256_file(rel) != digest:
            raise FirewallError(f"Frozen checkpoint changed or missing: {rel}")
    return man


def assert_split_access(split: str, purpose: str, paths: dict | None = None) -> None:

    if split in TRAIN_SPLITS:
        return
    if split not in EVAL_SPLITS:
        raise ValueError(f"unknown split '{split}'")
    if purpose != "evaluation":
        raise FirewallError(f"split '{split}' requested with purpose='{purpose}'; only 'evaluation' may read it.")
    if paths is None:
        raise FirewallError("evaluation access requires paths to verify the checkpoint manifest")
    require_checkpoint(paths)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def slurm_context() -> dict:
    keys = ["SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_PARTITION",
            "SLURM_JOB_ACCOUNT", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE"]
    ctx = {k: os.environ.get(k) for k in keys}
    ctx["host"] = socket.gethostname()
    ctx["inside_slurm"] = ctx["SLURM_JOB_ID"] is not None
    return ctx


def software_versions() -> dict:
    import importlib

    out = {}
    for mod in ["torch", "torchvision", "timm", "transformers", "open_clip", "numpy", "pandas", "sklearn", "scipy", "cv2", "imagehash", "PIL"]:
        try:
            m = importlib.import_module(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception as exc:
            out[mod] = f"NOT IMPORTABLE: {exc.__class__.__name__}"
    try:
        import torch

        out["cuda_build"] = torch.version.cuda
        out["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        pass
    return out


RUN_STATES = ("RUNNING", "TRAINING_COMPLETED", "COMPLETED", "FAILED")


class RunRecord:


    def __init__(self, run_dir: Path, run_id: str, meta: dict):
        self.path = Path(run_dir) / "run_record.json"
        if self.path.exists():
            self.data = read_json(self.path)
            if self.data["run_id"] != run_id:
                raise RuntimeError(f"run_id mismatch in {self.path}")
        else:
            self.data = {"run_id": run_id, "created": now_iso(), "history": [], **meta}
            self._save()

    def _save(self) -> None:
        write_json(self.path, self.data)

    def event(self, state: str, **info) -> None:
        if state not in RUN_STATES:
            raise ValueError(state)
        self.data["state"] = state
        self.data["history"].append({"time": now_iso(), "state": state, "slurm": slurm_context(), **info})
        self._save()

    def update(self, **kv) -> None:
        self.data.update(kv)
        self._save()


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **rec) -> None:
        rec.setdefault("time", now_iso())
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
