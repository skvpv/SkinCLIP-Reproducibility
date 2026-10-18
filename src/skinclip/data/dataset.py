from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T


def build_augment(cfg: dict) -> T.Compose:
    return T.Compose([
        T.RandomRotation(cfg["rotation_deg"]),
        T.RandomHorizontalFlip(cfg["hflip_p"]),
        T.RandomVerticalFlip(cfg["vflip_p"]),
        T.ColorJitter(cfg["brightness"], cfg["contrast"], cfg["saturation"], cfg["hue"]),
    ])


class LesionDataset(Dataset):


    def __init__(self, images, meta: np.ndarray, labels: np.ndarray, mean, std, augment: T.Compose | None = None,
                 out_size: int | None = None, features: np.ndarray | None = None):
        self.images, self.meta, self.labels, self.features = images, meta.astype(np.float32), labels.astype(np.int64), features
        self.augment = augment
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.out_size = out_size

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {"meta": torch.from_numpy(self.meta[i]), "label": torch.tensor(self.labels[i]), "index": torch.tensor(i)}
        if self.features is not None:
            item["features"] = torch.from_numpy(np.asarray(self.features[i], dtype=np.float32))
        if self.images is not None:
            x = torch.from_numpy(np.array(self.images[i])).permute(2, 0, 1)
            if self.augment is not None:
                x = self.augment(x)
            x = x.float().div_(255.0)
            if self.out_size is not None and x.shape[-1] != self.out_size:
                x = torch.nn.functional.interpolate(x[None], size=(self.out_size, self.out_size), mode="bilinear",
                                                    align_corners=False)[0]
            item["image"] = (x - self.mean) / self.std
        return item


def _worker_init(_):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, seed: int, epoch: int, num_workers: int,
                drop_last: bool = False) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(int(seed) * 100_003 + int(epoch))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=g,
                      worker_init_fn=_worker_init, pin_memory=torch.cuda.is_available(), drop_last=drop_last,
                      persistent_workers=False)
