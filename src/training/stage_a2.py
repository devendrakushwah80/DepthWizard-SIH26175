"""Stage A2 datasets and the reproducible height-aware crop sampler."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class ManifestEntry:
    sample_id: str
    split: str
    rgb_path: Path
    height_path: Path
    class_path: Path


def read_manifest(manifest_path: str | Path, data_dir: str | Path) -> list[ManifestEntry]:
    data_dir = Path(data_dir)
    rows: list[ManifestEntry] = []
    for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        rows.append(
            ManifestEntry(
                sample_id=parts[0],
                split=parts[1],
                rgb_path=data_dir / parts[2],
                height_path=data_dir / parts[3],
                class_path=data_dir / parts[4],
            )
        )
    return rows


def _read_h5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        keys = list(handle.keys())
        if not keys:
            raise ValueError(f"empty H5: {path}")
        return np.asarray(handle[keys[0]])


def _read_h5_crop(path: Path, top: int, left: int, height: int, width: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        dataset = handle[list(handle.keys())[0]]
        if dataset.ndim == 2:
            return np.asarray(dataset[top : top + height, left : left + width])
        if dataset.ndim == 3 and dataset.shape[0] in {1, 3}:
            return np.asarray(dataset[:, top : top + height, left : left + width])
        if dataset.ndim == 3:
            return np.asarray(dataset[top : top + height, left : left + width, :])
        raise ValueError(f"unsupported H5 array shape {dataset.shape}: {path}")


class HeightAwareCropSampler(Sampler[tuple[int, str, int, int, int]]):
    """Yield category-controlled crop requests with deterministic epoch shuffles.

    Candidate pools come from a fixed 3x3 overlapping grid audit.  Coordinates
    receive bounded jitter so a rare-tall base crop is not replayed identically
    every time.  Pools are permuted before reuse within an epoch.
    """

    CATEGORIES = ("random", "building_rich", "gt10", "gt20", "rare_tall")

    def __init__(
        self,
        dataset: "GAMUSStageA2Dataset",
        crop_index_path: str | Path,
        ratios: dict[str, float],
        seed: int = 42,
        samples_per_epoch: int | None = None,
        jitter: int = 96,
    ):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self.samples_per_epoch = samples_per_epoch or len(dataset)
        self.jitter = int(jitter)
        raw = np.asarray([max(0.0, float(ratios.get(k, 0.0))) for k in self.CATEGORIES])
        if raw.sum() <= 0:
            raise ValueError("at least one sampler ratio must be positive")
        self.ratios = raw / raw.sum()

        id_to_index = {entry.sample_id: i for i, entry in enumerate(dataset.entries)}
        self.pools: dict[str, list[tuple[int, int, int]]] = {
            category: [] for category in self.CATEGORIES[1:]
        }
        with Path(crop_index_path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["sample_id"] not in id_to_index:
                    continue
                candidate = (
                    id_to_index[row["sample_id"]],
                    int(row["top"]),
                    int(row["left"]),
                )
                for category in self.CATEGORIES[1:]:
                    if row.get(category, "0") in {"1", "True", "true"}:
                        self.pools[category].append(candidate)
        missing = [name for name, pool in self.pools.items() if not pool and self.ratios[self.CATEGORIES.index(name)] > 0]
        if missing:
            raise ValueError(f"crop index has no candidates for: {missing}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __iter__(self) -> Iterator[tuple[int, str, int, int, int]]:
        rng = np.random.default_rng(self.seed + 1_000_003 * self.epoch)
        exact = self.ratios * self.samples_per_epoch
        counts = np.floor(exact).astype(int)
        remainder = self.samples_per_epoch - int(counts.sum())
        for index in np.argsort(-(exact - counts))[:remainder]:
            counts[index] += 1
        categories = np.concatenate(
            [np.repeat(name, count) for name, count in zip(self.CATEGORIES, counts)]
        )
        rng.shuffle(categories)

        pool_orders: dict[str, np.ndarray] = {}
        pool_offsets = {name: 0 for name in self.pools}
        for name, pool in self.pools.items():
            pool_orders[name] = rng.permutation(len(pool))

        crop_h, crop_w = self.dataset.crop_size
        for request_number, category_value in enumerate(categories):
            category = str(category_value)
            if category == "random":
                tile_index = int(rng.integers(0, len(self.dataset)))
                height, width = self.dataset.tile_shapes[tile_index]
                top = int(rng.integers(0, max(1, height - crop_h + 1)))
                left = int(rng.integers(0, max(1, width - crop_w + 1)))
            else:
                pool = self.pools[category]
                offset = pool_offsets[category]
                if offset and offset % len(pool) == 0:
                    pool_orders[category] = rng.permutation(len(pool))
                selected = int(pool_orders[category][offset % len(pool)])
                pool_offsets[category] += 1
                tile_index, top, left = pool[selected]
                height, width = self.dataset.tile_shapes[tile_index]
                top = int(np.clip(top + rng.integers(-self.jitter, self.jitter + 1), 0, max(0, height - crop_h)))
                left = int(np.clip(left + rng.integers(-self.jitter, self.jitter + 1), 0, max(0, width - crop_w)))
            augmentation_seed = int(
                self.seed * 10_000_019 + self.epoch * self.samples_per_epoch + request_number
            )
            yield tile_index, category, top, left, augmentation_seed


class GAMUSStageA2Dataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        data_dir: str | Path = "data/gamus/dataset",
        dav2_cache_dir: str | Path = "data/gamus/cache_dav2",
        crop_size: tuple[int, int] = (512, 512),
        is_training: bool = True,
        augmentation_strength: str = "mild",
        require_depth: bool = True,
    ):
        self.entries = read_manifest(manifest_path, data_dir)
        self.crop_size = tuple(int(x) for x in crop_size)
        self.is_training = bool(is_training)
        self.augmentation_strength = augmentation_strength
        self.dav2_cache_dir = Path(dav2_cache_dir)
        self.require_depth = bool(require_depth)
        if augmentation_strength not in {"none", "mild", "moderate"}:
            raise ValueError("augmentation_strength must be none, mild, or moderate")
        self.tile_shapes: list[tuple[int, int]] = []
        for entry in self.entries:
            with h5py.File(entry.height_path, "r") as handle:
                shape = tuple(handle[list(handle.keys())[0]].shape)
            if len(shape) == 3:
                shape = shape[-2:]
            self.tile_shapes.append((int(shape[0]), int(shape[1])))

    def __len__(self) -> int:
        return len(self.entries)

    def _augmentation(self, rgb, height, semantic, mask, depth, rng):
        if self.augmentation_strength == "none":
            return rgb, height, semantic, mask, depth
        probability = 0.30 if self.augmentation_strength == "mild" else 0.50
        if rng.random() < probability:
            rgb, height, semantic, mask, depth = (
                np.fliplr(array) for array in (rgb, height, semantic, mask, depth)
            )
        if rng.random() < probability:
            rgb, height, semantic, mask, depth = (
                np.flipud(array) for array in (rgb, height, semantic, mask, depth)
            )
        if rng.random() < probability:
            k = int(rng.integers(1, 4))
            rgb, height, semantic, mask, depth = (
                np.rot90(array, k=k) for array in (rgb, height, semantic, mask, depth)
            )

        # Photometric transforms never touch targets or the frozen depth prior.
        magnitude = 0.08 if self.augmentation_strength == "mild" else 0.15
        if rng.random() < probability:
            rgb = rgb * float(rng.uniform(1.0 - magnitude, 1.0 + magnitude))
        if rng.random() < probability:
            contrast = float(rng.uniform(1.0 - magnitude, 1.0 + magnitude))
            rgb = (rgb - 0.5) * contrast + 0.5
        if rng.random() < probability:
            saturation = float(rng.uniform(1.0 - magnitude, 1.0 + magnitude))
            gray = rgb.mean(axis=2, keepdims=True)
            rgb = gray + saturation * (rgb - gray)
        return np.clip(rgb, 0.0, 1.0), height, semantic, mask, depth

    def __getitem__(self, request):
        if isinstance(request, tuple):
            index, category, top, left, augmentation_seed = request
        else:
            index = int(request)
            category = "center"
            augmentation_seed = index
            height_px, width_px = self.tile_shapes[index]
            crop_h, crop_w = self.crop_size
            top = max(0, (height_px - crop_h) // 2)
            left = max(0, (width_px - crop_w) // 2)

        entry = self.entries[int(index)]
        crop_h, crop_w = self.crop_size
        top, left = int(top), int(left)
        rgb = _read_h5_crop(entry.rgb_path, top, left, crop_h, crop_w)
        height = _read_h5_crop(entry.height_path, top, left, crop_h, crop_w).astype(np.float32)
        semantic = _read_h5_crop(entry.class_path, top, left, crop_h, crop_w).astype(np.int64)
        if rgb.ndim == 3 and rgb.shape[0] == 3:
            rgb = np.transpose(rgb, (1, 2, 0))
        if height.ndim == 3:
            height = np.squeeze(height)
        if semantic.ndim == 3:
            semantic = np.squeeze(semantic)
        rgb = rgb.astype(np.float32)
        if rgb.max() > 1.5:
            rgb /= 255.0
        mask = np.isfinite(height) & (height >= 0.0)
        mask &= (semantic >= 0) & (semantic <= 6)

        cache_path = self.dav2_cache_dir / f"{entry.sample_id}_dav2.npy"
        if cache_path.is_file():
            depth = np.asarray(
                np.load(cache_path, mmap_mode="r")[top : top + crop_h, left : left + crop_w],
                dtype=np.float32,
            )
        elif self.require_depth:
            raise FileNotFoundError(f"missing DAV2 cache: {cache_path}")
        else:
            depth = np.zeros_like(height, dtype=np.float32)

        if self.is_training:
            rgb, height, semantic, mask, depth = self._augmentation(
                rgb,
                height,
                semantic,
                mask,
                depth,
                np.random.default_rng(augmentation_seed),
            )

        valid_height = height[mask]
        building_fraction = float(np.mean((semantic == 3) & mask))
        max_height = float(np.max(valid_height)) if valid_height.size else 0.0
        return {
            "sample_id": entry.sample_id,
            "rgb": torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float(),
            "height": torch.from_numpy(np.ascontiguousarray(height[None])).float(),
            "semantic": torch.from_numpy(np.ascontiguousarray(semantic)).long(),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(mask[None])).bool(),
            "rel_depth": torch.from_numpy(np.ascontiguousarray(depth[None])).float(),
            "sampling_category": category,
            "crop_top": top,
            "crop_left": left,
            "crop_max_gt": max_height,
            "building_fraction": building_fraction,
        }
