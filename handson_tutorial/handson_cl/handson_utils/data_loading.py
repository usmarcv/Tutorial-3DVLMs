"""Lightweight ShapeNet subset loading for the 3D-VLM contrastive-learning tutorial.

This module is fully self-contained within `tutorial/handson_cl/`: it does not
import `data.py` (the repo's full dataset module imports MinkowskiEngine at the
top level, a heavy CUDA-oriented dependency we don't need here) nor the repo's
top-level `utils` package - `normalize_pc` is vendored locally in `pc_utils.py`.

The one-time `build_manifest()` step is the only thing that reaches outside this
folder: it reads the full ~52k-object ShapeNet split from `data/meta_data/...`
and copies the ~110 selected objects' raw `.npy` files into `dataset/` right here.
After that, every other function in this module (and the notebook) only ever
touches files inside `tutorial/handson_cl/`.
"""
import json
import os
import random
import shutil

import numpy as np
import torch

from .pc_utils import normalize_pc

HANDSON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(HANDSON_DIR, "dataset")

REPO_ROOT = os.path.abspath(os.path.join(HANDSON_DIR, "..", ".."))
SPLIT_PATH = os.path.join(REPO_ROOT, "data/meta_data/split/ablation/train_shapenet_only.json")


def _resolve_source_path(raw_path: str) -> str:
    """Rewrite the training-machine's absolute path to one relative to this repo.
    Only used by `build_manifest()` to locate the original source `.npy` files."""
    rel = raw_path.replace("/mnt/data/", "data/").replace("/mnt/data", "data")
    if os.path.isabs(rel):
        return rel
    return os.path.join(REPO_ROOT, rel)


def build_manifest(out_path: str, split_path: str = SPLIT_PATH, n_per_synset: int = 2, seed: int = 0):
    """Select ~100 ShapeNet objects (n_per_synset per synset), copy their raw
    `.npy` files into `tutorial/handson_cl/dataset/<synset>/<uid>.npy` so this
    folder is self-contained (no dependency on the full training data mount),
    and cache their lightweight metadata to `out_path`.
    """
    with open(split_path, "r") as f:
        split = json.load(f)

    by_synset = {}
    for entry in split:
        by_synset.setdefault(entry["group"], []).append(entry)

    rng = random.Random(seed)
    manifest = []
    for synset, entries in sorted(by_synset.items()):
        shuffled = entries[:]
        rng.shuffle(shuffled)
        picked = 0
        for entry in shuffled:
            if picked >= n_per_synset:
                break
            src_path = _resolve_source_path(entry["data_path"])
            if not os.path.isfile(src_path):
                continue
            try:
                data = np.load(src_path, allow_pickle=True).item()
                # data["text"][0] is a WordNet-style comma-joined synonym string
                # (e.g. "ashcan,trash can,garbage can,..."); keep only the first,
                # most common synonym as a short, display-friendly category label.
                raw_label = str(data["text"][0]) if len(data["text"]) > 0 else synset
                category = raw_label.split(",")[0].strip() or synset
                caption = str(data.get("blip_caption", ""))
            except Exception:
                continue

            dest_dir = os.path.join(DATASET_DIR, synset)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, os.path.basename(src_path))
            if not os.path.isfile(dest_path):
                shutil.copy2(src_path, dest_path)

            manifest.append({
                "uid": entry["id"],
                "synset": synset,
                "category": category,
                # relative to tutorial/handson_cl/ (this folder), NOT the repo root -
                # keeps the manifest usable even if this folder is copied elsewhere.
                "data_path": os.path.relpath(dest_path, HANDSON_DIR),
                "caption": caption,
            })
            picked += 1

    manifest.sort(key=lambda r: (r["category"], r["uid"]))
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_manifest(path: str, **build_kwargs):
    """Load the cached manifest, building it first if it doesn't exist yet."""
    if not os.path.isfile(path):
        return build_manifest(path, **build_kwargs)
    with open(path, "r") as f:
        return json.load(f)


def load_object(record: dict, num_points: int = 1024, y_up: bool = True, seed=None):
    """Load one object's point cloud + frozen CLIP embeddings, preprocessed the
    same way as the real `Four.get_data` (minus training-only augmentation).

    Returns a dict with:
      xyz, rgb        : (num_points, 3) float32 arrays (xyz normalized to the unit sphere)
      features        : (num_points, 6) float32 array = concat([xyz, rgb])
      img_feat        : (1280,) float32, L2-normalized average of the 12 precomputed CLIP image views
      text_feat       : (1280,) float32, precomputed CLIP embedding of `blip_caption`
      caption/category: strings
    """
    data_path = os.path.join(HANDSON_DIR, record["data_path"])
    data = np.load(data_path, allow_pickle=True).item()

    rng = np.random.default_rng(seed)
    n = data["xyz"].shape[0]
    idx = rng.choice(n, size=min(num_points, n), replace=False)

    xyz = data["xyz"][idx].astype(np.float32)
    rgb = data["rgb"][idx].astype(np.float32)

    if y_up:
        xyz = xyz.copy()
        xyz[:, [1, 2]] = xyz[:, [2, 1]]
    xyz = normalize_pc(xyz).astype(np.float32)

    features = np.concatenate([xyz, rgb], axis=1).astype(np.float32)

    img_feat = data["image_feat"].astype(np.float32)          # (12, 1280)
    img_feat = img_feat.mean(axis=0)
    img_feat = img_feat / (np.linalg.norm(img_feat) + 1e-8)

    text_feat = np.asarray(data["blip_caption_feat"]["original"], dtype=np.float32).reshape(-1)
    text_feat = text_feat / (np.linalg.norm(text_feat) + 1e-8)

    return {
        "xyz": xyz,
        "rgb": rgb,
        "features": features,
        "img_feat": img_feat,
        "text_feat": text_feat,
        "caption": str(data.get("blip_caption", "")),
        "category": record["category"],
        "uid": record["uid"],
    }


def make_batch(records, num_points: int = 1024, seed=None):
    """Stack a list of manifest records into batched torch tensors."""
    objs = [load_object(r, num_points=num_points, seed=None if seed is None else seed + i)
            for i, r in enumerate(records)]
    xyz = torch.from_numpy(np.stack([o["xyz"] for o in objs]))
    features = torch.from_numpy(np.stack([o["features"] for o in objs]))
    img_feat = torch.from_numpy(np.stack([o["img_feat"] for o in objs]))
    text_feat = torch.from_numpy(np.stack([o["text_feat"] for o in objs]))
    categories = [o["category"] for o in objs]
    captions = [o["caption"] for o in objs]
    return {
        "xyz": xyz, "features": features,
        "img_feat": img_feat, "text_feat": text_feat,
        "categories": categories, "captions": captions,
    }


def iter_batches(manifest, batch_size: int, num_points: int = 1024, seed: int = 0):
    """Infinite generator yielding shuffled mini-batches (as dicts of tensors)
    over the manifest, reshuffling every epoch."""
    rng = random.Random(seed)
    order = list(range(len(manifest)))
    step = 0
    while True:
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            chunk = [manifest[i] for i in order[start:start + batch_size]]
            if len(chunk) < 2:
                continue
            yield make_batch(chunk, num_points=num_points, seed=seed * 10_000 + step)
            step += 1
