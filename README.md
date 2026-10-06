# Hands-on: Contrastive Learning for 3D Vision-Language Models

A hands-on tutorial notebook implementing, end to end, a 3D Vision-Language
contrastive-learning pipeline: point cloud -> 3D encoder -> normalize -> joint
embedding space with frozen OpenCLIP image/text embeddings -> similarity
matrix -> symmetric InfoNCE loss -> a training loop that updates only the 3D
encoder.

## Contents

- `3D_VLM_Contrastive_Learning.ipynb` - the tutorial notebook, 14 sections. Its
  first code cell (`!pip install ...`) installs every non-stdlib, non-`torch`
  dependency, so a fresh environment just needs `torch` already present.
- `handson_utils/` - small helper package used by the notebook. Every file here
  is fully self-contained inside this folder - nothing imports the outer
  `3DMRL` repo (the `data/`, `models/`, or `utils/` packages) at runtime:
  - `data_loading.py` - selects/caches the 110-object ShapeNet subset and loads
    point clouds + precomputed OpenCLIP embeddings per object. All paths it reads
    at runtime (`subset_manifest.json`, `dataset/...`) are relative to this
    folder, not the repo root. (Its one-time `build_manifest()` bootstrap step is
    the single exception - see below.)
  - `pc_utils.py` - `normalize_pc`, vendored from the repo's `utils/data.py`.
  - `point_bert.py` / `pointnet_util.py` - the Point-BERT 3D encoder
    (arXiv:2111.14819) and its PointNet++ patchify layer, vendored from
    `models/ppat.py` / `models/pointnet_util.py`. Needs `torch_redstone`,
    `einops`. Pure PyTorch otherwise - farthest-point sampling is a plain
    PyTorch loop rather than a `dgl` call, to avoid that package's large,
    version-fragile dependency chain.
  - `logit_scale.py` - `LogitScaleNetwork`, vendored from
    `models/LogitScaleNetwork.py`.
  - `model_setup.py` - thin `build_pointbert()`/`build_logit_scale()` wrappers
    around the above (`scaling=1`, the smallest of six size presets - ~5M
    params, fast enough to train live on CPU).
  - `viz.py` - Plotly/matplotlib helpers for point clouds, joint-embedding PCA
    scatter plots, similarity-matrix heatmaps, and per-category zero-shot
    classification heatmaps.
- `subset_manifest.json` - cached metadata (uid, synset, category, path, caption)
  for the 110 selected ShapeNet objects, 2 per category (4 for one) across 54 categories,
  drawn from `data/meta_data/split/ablation/train_shapenet_only.json`.
- `dataset/<synset>/<uid>.npy` - the actual raw data for those 110 objects
  (point cloud + precomputed OpenCLIP embeddings + captions), copied here so this
  folder is fully self-contained (~33MB total) and doesn't depend on the full
  `data/objaverse-processed/...` mount being present. Delete `subset_manifest.json`
  (and, if you want a truly fresh copy, `dataset/`) to rebuild from scratch - the
  selection is still seeded (seed=0 by default), so re-running reproduces the same
  110 objects.

## Running it

This folder is self-contained and portable - it can be copied to a different
machine on its own (no need for the rest of the `3DMRL` repo) as long as a
Jupyter frontend and `torch` are available there:

```bash
cd tutorial/handson_cl
jupyter lab 3D_VLM_Contrastive_Learning.ipynb   # or: jupyter notebook ...
```

Run the notebook's first code cell once per environment to install everything
else it needs (`torch_redstone`, `einops`, `plotly`, `ipywidgets`,
`scikit-learn`, `matplotlib`) - no OpenCLIP and no MinkowskiEngine required, no
GPU required either. Everything runs on CPU in well under a minute (the
Section 10 training loop is ~60 optimizer steps over 110 objects, ~15s
measured on this machine).

## What's simplified, on purpose

- No live OpenCLIP calls: image/text embeddings are read from each object's
  precomputed `image_feat`/`blip_caption_feat` fields - exactly how the real
  production trainer (`trainers/trainer_3dmrl.py`) uses them too.
- No raw rendered photos exist for these objects in this environment, so the
  "Image" modality is shown as a labeled point-cloud pseudo-render proxy; the
  actual `z_img` used in every computation is still the real precomputed CLIP
  embedding, never the proxy image.
- A single flat 1280-dim embedding, to keep the story simple, rather than the
  production Matryoshka/MRL nested-dimension setup (`trainers/MRL.py`).
- Section 12's zero-shot classification uses category "prototypes" built by
  averaging each category's real, precomputed caption embeddings, rather than
  encoding hand-written prompts (e.g. `"a point cloud of a chair"`) with a live
  text encoder - consistent with the "no live OpenCLIP calls" simplification above.

No hard-negative mining, no learning-rate schedule, no data augmentation, and no
multi-GPU/distributed training either - see the real production trainer
(`trainers/trainer_3dmrl.py`) for what large-scale training adds on top of this toy
demo.
