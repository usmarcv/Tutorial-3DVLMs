# Hands-on: Contrastive Learning for 3D Vision-Language Models

Hands-on material for the tutorial **An overview of 3D Vision-Language Models**
(SIBGRAPI 2026).

[Project page](https://usmarcv.github.io/Tutorial-3DVLMs/) |
[Paper (arXiv)](https://arxiv.org/abs/2609.05583) |
[Slides](https://usmarcv.github.io/Tutorial-3DVLMs/code.html)

The notebook implements, end to end, a 3D Vision-Language contrastive-learning
pipeline: point cloud -> 3D encoder -> normalize -> joint embedding space with
frozen OpenCLIP image/text embeddings -> similarity matrix -> symmetric InfoNCE
loss -> a training loop that updates only the 3D encoder.

## Contents

- `handson_3DVLM_contrastive.ipynb` - the tutorial notebook, 11 sections:
  setup, exploring ShapeNet, interactive point-cloud visualization, the three
  modalities, feature extraction, normalization, the joint embedding space, the
  similarity matrix, the contrastive loss, the training loop, and a
  before/after comparison.
- `handson_utils/` - small helper package used by the notebook. It is
  self-contained: nothing imports code from outside this repository.
  - `data_loading.py` - loads the cached 110-object ShapeNet subset: point
    clouds plus precomputed OpenCLIP embeddings per object.
  - `pc_utils.py` - `normalize_pc`.
  - `point_bert.py` / `pointnet_util.py` - the Point-BERT 3D encoder
    (arXiv:2111.14819) and its PointNet++ patchify layer. Needs
    `torch_redstone` and `einops`; pure PyTorch otherwise (farthest-point
    sampling is a plain PyTorch loop, not a `dgl` call).
  - `logit_scale.py` - `LogitScaleNetwork`, the learnable CLIP-style
    temperature.
  - `model_setup.py` - thin `build_pointbert()` / `build_logit_scale()`
    wrappers (`scaling=1`, the smallest of six size presets: about 5M
    parameters, fast enough to train live on CPU).
  - `viz.py` - Plotly/matplotlib helpers for point clouds, joint-embedding PCA
    scatter plots, and similarity-matrix heatmaps.
- `subset_manifest.json` - metadata (uid, synset, category, path, caption) for
  the 110 selected ShapeNet objects: 2 per category (4 for one) across 54
  categories.
- `dataset/<synset>/<uid>.npy` - the data for those 110 objects (point cloud,
  precomputed OpenCLIP embeddings, captions), about 34 MB in total.

The subset was selected once (seed 0) from the ShapeNet training split used by
the original 3DMRL research code, which is not part of this repository. The
manifest and data are shipped as-is and cannot be rebuilt from here, so do not
delete `subset_manifest.json` or `dataset/`.

## Running it

You need Python with PyTorch already installed (see
[pytorch.org/get-started](https://pytorch.org/get-started/locally/)).

```bash
git clone https://github.com/usmarcv/Tutorial-3DVLMs.git
cd Tutorial-3DVLMs
pip install torch_redstone einops plotly ipywidgets scikit-learn matplotlib nbformat
jupyter lab handson_3DVLM_contrastive.ipynb   # or: jupyter notebook ...
```

The notebook's first code cell runs the same `pip install`, so it is enough to
open the notebook and run all cells.

On Google Colab, the notebook needs the repository files next to it. Run this
in a cell before the notebook's own setup cell:

```
!git clone https://github.com/usmarcv/Tutorial-3DVLMs.git
%cd Tutorial-3DVLMs
```

No OpenCLIP, no MinkowskiEngine and no GPU are required. The whole notebook
runs on CPU; the training loop in Section 10 is 60 optimizer steps over the
110 objects and takes roughly half a minute on a multi-core CPU.

## What's simplified, on purpose

- No live OpenCLIP calls: image and text embeddings are read from each
  object's precomputed `image_feat` / `blip_caption_feat` fields, the same way
  the full-scale trainer uses them.
- No rendered photos are shipped for these objects, so the "Image" modality is
  shown as a labeled point-cloud pseudo-render proxy. The `z_img` used in every
  computation is still the real precomputed CLIP embedding, never the proxy
  image.
- A single flat 1280-dim embedding, to keep the story simple, rather than a
  Matryoshka (nested-dimension) setup.
- No hard-negative mining, no learning-rate schedule, no data augmentation and
  no multi-GPU training. The before/after comparison is measured on objects
  that were also trained on, so it shows that the loss fits the alignment, not
  that the encoder generalizes.

## Citation

```bibtex
@misc{lobo2026overview3dvisionlanguagemodels,
    title={An overview of 3D Vision-Language Models},
    author={Márcus Lobo and Vitor Matias and Afonso Paiva and Jeová Farias and Tiago Novello and Moacir Ponti},
    year={2026},
    eprint={2609.05583},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2609.05583},
}
```
