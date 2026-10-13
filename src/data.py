"""Data loading for the caption-image retrieval project.

Loads the Vendrov et al. (2015) 10-crop VGG-19 features together with the
Karpathy & Fei-Fei (2015) split captions.

Layout expected on disk (see ../data/README.md):

    data/coco/
    ├── train.txt          # 566,435 lines = 5 captions x 113,287 images
    ├── val.txt            #  25,000 lines = 5 captions x   5,000 images
    ├── test.txt           #  25,000 lines
    └── images/10crop/
        ├── train.npy      # (113287, 4096) float32
        ├── val.npy        # (  5000, 4096) float32
        └── test.npy       # (  5000, 4096) float32

The caption <-> feature mapping is positional only: lines 5i..5i+4 of a
.txt file are the five captions of row i of the matching .npy array.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

CAPTIONS_PER_IMAGE = 5

# Feature folders under images/, in order of preference: the Vendrov 10-crop
# release, or the centre-crop COCO 2017 features from extract_features.py.
FEATURE_SUBDIRS = ("10crop", "1crop")


def feature_path(data_dir: str, split: str) -> str:
    """Path of a split's feature array (the first FEATURE_SUBDIRS that has it)."""
    for sub in FEATURE_SUBDIRS:
        p = os.path.join(data_dir, "images", sub, f"{split}.npy")
        if os.path.exists(p):
            return p
    return os.path.join(data_dir, "images", FEATURE_SUBDIRS[0], f"{split}.npy")


def _load_captions(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def load_split(data_dir: str, split: str) -> Tuple[np.ndarray, List[str]]:
    """Load one split's features and captions.

    Parameters
    ----------
    data_dir:
        Directory that directly contains ``train.txt``/``val.txt``/``test.txt``
        and an ``images/10crop`` (or ``images/1crop``) subdirectory.
    split:
        One of ``"train"``, ``"val"``, ``"test"``.

    Returns
    -------
    feats:
        ``(num_images, 4096)`` float32 array.
    caps:
        List of ``num_images * 5`` caption strings, ordered by image then by
        caption index.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split: {split!r}")

    feat_path = feature_path(data_dir, split)
    cap_path = os.path.join(data_dir, f"{split}.txt")

    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"missing features: {feat_path}\n"
            "See data/README.md for how to obtain the Vendrov 10-crop features."
        )
    if not os.path.exists(cap_path):
        raise FileNotFoundError(f"missing captions: {cap_path}")

    feats = np.load(feat_path).astype(np.float32, copy=False)
    caps = _load_captions(cap_path)

    expected = feats.shape[0] * CAPTIONS_PER_IMAGE
    if len(caps) != expected:
        raise ValueError(
            f"caption count mismatch for {split!r}: "
            f"got {len(caps)}, expected {expected} "
            f"({feats.shape[0]} images x {CAPTIONS_PER_IMAGE})"
        )

    return feats, caps


def captions_for_image(caps: List[str], image_index: int) -> List[str]:
    """Return the five captions belonging to ``image_index``."""
    start = image_index * CAPTIONS_PER_IMAGE
    return caps[start : start + CAPTIONS_PER_IMAGE]


def expand_features(feats: np.ndarray) -> np.ndarray:
    """Repeat each image feature ``CAPTIONS_PER_IMAGE`` times.

    ``feats`` is ``(num_images, D)``; the result is ``(num_images * 5, D)``
    where ``out[5*i + j]`` equals ``feats[i]``. This aligns the feature rows
    with the caption list so that caption ``5*i + j`` pairs with its image.
    """
    return np.repeat(feats, CAPTIONS_PER_IMAGE, axis=0)
