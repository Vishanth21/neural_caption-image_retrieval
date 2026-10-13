"""Build the COCO 2017 train/val/test splits in the layout ``data.py`` reads.

COCO 2017 releases 118,287 train and 5,000 val images; the official test
images have no public captions. We therefore use

    train : train2017 minus 5,000 held-out images   (113,287 images)
    val   : those 5,000 held-out train2017 images     (model selection)
    test  : all of val2017                            (5,000 images, reported)

which matches the sizes of the Karpathy split used by the reference work.
Images are ordered by COCO image id (the held-out set is a seeded random
sample). Each image keeps its first five captions by annotation id (a few
COCO images have six or seven), lower-cased with punctuation stripped, the
same normalization as the Vendrov caption files.

Image features are L2-normalized per image. Raw fc7 vectors from
``extract_features.py`` have norms of roughly 19-82; left unnormalized, the
GRU model memorizes the training pairs and stays at chance on held-out
images (a 3,000-image pilot: val R@10 2% raw vs 49% normalized).

Inputs: ``<data-dir>/annotations/captions_{train,val}2017.json`` and the
shards written by ``extract_features.py`` under ``<data-dir>/shards/``.
Outputs (under ``<data-dir>``)::

    {train,val,test}.txt          5 captions per image, one per line
    {train,val,test}_path.txt     COCO filename per image (for the demo)
    images/<crop>/{split}.npy     (num_images, 4096) float32, <crop> = --crop-name

Usage
-----
    python3 src/prepare_coco2017.py --data-dir data/coco2017
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as data_mod  # noqa: E402

_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize(caption: str) -> str:
    return " ".join(_NON_WORD.sub(" ", caption.lower()).split())


def load_captions(path: str) -> Dict[str, List[str]]:
    """filename -> its first five normalized captions (by annotation id)."""
    with open(path, encoding="utf-8") as f:
        js = json.load(f)
    names = {img["id"]: img["file_name"] for img in js["images"]}
    caps = defaultdict(list)
    for ann in sorted(js["annotations"], key=lambda a: a["id"]):
        caps[names[ann["image_id"]]].append(normalize(ann["caption"]))
    short = [n for n in names.values() if len(caps[n]) < data_mod.CAPTIONS_PER_IMAGE]
    if short:
        raise ValueError(f"{len(short)} images have < 5 captions, e.g. {short[:3]}")
    return {n: caps[n][: data_mod.CAPTIONS_PER_IMAGE] for n in names.values()}


def load_shards(shard_dir: str) -> Dict[str, np.ndarray]:
    """filename -> 4096-d feature, from every shard in ``shard_dir``."""
    out: Dict[str, np.ndarray] = {}
    for p in sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz"))):
        d = np.load(p)
        out.update(zip(d["names"].tolist(), d["feats"]))
    return out


def write_split(data_dir: str, split: str, names: List[str],
                caps: Dict[str, List[str]], feats: Dict[str, np.ndarray],
                crop_name: str = "1crop") -> None:
    missing = [n for n in names if n not in feats]
    if missing:
        raise ValueError(f"{split}: {len(missing)} images lack features "
                         f"(extraction incomplete?), e.g. {missing[:3]}")
    with open(os.path.join(data_dir, f"{split}.txt"), "w", encoding="utf-8") as f:
        for n in names:
            f.write("\n".join(caps[n]) + "\n")
    with open(os.path.join(data_dir, f"{split}_path.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")
    feat_dir = os.path.join(data_dir, "images", crop_name)
    os.makedirs(feat_dir, exist_ok=True)
    X = np.stack([feats[n] for n in names]).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    np.save(os.path.join(feat_dir, f"{split}.npy"), X)
    print(f"      {split:5s}: {len(names)} images, {5 * len(names)} captions")


def main() -> None:
    p = argparse.ArgumentParser(description="Assemble COCO 2017 splits")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--num-val", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-partial", action="store_true",
                   help="train on only the train2017 images that have "
                        "features so far (val and test must be complete)")
    p.add_argument("--crop-name", default="1crop", choices=["1crop", "10crop"],
                   help="feature folder; match extract_features.py --crops")
    args = p.parse_args()
    ann = os.path.join(args.data_dir, "annotations")
    shards = os.path.join(args.data_dir, "shards")

    print("[1/3] loading captions ...")
    train_caps = load_captions(os.path.join(ann, "captions_train2017.json"))
    test_caps = load_captions(os.path.join(ann, "captions_val2017.json"))

    print("[2/3] loading feature shards ...")
    train_feats = load_shards(os.path.join(shards, "train2017"))
    test_feats = load_shards(os.path.join(shards, "val2017"))

    print("[3/3] writing splits ...")
    pool = sorted(train_caps)                     # filenames sort by image id
    rng = np.random.default_rng(args.seed)
    held = set(rng.choice(len(pool), size=args.num_val, replace=False).tolist())
    train = [n for i, n in enumerate(pool) if i not in held]
    val = [n for i, n in enumerate(pool) if i in held]
    if args.allow_partial:
        n_all = len(train)
        train = [n for n in train if n in train_feats]
        print(f"      --allow-partial: {len(train)}/{n_all} train images have features")
        # held-out val images need features too; top the set back up from
        # the train images that have them, so val stays at --num-val images
        missing = [n for n in val if n not in train_feats]
        if missing:
            val = [n for n in val if n in train_feats]
            extra, train = train[: len(missing)], train[len(missing):]
            val = sorted(val + extra)
            print(f"      refilled {len(missing)} val images from train")
    write_split(args.data_dir, "train", train, train_caps, train_feats,
                args.crop_name)
    write_split(args.data_dir, "val", val, train_caps, train_feats, args.crop_name)
    write_split(args.data_dir, "test", sorted(test_caps), test_caps, test_feats,
                args.crop_name)


if __name__ == "__main__":
    main()
