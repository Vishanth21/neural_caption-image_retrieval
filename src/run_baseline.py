"""End-to-end baseline run: train on the train split, evaluate on val.

Usage
-----
    python3 src/run_baseline.py \
        --data-dir /home/ubuntu/work/CS229_project/data/coco \
        --glove /home/ubuntu/work/glove.6B.300d.txt \
        --num-train-images 5000

Omit ``--num-train-images`` (or pass 0) to use the full training split.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import baseline  # noqa: E402
import data as data_mod  # noqa: E402
import glove as glove_mod  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline multi-response regression")
    p.add_argument("--data-dir", required=True,
                   help="directory containing train.txt/val.txt and images/10crop")
    p.add_argument("--glove", required=True, help="path to glove.6B.300d.txt")
    p.add_argument("--num-train-images", type=int, default=0,
                   help="subset of training images to use (0 = all)")
    p.add_argument("--num-val-images", type=int, default=0,
                   help="subset of validation images to evaluate (0 = all)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("[1/5] loading GloVe ...")
    t0 = time.time()
    emb = glove_mod.load_glove(args.glove)
    dim = glove_mod.embedding_dim(emb)
    print(f"      vocab={len(emb)} dim={dim} ({time.time() - t0:.1f}s)")

    print("[2/5] loading features + captions ...")
    t0 = time.time()
    train_feats, train_caps = data_mod.load_split(args.data_dir, "train")
    val_feats, val_caps = data_mod.load_split(args.data_dir, "val")
    print(f"      train images={train_feats.shape[0]} "
          f"captions={len(train_caps)}")
    print(f"      val   images={val_feats.shape[0]} "
          f"captions={len(val_caps)}")
    print(f"      ({time.time() - t0:.1f}s)")

    # optional subsets (keep whole images, so keep multiples of 5 captions)
    n_train = args.num_train_images or train_feats.shape[0]
    n_val = args.num_val_images or val_feats.shape[0]
    train_feats_s = train_feats[:n_train]
    val_feats_s = val_feats[:n_val]
    train_caps_s = train_caps[: n_train * data_mod.CAPTIONS_PER_IMAGE]
    val_caps_s = val_caps[: n_val * data_mod.CAPTIONS_PER_IMAGE]

    print("[3/5] featurizing captions ...")
    t0 = time.time()
    X = glove_mod.captions_to_matrix(train_caps_s, emb, dim)
    Xv = glove_mod.captions_to_matrix(val_caps_s, emb, dim)
    print(f"      X={X.shape} Xv={Xv.shape} ({time.time() - t0:.1f}s)")

    print("[4/5] fitting W ...")
    t0 = time.time()
    Y = data_mod.expand_features(train_feats_s)
    W = baseline.fit(X, Y)
    print(f"      W={W.shape} ({time.time() - t0:.1f}s)")

    print("[5/5] evaluating Recall@K on val ...")
    t0 = time.time()
    projected = Xv @ W
    # ground-truth image row for each val caption: 5 consecutive captions
    # share one image, so index = position // 5
    image_index = np.arange(len(val_caps_s)) // data_mod.CAPTIONS_PER_IMAGE
    metrics = baseline.recall_at_k(
        projected, val_feats_s, image_index, ks=(1, 5, 10)
    )
    print(f"      ({time.time() - t0:.1f}s)")
    print()
    print("  results (baseline, mean-pooled GloVe -> VGG fc7):")
    for k in (1, 5, 10):
        print(f"    R@{k:<2d} = {metrics[f'R@{k}']:.1f}%")
    print(f"    mean rank = {metrics['mean_rank']:.1f}")


if __name__ == "__main__":
    main()
