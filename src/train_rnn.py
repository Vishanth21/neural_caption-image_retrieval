"""Train the GRU + GloVe caption-image retrieval model.

Usage
-----
    python3 src/train_rnn.py \
        --data-dir data/coco \
        --glove /home/ubuntu/work/glove.6B.300d.txt \
        --num-train-images 5000 \
        --epochs 30

Pass ``--num-train-images 0`` to train on the full 113,287-image split.

Training pairs each image with one of its five captions per epoch (chosen
randomly). Negatives are the other examples in the batch, per the margin
ranking objective.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import baseline  # noqa: E402
import data as data_mod  # noqa: E402
import glove as glove_mod  # noqa: E402
import models  # noqa: E402
import vocab as vocab_mod  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GRU caption encoder")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--glove", required=True)
    p.add_argument("--num-train-images", type=int, default=5000,
                   help="subset of training images (0 = full 113,287 split)")
    p.add_argument("--num-val-images", type=int, default=5000,
                   help="subset of validation images (0 = all 5,000)")
    p.add_argument("--embed-dim", type=int, default=1024)
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--max-len", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="rnn_checkpoint.pt")
    p.add_argument("--metrics-out", default="",
                   help="optional JSON file for per-epoch and test metrics")
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda | mps (auto picks cuda, then mps, then cpu)")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_batches(n: int, batch_size: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        chunk = idx[start : start + batch_size]
        if len(chunk) < 2:
            continue
        yield chunk


def evaluate(
    img_enc: models.ImageEncoder,
    cap_enc: models.CaptionEncoder,
    val_feats: np.ndarray,
    val_caps: list,
    vocab: dict,
    max_len: int,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Rank all val images for each val caption; report R@K and mean rank."""
    img_enc.eval()
    cap_enc.eval()
    with torch.no_grad():
        vf = torch.from_numpy(val_feats).to(device)
        img_emb = img_enc(vf)                                   # (V, D)

        num_caps = len(val_caps)
        cap_emb = torch.empty(num_caps, img_emb.shape[1], device=device)
        for start in range(0, num_caps, batch_size):
            stop = min(start + batch_size, num_caps)
            ids = []
            lens = []
            for c in val_caps[start:stop]:
                i, l = vocab_mod.encode(c, vocab, max_len)
                ids.append(i)
                lens.append(l)
            ids_t = torch.from_numpy(np.stack(ids))
            lens_t = torch.from_numpy(np.asarray(lens, dtype=np.int64))
            cap_emb[start:stop] = cap_enc(ids_t.to(device), lens_t)

        # squared distance caption -> image. A caption's rank is 1 + the
        # number of images strictly closer than its own (no full sort needed).
        a2 = (img_emb * img_emb).sum(dim=1, keepdim=True)       # (V, 1)
        b2 = (cap_emb * cap_emb).sum(dim=1).unsqueeze(0)        # (1, C)
        d2 = a2 - 2.0 * img_emb @ cap_emb.t() + b2              # (V, C)
        cols = torch.arange(num_caps, device=device)
        image_index = cols // data_mod.CAPTIONS_PER_IMAGE
        true_d2 = d2[image_index, cols]
        ranks = ((d2 < true_d2.unsqueeze(0)).sum(dim=0) + 1).cpu()

    out = {}
    for k in (1, 5, 10):
        out[f"R@{k}"] = float((ranks <= k).float().mean() * 100.0)
    out["mean_rank"] = float(ranks.float().mean())
    return out


def evaluate_test(img_enc, cap_enc, data_dir, vocab, max_len, device) -> dict:
    """Score a model on the test split: the full pool, plus the 1k-image
    protocol (five disjoint 1,000-image folds, averaged) used by the
    reference work."""
    feats, caps = data_mod.load_split(data_dir, "test")
    out = {"5k": evaluate(img_enc, cap_enc, feats, caps, vocab, max_len,
                          device=device)}
    per = data_mod.CAPTIONS_PER_IMAGE
    folds = [
        evaluate(img_enc, cap_enc, feats[i:i + 1000],
                 caps[i * per:(i + 1000) * per], vocab, max_len, device=device)
        for i in range(0, feats.shape[0] - 999, 1000)
    ]
    out["1k"] = {k: float(np.mean([f[k] for f in folds])) for k in folds[0]}
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)
    print(f"      device={device}")

    print("[1/6] loading GloVe ...")
    emb = glove_mod.load_glove(args.glove)
    glove_dim = glove_mod.embedding_dim(emb)
    print(f"      vocab={len(emb)} dim={glove_dim}")

    print("[2/6] loading features + captions ...")
    train_feats, train_caps = data_mod.load_split(args.data_dir, "train")
    val_feats, val_caps = data_mod.load_split(args.data_dir, "val")

    n_train = args.num_train_images or train_feats.shape[0]
    n_val = args.num_val_images or val_feats.shape[0]
    train_feats = train_feats[:n_train]
    val_feats = val_feats[:n_val]
    train_caps = train_caps[: n_train * data_mod.CAPTIONS_PER_IMAGE]
    val_caps = val_caps[: n_val * data_mod.CAPTIONS_PER_IMAGE]
    print(f"      train images={train_feats.shape[0]} val images={val_feats.shape[0]}")

    print("[3/6] building vocab + embedding matrix ...")
    vocab = vocab_mod.build_vocab(train_caps)
    glove_mat = vocab_mod.embedding_matrix(
        vocab, emb, glove_dim, seed=args.seed
    )
    print(f"      vocab size={len(vocab)}")

    print("[4/6] encoding captions ...")
    # Pre-encode every caption once: (num_caps, max_len) ids and lengths.
    cap_ids = np.zeros((len(train_caps), args.max_len), dtype=np.int64)
    cap_lens = np.zeros(len(train_caps), dtype=np.int64)
    for i, c in enumerate(train_caps):
        ids, length = vocab_mod.encode(c, vocab, args.max_len)
        cap_ids[i] = ids
        cap_lens[i] = length

    print("[5/6] building model ...")
    img_enc = models.ImageEncoder(4096, args.embed_dim).to(device)
    cap_enc = models.CaptionEncoder(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        glove_matrix=torch.from_numpy(glove_mat),
    ).to(device)
    params = list(img_enc.parameters()) + list(cap_enc.parameters())
    optim = torch.optim.Adam(params, lr=args.lr)

    # (n_images, 5) view of caption rows; each image owns 5 consecutive rows.
    cap_ids = cap_ids.reshape(n_train, data_mod.CAPTIONS_PER_IMAGE, args.max_len)
    cap_lens = cap_lens.reshape(n_train, data_mod.CAPTIONS_PER_IMAGE)
    feats_t = torch.from_numpy(train_feats)

    print("[6/6] training ...")
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        img_enc.train()
        cap_enc.train()
        t0 = time.time()
        total = 0.0
        nb = 0
        for chunk in make_batches(n_train, args.batch_size, rng):
            b = len(chunk)
            # one random caption per image in the batch
            which = rng.integers(0, data_mod.CAPTIONS_PER_IMAGE, size=b)
            ids = torch.from_numpy(cap_ids[chunk, which]).to(device)  # (b, L)
            lens = torch.from_numpy(cap_lens[chunk, which])           # (b,) cpu
            imgs = feats_t[torch.from_numpy(chunk)].to(device)        # (b, 4096)

            img_emb = img_enc(imgs)
            cap_emb = cap_enc(ids, lens)
            sim = models.similarity(img_emb, cap_emb)
            loss = models.margin_loss(sim, args.margin)

            optim.zero_grad()
            loss.backward()
            optim.step()
            total += float(loss.detach())
            nb += 1

        metrics = evaluate(
            img_enc, cap_enc, val_feats, val_caps, vocab, args.max_len,
            device=device,
        )
        print(
            f"  epoch {epoch:2d} | loss {total / max(nb, 1):8.2f} | "
            f"R@1 {metrics['R@1']:5.1f} | R@5 {metrics['R@5']:5.1f} | "
            f"R@10 {metrics['R@10']:5.1f} | mr {metrics['mean_rank']:6.1f} "
            f"| {time.time() - t0:5.1f}s"
        )
        history.append({"epoch": epoch, "loss": total / max(nb, 1), **metrics})
        if metrics["R@10"] > best:
            best = metrics["R@10"]
            torch.save(
                {
                    "img_enc": {k: v.cpu() for k, v in img_enc.state_dict().items()},
                    "cap_enc": {k: v.cpu() for k, v in cap_enc.state_dict().items()},
                    "vocab": vocab,
                    "args": vars(args),
                    "metrics": metrics,
                },
                args.out,
            )
    print(f"done. best val R@10 = {best:.1f}%  (checkpoint: {args.out})")

    test = None
    if os.path.exists(data_mod.feature_path(args.data_dir, "test")):
        ckpt = torch.load(args.out, map_location="cpu")
        img_enc.load_state_dict(ckpt["img_enc"])
        cap_enc.load_state_dict(ckpt["cap_enc"])
        test = evaluate_test(img_enc, cap_enc, args.data_dir, vocab,
                             args.max_len, device)
        for pool, m in test.items():
            print(f"  test {pool}: R@1 {m['R@1']:5.1f} | R@5 {m['R@5']:5.1f} | "
                  f"R@10 {m['R@10']:5.1f} | mr {m['mean_rank']:6.1f}")
    if args.metrics_out:
        with open(args.metrics_out, "w") as f:
            json.dump({"args": vars(args), "history": history,
                       "best_val_R@10": best, "test": test}, f, indent=2)


if __name__ == "__main__":
    main()
