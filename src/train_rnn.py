"""Train the GRU + GloVe caption-image retrieval model.

Usage
-----
    python3 src/train_rnn.py \
        --data-dir data/coco \
        --glove /home/ubuntu/work/glove.6B.300d.txt \
        --num-train-images 5000 \
        --epochs 10

Training pairs each image with one of its five captions per epoch (chosen
randomly). Negatives are the other examples in the batch, per the margin
ranking objective.
"""

from __future__ import annotations

import argparse
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
    p.add_argument("--num-train-images", type=int, default=5000)
    p.add_argument("--num-val-images", type=int, default=5000)
    p.add_argument("--embed-dim", type=int, default=1024)
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--max-len", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="rnn_checkpoint.pt")
    return p.parse_args()


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
) -> dict:
    """Rank all val images for each val caption; report R@K and mean rank."""
    img_enc.eval()
    cap_enc.eval()
    with torch.no_grad():
        vf = torch.from_numpy(val_feats)
        img_emb = img_enc(vf)                                   # (V, D)

        num_caps = len(val_caps)
        cap_emb = torch.empty(num_caps, img_emb.shape[1])
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
            cap_emb[start:stop] = cap_enc(ids_t, lens_t)

        # squared distance caption -> image
        a2 = (img_emb * img_emb).sum(dim=1, keepdim=True)       # (V, 1)
        b2 = (cap_emb * cap_emb).sum(dim=1).unsqueeze(0)        # (1, C)
        d2 = a2 - 2.0 * img_emb @ cap_emb.t() + b2              # (V, C)
        order = torch.argsort(d2, dim=0)                        # per caption
        image_index = torch.arange(num_caps) // data_mod.CAPTIONS_PER_IMAGE
        pos = (order == image_index.unsqueeze(0)).nonzero()
        ranks = torch.zeros(num_caps, dtype=torch.long)
        ranks[pos[:, 1]] = pos[:, 0] + 1

    out = {}
    for k in (1, 5, 10):
        out[f"R@{k}"] = float((ranks <= k).float().mean() * 100.0)
    out["mean_rank"] = float(ranks.float().mean())
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

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
    img_enc = models.ImageEncoder(4096, args.embed_dim)
    cap_enc = models.CaptionEncoder(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        glove_matrix=torch.from_numpy(glove_mat),
    )
    params = list(img_enc.parameters()) + list(cap_enc.parameters())
    optim = torch.optim.Adam(params, lr=args.lr)

    # (n_images, 5) view of caption rows; each image owns 5 consecutive rows.
    cap_ids = cap_ids.reshape(n_train, data_mod.CAPTIONS_PER_IMAGE, args.max_len)
    cap_lens = cap_lens.reshape(n_train, data_mod.CAPTIONS_PER_IMAGE)
    feats_t = torch.from_numpy(train_feats)

    print("[6/6] training ...")
    best = -1.0
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
            ids = torch.from_numpy(cap_ids[chunk, which])          # (b, L)
            lens = torch.from_numpy(cap_lens[chunk, which])        # (b,)
            imgs = feats_t[torch.from_numpy(chunk)]                # (b, 4096)

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
            img_enc, cap_enc, val_feats, val_caps, vocab, args.max_len
        )
        print(
            f"  epoch {epoch:2d} | loss {total / max(nb, 1):8.2f} | "
            f"R@1 {metrics['R@1']:5.1f} | R@5 {metrics['R@5']:5.1f} | "
            f"R@10 {metrics['R@10']:5.1f} | mr {metrics['mean_rank']:6.1f} "
            f"| {time.time() - t0:5.1f}s"
        )
        if metrics["R@10"] > best:
            best = metrics["R@10"]
            torch.save(
                {
                    "img_enc": img_enc.state_dict(),
                    "cap_enc": cap_enc.state_dict(),
                    "vocab": vocab,
                    "args": vars(args),
                    "metrics": metrics,
                },
                args.out,
            )
    print(f"done. best R@10 = {best:.1f}%  (checkpoint: {args.out})")


if __name__ == "__main__":
    main()
