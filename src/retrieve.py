"""Retrieval core: encode a caption and rank images by similarity.

Model-agnostic. Supports two backends:

- ``baseline``: mean-pooled GloVe caption vector projected through the
  least-squares matrix ``W`` (see ``baseline.py`` / ``run_baseline.py``).
- ``rnn``: the GRU caption encoder and image projection trained by
  ``train_rnn.py`` (loads ``rnn_checkpoint.pt``).

Given a query caption, returns the top-K image rows with their captions and
(if a path file is supplied) their filenames.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

import data as data_mod
import glove as glove_mod
import models
import vocab as vocab_mod


@dataclass
class Retriever:
    """Holds everything needed to answer caption -> images queries."""

    backend: str                       # "baseline" or "rnn"
    features: np.ndarray               # (num_images, 4096)
    captions: List[str]                # 5 per image
    paths: Optional[List[str]] = None  # filename per image, or None

    # baseline
    W: Optional[np.ndarray] = None
    glove: Optional[dict] = None

    # rnn
    img_enc: Optional[torch.nn.Module] = None
    cap_enc: Optional[torch.nn.Module] = None
    vocab: Optional[dict] = None
    max_len: int = 20
    image_embeddings: Optional[np.ndarray] = None

    @property
    def num_images(self) -> int:
        return self.features.shape[0]

    def encode_caption(self, caption: str) -> np.ndarray:
        """Return the query embedded in image space, shape ``(4096,)`` for the
        baseline (projected through W) or ``(embed_dim,)`` for the RNN."""
        if self.backend == "baseline":
            dim = glove_mod.embedding_dim(self.glove)
            v = glove_mod.caption_vector(caption, self.glove, dim)
            return v @ self.W

        if self.backend == "rnn":
            ids, length = vocab_mod.encode(caption, self.vocab, self.max_len)
            ids_t = torch.from_numpy(ids[None, :])
            len_t = torch.from_numpy(np.asarray([max(length, 1)], dtype=np.int64))
            with torch.no_grad():
                emb = self.cap_enc(ids_t, len_t)
            return emb[0].numpy()

        raise ValueError(f"unknown backend: {self.backend!r}")

    def image_matrix(self) -> np.ndarray:
        """Image-side matrix to rank against.

        For the baseline this is the raw VGG features; for the RNN it is the
        projected image embeddings, computed once and cached.
        """
        if self.backend == "baseline":
            return self.features
        if self.image_embeddings is None:
            with torch.no_grad():
                self.image_embeddings = (
                    self.img_enc(torch.from_numpy(self.features)).numpy()
                )
        return self.image_embeddings

    def retrieve(self, caption: str, k: int = 10) -> List[Tuple[int, str, str]]:
        """Return ``(row, caption, filename)`` for the top-K images."""
        q = self.encode_caption(caption)
        M = self.image_matrix()

        if self.backend == "rnn":
            # encoders are trained with L2-normalized embeddings; match that
            q = q / (np.linalg.norm(q) + 1e-8)
            M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)

        d2 = ((M - q) ** 2).sum(axis=1)
        order = np.argsort(d2)[:k]

        out = []
        for row in order:
            cap = data_mod.captions_for_image(self.captions, int(row))[0]
            name = self.paths[int(row)] if self.paths else ""
            out.append((int(row), cap, name))
        return out


def load_baseline(
    data_dir: str,
    glove_path: str,
    W_path: str,
    split: str = "val",
) -> Retriever:
    """Build a Retriever using the baseline ``W`` matrix."""
    feats, caps = data_mod.load_split(data_dir, split)
    emb = glove_mod.load_glove(glove_path)
    W = np.load(W_path)
    paths = _maybe_paths(data_dir, split)
    return Retriever(
        backend="baseline",
        features=feats,
        captions=caps,
        paths=paths,
        W=W,
        glove=emb,
    )


def load_rnn(
    data_dir: str,
    checkpoint: str,
    split: str = "val",
) -> Retriever:
    """Build a Retriever from a ``train_rnn.py`` checkpoint."""
    ckpt = torch.load(checkpoint, map_location="cpu")
    args = ckpt["args"]
    vocab = ckpt["vocab"]

    img_enc = models.ImageEncoder(4096, args["embed_dim"])
    img_enc.load_state_dict(ckpt["img_enc"])
    img_enc.eval()

    cap_enc = models.CaptionEncoder(
        embed_dim=args["embed_dim"],
        hidden_dim=args["hidden_dim"],
        vocab_size=len(vocab),
    )
    cap_enc.load_state_dict(ckpt["cap_enc"])
    cap_enc.eval()

    feats, caps = data_mod.load_split(data_dir, split)
    paths = _maybe_paths(data_dir, split)
    return Retriever(
        backend="rnn",
        features=feats,
        captions=caps,
        paths=paths,
        img_enc=img_enc,
        cap_enc=cap_enc,
        vocab=vocab,
        max_len=args.get("max_len", 20),
    )


def _maybe_paths(data_dir: str, split: str) -> Optional[List[str]]:
    p = os.path.join(data_dir, f"{split}_path.txt")
    if os.path.exists(p):
        with open(p) as f:
            return [line.strip() for line in f]
    return None
