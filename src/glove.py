"""GloVe word-vector loading and caption featurization.

Caption representation for the baseline: the mean of the GloVe vectors of the
caption's words (Vendrov et al., 2015; section 4.1 of the reference report).
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def load_glove(path: str) -> Dict[str, np.ndarray]:
    """Load a GloVe text file into ``{word: vector}``.

    Each line has the form ``word v1 v2 ... vd``.
    """
    emb: Dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < 2:
                continue
            emb[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    return emb


def embedding_dim(emb: Dict[str, np.ndarray]) -> int:
    """Infer the vector dimension from any entry in the embedding table."""
    return int(next(iter(emb.values())).shape[0])


def caption_vector(
    caption: str,
    emb: Dict[str, np.ndarray],
    dim: int | None = None,
) -> np.ndarray:
    """Mean-pool the GloVe vectors of the words in ``caption``.

    Words absent from ``emb`` are skipped. A caption with no known words
    yields a zero vector.
    """
    if dim is None:
        dim = embedding_dim(emb)

    words = caption.lower().split()
    vecs = [emb[w] for w in words if w in emb]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


def captions_to_matrix(
    captions: Iterable[str],
    emb: Dict[str, np.ndarray],
    dim: int | None = None,
) -> np.ndarray:
    """Stack caption vectors into an ``(num_captions, dim)`` matrix."""
    if dim is None:
        dim = embedding_dim(emb)
    return np.stack([caption_vector(c, emb, dim) for c in captions])
