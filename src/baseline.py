"""Baseline: multi-response linear regression (report section 4.1).

We learn a weight matrix W mapping caption (GloVe mean) vectors into the
VGG-19 image-feature space:

    W = argmin_W || X W - Y ||_2^2

where X is (num_captions, d_text) and Y is (num_captions, 4096). Retrieval
projects a query caption with W and returns the nearest image features.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def fit(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Least-squares solution of ``X @ W ~= Y``.

    Parameters
    ----------
    X:
        ``(num_captions, d_text)`` caption features.
    Y:
        ``(num_captions, d_image)`` image features aligned row-for-row with X.

    Returns
    -------
    W:
        ``(d_text, d_image)`` weight matrix.
    """
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return W.astype(np.float32)


def fit_grouped(
    X: np.ndarray,
    feats: np.ndarray,
    per_image: int,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Same least-squares ``W`` as ``fit(X, expand_features(feats))``, without
    materializing the expanded target.

    Every image owns ``per_image`` consecutive caption rows of ``X`` and they
    all share one target row, so ``X^T Y = (sum of each image's caption
    rows)^T @ feats``. Solving the normal equations then needs only a
    ``(d_text, d_text)`` system. On the full train split this avoids a
    566,435 x 4096 (9.3 GB) array. ``ridge`` adds a tiny ``lambda * I`` for
    numerical stability.
    """
    X64 = X.astype(np.float64)
    n_img = feats.shape[0]
    Xsum = X64.reshape(n_img, per_image, -1).sum(axis=1)   # (n_img, d_text)
    XtX = X64.T @ X64
    XtY = Xsum.T @ feats.astype(np.float64)                # (d_text, d_image)
    XtX[np.diag_indices_from(XtX)] += ridge
    return np.linalg.solve(XtX, XtY).astype(np.float32)


def recall_at_k(
    projected: np.ndarray,
    target_feats: np.ndarray,
    image_index: np.ndarray,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Compute Recall@K for projected captions.

    Parameters
    ----------
    projected:
        ``(num_captions, d_image)`` projected caption vectors.
    target_feats:
        ``(num_images, d_image)`` image features to search over.
    image_index:
        ``(num_captions,)`` ground-truth image row for each projected caption.
    ks:
        Recall cut-offs to report.

    Returns
    -------
    dict mapping ``"R@k"`` to a percentage, plus ``"mean_rank"``.
    """
    num_captions = projected.shape[0]
    max_k = max(ks)

    # Squared Euclidean distance from every caption to every image, computed
    # in chunks to avoid an (num_captions x num_images) dense blow-up.
    ranks = np.empty(num_captions, dtype=np.int64)
    chunk = 512
    for start in range(0, num_captions, chunk):
        stop = min(start + chunk, num_captions)
        p = projected[start:stop]                       # (c, D)
        # ||p - t||^2 = ||p||^2 - 2 p.t + ||t||^2
        d2 = (
            np.sum(p * p, axis=1, keepdims=True)
            - 2.0 * p @ target_feats.T
            + np.sum(target_feats * target_feats, axis=1)[None, :]
        )                                       # (c, num_images)
        order = np.argsort(d2, axis=1)          # ascending distance
        targets = image_index[start:stop][:, None]
        ranks[start:stop] = np.argmax(order == targets, axis=1)

    ranks += 1  # 1-based
    out = {f"R@{k}": float(np.mean(ranks <= k) * 100.0) for k in ks}
    out["mean_rank"] = float(np.mean(ranks))
    return out
