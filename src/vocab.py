"""Vocabulary construction and GloVe embedding-matrix initialization.

Captions are tokenized by simple whitespace splitting (as in the baseline),
lowercased. A special ``<pad>`` id 0 and ``<unk>`` id 1 are reserved. The
embedding matrix initializes each in-vocab word from its GloVe vector and
leaves out-of-vocab words at a small random value; the model fine-tunes the
whole matrix.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1


def tokenize(caption: str) -> List[str]:
    return caption.lower().split()


def build_vocab(
    captions: Sequence[str],
    min_count: int = 1,
) -> Dict[str, int]:
    """Build a word -> id mapping from the training captions.

    Ids 0 and 1 are reserved for padding and unknown words.
    """
    counts: Dict[str, int] = {}
    for cap in captions:
        for w in tokenize(cap):
            counts[w] = counts.get(w, 0) + 1

    vocab: Dict[str, int] = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for word, c in sorted(counts.items()):
        if c >= min_count and word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def encode(
    caption: str,
    vocab: Dict[str, int],
    max_len: int,
) -> Tuple[np.ndarray, int]:
    """Tokenize and map a caption to a fixed-length id array.

    Returns the id array (length ``max_len``, zero-padded) and the true
    (unpadded) length.
    """
    ids = [vocab.get(w, UNK_ID) for w in tokenize(caption)][:max_len]
    length = len(ids)
    if length < max_len:
        ids = ids + [PAD_ID] * (max_len - length)
    return np.asarray(ids, dtype=np.int64), length


def embedding_matrix(
    vocab: Dict[str, int],
    glove: Dict[str, np.ndarray],
    dim: int,
    seed: int = 0,
) -> np.ndarray:
    """Initialize an ``(len(vocab), dim)`` matrix from GloVe.

    In-vocab words are initialized from their GloVe vector. Words missing
    from GloVe (and the reserved tokens) get small random values.
    """
    rng = np.random.default_rng(seed)
    mat = rng.normal(0.0, 0.05, size=(len(vocab), dim)).astype(np.float32)
    mat[PAD_ID] = 0.0

    hits = 0
    for word, idx in vocab.items():
        if word in glove:
            vec = glove[word]
            if vec.shape[0] == dim:
                mat[idx] = vec
                hits += 1
    print(f"      embedding init: {hits}/{len(vocab)} words from GloVe")
    return mat
