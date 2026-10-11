"""Neural caption-image retrieval models (Vendrov et al., 2015).

Two encoders map images and captions into one shared ``embed_dim`` space:

    f_i(i) = W_i @ f_VGG(i)                       # linear projection
    f_c(c) = GRU(emb(c))[-1] -> linear            # recurrent encoder

Similarity is negative squared Euclidean distance and training uses the
margin ranking loss (eq. 2 in the reference report), with negatives drawn
from the other examples in the same batch.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """Linear projection of precomputed VGG-19 fc7 features."""

    def __init__(self, in_dim: int = 4096, embed_dim: int = 1024):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.proj(feats)


class CaptionEncoder(nn.Module):
    """GRU over word embeddings, taking the final hidden state."""

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 1024,
        glove_matrix: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
        vocab_size: int | None = None,
        glove_dim: int = 300,
    ):
        super().__init__()
        if glove_matrix is not None:
            vocab_size, glove_dim = glove_matrix.shape
        if vocab_size is None:
            raise ValueError(
                "CaptionEncoder needs either glove_matrix or vocab_size"
            )
        self.embedding = nn.Embedding(vocab_size, glove_dim, padding_idx=0)
        if glove_matrix is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(glove_matrix)
        self.embedding.weight.requires_grad = not freeze_embeddings

        self.gru = nn.GRU(
            glove_dim, hidden_dim, batch_first=True, bidirectional=False
        )
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(
        self, ids: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        emb = self.embedding(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = h[-1]                       # (batch, hidden_dim)
        return self.proj(h)


def similarity(img_emb: torch.Tensor, cap_emb: torch.Tensor) -> torch.Tensor:
    """Pairwise negative squared distance, shape ``(num_images, num_captions)``.

    Embeddings are L2-normalized first so the space is bounded and the
    margin in the ranking loss is meaningful.
    """
    img_emb = F.normalize(img_emb, dim=1)
    cap_emb = F.normalize(cap_emb, dim=1)
    # ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2  (here = 2 - 2 cos)
    a2 = (img_emb * img_emb).sum(dim=1, keepdim=True)      # (I, 1)
    b2 = (cap_emb * cap_emb).sum(dim=1).unsqueeze(0)       # (1, C)
    cross = img_emb @ cap_emb.t()                          # (I, C)
    return -(a2 - 2.0 * cross + b2)


def margin_loss(
    sim: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """Margin ranking loss over the batch similarity matrix.

    ``sim[i, j]`` is the similarity of image ``i`` and caption ``j``; the
diagonal holds the positive pairs. Two terms penalize (a) captions matched
to the wrong image and (b) images matched to the wrong caption.
    """
    n = sim.shape[0]
    diag = sim.diagonal().unsqueeze(1)                     # (n, 1)
    cost_c = F.relu(margin - diag + sim).sum() - margin * n  # wrong captions
    cost_i = F.relu(margin - diag + sim.t()).sum() - margin * n
    return cost_c + cost_i
