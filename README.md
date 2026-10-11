# Caption-Image Retrieval (UE24CS352A Mini-Project)

Content-based image retrieval: given a text query (caption), retrieve the most
relevant images from a collection. Trained on the Microsoft COCO dataset with
the Karpathy & Fei-Fei (2015) split and Vendrov et al. (2015) 10-crop VGG-19
features.

## Problem Statement

Input: a collection of images + a query sentence describing a desired image.
Output: a ranked list of the top-K images most relevant to the query.

Approach: learn a shared embedding space where captions and images are aligned,
so that nearest-neighbour search returns the correct image.

## Team

- Vishanth - PES1UG24AM323
- Kshitij Gupta - PES1UG24AM915

## Dataset

Microsoft COCO (Lin et al., 2014), using the Karpathy & Fei-Fei (2015) split.

- 123,287 images total
- 113,287 train / 5,000 validation / 5,000 test
- 5 human-annotated captions per image

Image features are the precomputed 10-crop VGG-19 `fc7` vectors shipped with
the Vendrov et al. release. Raw images are **not** needed to train or evaluate;
they are only required to display retrieved pictures. See
[`data/README.md`](data/README.md) for download links and the on-disk layout.

## Models

1. **Baseline** - multi-response linear regression aligning mean-pooled GloVe
   caption vectors with VGG-19 `fc7` image features (closed-form least squares).
2. **GRU + GloVe** - a GRU caption encoder initialized from GloVe and a linear
   image projection into a shared space, trained with a margin ranking loss.

Evaluation metric: Recall@K (R@K), the fraction of queries whose correct image
is within the top K retrieved images.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   └── README.md         # dataset sources + on-disk layout
├── docs/                 # course guidelines + reference material
├── deliverables/         # final submission artifacts (write-up, slides)
├── notebooks/            # exploration notebooks
└── src/
    ├── data.py           # load a split's features + captions
    ├── glove.py          # GloVe loading and caption featurization
    ├── baseline.py       # least-squares W and Recall@K
    ├── run_baseline.py   # baseline end-to-end
    ├── vocab.py          # word vocab + GloVe embedding matrix
    ├── models.py         # image encoder, GRU caption encoder, margin loss
    ├── train_rnn.py      # GRU training + evaluation
    ├── retrieve.py       # load a trained model, retrieve top-K images for a caption
    ├── demo.py           # interactive text-to-image retrieval demo
    └── README.md         # module-level notes
```

Data files (the `.npy` features, `.txt` captions, GloVe) are **not** committed;
see [`data/README.md`](data/README.md).

## Setup

Python 3.10+ is assumed.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch is installed from the CPU wheel index (there is no GPU requirement for
the 5k-image subset):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Data

Download the Vendrov 10-crop features and GloVe as described in
[`data/README.md`](data/README.md). The commands below assume:

- features + captions at `data/coco/`
- GloVe at `~/work/glove.6B.300d.txt`

## Running

### Baseline

```bash
python3 src/run_baseline.py \
    --data-dir data/coco \
    --glove ~/work/glove.6B.300d.txt \
    --num-train-images 5000
```

Pass `--num-train-images 0` to use the full 113,287-image training split.

### GRU + GloVe

```bash
python3 src/train_rnn.py \
    --data-dir data/coco \
    --glove ~/work/glove.6B.300d.txt \
    --num-train-images 5000 \
    --epochs 30
```

The best-R@10 checkpoint is written to `rnn_checkpoint.pt`. Useful flags:
`--embed-dim`, `--hidden-dim`, `--batch-size`, `--lr`, `--margin`.

Both scripts print Recall@1/5/10 and mean rank on the validation split.

### Demo (interactive retrieval)

With a trained checkpoint and the val images extracted:

```bash
python3 src/demo.py \
    --data-dir data/coco \
    --images /path/to/val2014 \
    --checkpoint rnn_checkpoint.pt
```

Type a caption at the prompt and it prints the top-K retrieved images (row,
filename, caption) and writes `demo_montage.png` — a grid of the retrieved
images. The demo needs `data/coco/val_path.txt` (one COCO filename per line,
matching the row order of `val.txt` / `val.npy`) and the raw `val2014` JPEGs.

Results: our GRU + GloVe model reaches **R@10 = 28.3%** on the validation
split when trained on a 5,000-image subset, versus 16.7% for the baseline.

## Deliverables Checklist

- [x] Source code in private GitHub repo (shared with faculty/TAs)
- [x] README with setup & run instructions
- [x] One-page write-up (PDF)
- [x] Presentation slide deck
- [x] Live demonstration of working code
