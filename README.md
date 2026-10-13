# Caption-Image Retrieval (UE24CS352A Mini-Project)

Content-based image retrieval: given a text query (caption), retrieve the most
relevant images from a collection. Trained on the full Microsoft COCO 2017
dataset (cocodataset.org), with VGG-19 image features we extract ourselves
from the raw JPEGs.

## Problem Statement

Input: a collection of images + a query sentence describing a desired image.
Output: a ranked list of the top-K images most relevant to the query.

Approach: learn a shared embedding space where captions and images are aligned,
so that nearest-neighbour search returns the correct image.

## Team

- Vishanth - PES1UG24AM323
- Kshitij Gupta - PES1UG24AM915

## Dataset

Microsoft COCO 2017 (Lin et al., 2014), from https://cocodataset.org/#download.
The official test captions are not public, so:

| Split | Source | Images | Captions |
|---|---|---|---|
| Train | `train2017` minus 5,000 held out | 113,287 | 566,435 |
| Validation | 5,000 held-out `train2017` images (seed 0) | 5,000 | 25,000 |
| Test | `val2017` | 5,000 | 25,000 |

Each image keeps its first five captions. Image features are VGG-19 `fc7`
vectors (4,096-d, centre crop, L2-normalised) computed by
`src/extract_features.py`, which streams the 19.3 GB `train2017.zip` with HTTP
range requests instead of downloading it. See [`data/README.md`](data/README.md).

## Models

1. **Baseline** - multi-response linear regression aligning mean-pooled GloVe
   caption vectors with VGG-19 `fc7` image features (closed-form least squares).
2. **GRU + GloVe** - a GRU caption encoder initialized from GloVe and a linear
   image projection into a shared space, trained with a margin ranking loss.

Evaluation metric: Recall@K (R@K), the fraction of queries whose correct image
is within the top K retrieved images, on the 5,000 test images (5K) and on the
standard 1K protocol (five 1,000-image folds, averaged).

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   └── README.md         # dataset sources + on-disk layout
├── docs/                 # course guidelines + reference report/poster (Qian & Lamberti, CS229)
├── deliverables/         # write-up + slides (PDF/PPTX) and their Markdown sources
│   ├── writeup.md        # source of writeup.pdf
│   ├── slides.md         # Marp source of slides.pdf / slides.pptx
│   └── build.sh          # rebuilds all three
└── src/
    ├── remote_zip.py     # read members of a remote ZIP over HTTP ranges
    ├── extract_features.py # stream COCO 2017 images -> VGG-19 fc7 shards
    ├── prepare_coco2017.py # captions + shards -> train/val/test split files
    ├── data.py           # load a split's features + captions
    ├── glove.py          # GloVe loading and caption featurization
    ├── baseline.py       # least-squares W and Recall@K
    ├── run_baseline.py   # baseline end-to-end
    ├── vocab.py          # word vocab + GloVe embedding matrix
    ├── models.py         # image encoder, GRU caption encoder, margin loss
    ├── train_rnn.py      # GRU training + evaluation
    ├── retrieve.py       # load a trained model, retrieve top-K images for a caption
    ├── demo.py           # interactive text-to-image retrieval demo
    ├── make_paths.py     # build val_path.txt for the old Vendrov release
    └── README.md         # module-level notes
```

Data files (features, captions, images, GloVe) are **not** committed; see
[`data/README.md`](data/README.md). Metrics, logs, the fitted baseline `W` and
the GRU checkpoint are written to `results/coco2017/` (the `.npy`/`.pt` files
are gitignored).

## Setup

Python 3.10+ is assumed.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`curl` must be on the `PATH` (feature extraction uses it for the range
downloads). A GPU is recommended for feature extraction; `--device auto` picks
CUDA, then Apple MPS, then CPU.

## Running

### 1. Data (COCO 2017 + GloVe)

GloVe 300d goes in `data/glove/glove.6B.300d.txt` (see `data/README.md`).
Then fetch the captions and compute image features — the archives are
streamed, so the 19.3 GB `train2017.zip` is never stored:

```bash
mkdir -p data/coco2017/annotations
curl -L -o /tmp/ann.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -j /tmp/ann.zip 'annotations/captions_*2017.json' -d data/coco2017/annotations

# val2017 also keeps its JPEGs, for the demo
python3 src/extract_features.py --archive val2017 --crops 1 \
    --out-dir data/coco2017/shards/val2017 --save-images data/coco2017/val2017
python3 src/extract_features.py --archive train2017 --crops 1 \
    --out-dir data/coco2017/shards/train2017

python3 src/prepare_coco2017.py --data-dir data/coco2017
```

Extraction is resumable: each finished chunk is a shard file, and a rerun
skips them. On an Apple M3 Pro it ran at about 25–35 images/s, bound by a
roughly 5–10 MB/s connection. `--crops 10` gives the Vendrov 10-crop average
instead (about 10× the GPU work; pass `--crop-name 10crop` to the prepare
step).

### 2. Baseline

```bash
python3 src/run_baseline.py --data-dir data/coco2017 \
    --glove data/glove/glove.6B.300d.txt --num-train-images 0 \
    --save-W results/coco2017/W.npy --metrics-out results/coco2017/baseline.json
```

### 3. GRU + GloVe

```bash
python3 src/train_rnn.py --data-dir data/coco2017 \
    --glove data/glove/glove.6B.300d.txt \
    --num-train-images 0 --num-val-images 0 --epochs 20 \
    --out results/coco2017/rnn_checkpoint.pt --metrics-out results/coco2017/gru.json
```

Each epoch is scored on validation; the best-R@10 checkpoint is kept and then
evaluated once on the test split (5K and 1K). Useful flags: `--embed-dim`,
`--hidden-dim`, `--batch-size`, `--lr`, `--margin`, `--device`.

### 4. Demo (interactive retrieval)

```bash
python3 src/demo.py --data-dir data/coco2017 --split test \
    --images data/coco2017/val2017 --checkpoint results/coco2017/rnn_checkpoint.pt
```

Type a caption at the prompt; it prints the top-K test images and writes
`demo_montage.png`. The baseline can be queried too:

```bash
python3 src/demo.py --backend baseline --data-dir data/coco2017 --split test \
    --images data/coco2017/val2017 --W results/coco2017/W.npy \
    --glove data/glove/glove.6B.300d.txt
```

## Results

Trained on all 113,287 training images (GRU: 20 epochs, about 1.4 min each
on an Apple M3 Pro GPU) and scored on the 5,000 `val2017` test images. Recall
in %.

| Model | Pool | R@1 | R@5 | R@10 | Mean rank |
|---|---|---|---|---|---|
| Random ranking (chance) | 5K | 0.02 | 0.10 | 0.20 | 2,500.5 |
| Baseline | 5K | 2.8 | 10.7 | 17.5 | 156.7 |
| GRU + GloVe | 5K | 10.4 | 30.6 | 43.8 | 48.1 |
| Baseline | 1K | 9.7 | 31.6 | 47.7 | 32.2 |
| GRU + GloVe | 1K | 26.9 | 62.4 | 77.9 | 10.4 |

5K: every test caption ranks all 5,000 test images. 1K: five disjoint
1,000-image folds, averaged — the protocol of the reference report in `docs/`
(GRU + GloVe: R@1 37.0, R@10 86.8), which uses the Karpathy test images.
Full metrics and per-epoch validation scores are in `results/coco2017/`
(`baseline.json`, `gru.json`, and the training logs). See the write-up for the
analysis.

## Rebuilding the write-up and slides

`deliverables/writeup.md` and `deliverables/slides.md` (Marp) are the sources
of the PDFs and the PPTX. With Node.js and Google Chrome installed:

```bash
cd deliverables && ./build.sh
```

## Deliverables Checklist

- [x] Source code in private GitHub repo (shared with faculty/TAs)
- [x] README with setup & run instructions
- [x] One-page write-up (PDF)
- [x] Presentation slide deck
- [x] Live demonstration of working code
