# Source

Modules for the caption-image retrieval mini-project.

## Data pipeline (COCO 2017)

| File | Purpose |
| --- | --- |
| `remote_zip.py` | Read a remote ZIP's central directory and fetch its members in chunks over HTTP range requests (via `curl`). |
| `extract_features.py` | Stream `train2017.zip`/`val2017.zip`, compute VGG-19 `fc7` features (centre crop or 10-crop) on the GPU, write resumable shards. |
| `prepare_coco2017.py` | Captions JSON + shards -> `{train,val,test}.txt`, `_path.txt` and L2-normalised `images/1crop/*.npy`. |

## Baseline (multi-response linear regression)

| File | Purpose |
| --- | --- |
| `data.py` | Load a split's VGG features + captions; enforce the 5-captions-per-image layout. |
| `glove.py` | Load GloVe vectors and mean-pool them into a caption vector. |
| `baseline.py` | Least-squares `W` mapping captions into image space, and Recall@K. |
| `run_baseline.py` | End-to-end: train on the train split, evaluate on val. |

### Run

```bash
python3 src/run_baseline.py \
    --data-dir data/coco2017 \
    --glove data/glove/glove.6B.300d.txt \
    --num-train-images 0
```

`--num-train-images 0` uses the full 113,287-image training split.
The fitted `W` is saved to `W.npy` (`--save-W`) for `demo.py --backend baseline`.

The evaluation projects each caption through `W` and ranks the images by
squared Euclidean distance, reporting Recall@1/5/10 and mean rank on val, then
on test (all 5,000 images, and five 1,000-image folds averaged).

### Method

A caption is represented as the mean of the GloVe vectors of its words. A
weight matrix `W` is fit so that `X @ W` approximates the VGG-19 `fc7` image
features `Y` in a least-squares sense. Retrieval is nearest-neighbour search
in that shared space. `baseline.fit_grouped` solves the same least-squares
problem through the normal equations using per-image caption sums, so the
5×-repeated `Y` (9.3 GB on the full split) is never built.

## Neural model (GRU + GloVe)

| File | Purpose |
| --- | --- |
| `vocab.py` | Build word-to-id vocab and initialize an embedding matrix from GloVe. |
| `models.py` | Image projection, GRU caption encoder, margin ranking loss. |
| `train_rnn.py` | Train the model, evaluate R@K each epoch, checkpoint the best. |

### Run

```bash
python3 src/train_rnn.py \
    --data-dir data/coco2017 \
    --glove data/glove/glove.6B.300d.txt \
    --num-train-images 0 --num-val-images 0 \
    --epochs 20
```

Each epoch pairs every image with one randomly chosen caption and uses the
other batch members as negatives (margin ranking loss, margin 0.1, Adam lr
0.001). The best-validation-R@10 checkpoint is written to `rnn_checkpoint.pt`
(`--out`) and scored on the test split; `--metrics-out` saves per-epoch and
test metrics as JSON.

Note: the learning rate matters a great deal here. Adam at its default 0.001
converges; at 0.05 the loss falls but Recall@K stalls near chance.

Training uses `--device auto` (CUDA, then Apple MPS, then CPU).

## Demo helpers

| File | Purpose |
| --- | --- |
| `retrieve.py` | Load the baseline (`W.npy`) or a GRU checkpoint and rank images for a caption. |
| `demo.py` | Interactive prompt; prints the top-K rows and writes `demo_montage.png`. |
| `make_paths.py` | Old Vendrov release only: build `<split>_path.txt` from Karpathy's `dataset_coco.json`. (`prepare_coco2017.py` writes these directly.) |
