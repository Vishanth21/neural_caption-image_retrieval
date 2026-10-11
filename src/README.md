# Source

Modules for the caption-image retrieval mini-project.

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
    --data-dir /home/ubuntu/work/CS229_project/data/coco \
    --glove /home/ubuntu/work/glove.6B.300d.txt \
    --num-train-images 5000
```

Pass `--num-train-images 0` to use the full 113,287-image training split.

The evaluation projects each val caption through `W` and ranks all val
images by squared Euclidean distance, reporting Recall@1/5/10 and mean rank.

### Method

A caption is represented as the mean of the GloVe vectors of its words. A
weight matrix `W` is fit so that `X @ W` approximates the VGG-19 `fc7` image
features `Y` in a least-squares sense. Retrieval is nearest-neighbour search
in that shared space.

## Neural model (GRU + GloVe)

| File | Purpose |
| --- | --- |
| `vocab.py` | Build word-to-id vocab and initialize an embedding matrix from GloVe. |
| `models.py` | Image projection, GRU caption encoder, margin ranking loss. |
| `train_rnn.py` | Train the model, evaluate R@K each epoch, checkpoint the best. |

### Run

```bash
python3 src/train_rnn.py \
    --data-dir data/coco \
    --glove /home/ubuntu/work/glove.6B.300d.txt \
    --num-train-images 5000 \
    --epochs 10
```

Each epoch pairs every image with one randomly chosen caption and uses the
other batch members as negatives (margin ranking loss, margin 0.1, Adam lr
0.001). The best-R@10 checkpoint is written to `rnn_checkpoint.pt`.

Note: the learning rate matters a great deal here. Adam at its default 0.001
converges; at 0.05 the loss falls but Recall@K stalls near chance.
