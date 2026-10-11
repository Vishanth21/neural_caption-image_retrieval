# Data Sources

This project uses the **Microsoft COCO** dataset with the **Karpathy & Fei-Fei
(2015)** train/val/test split, and **Vendrov et al. (2015)** 10-crop VGG-19
`fc7` image features.

## 1. Features + split (the only download needed to train)

The reference implementation packages the Karpathy split together with the
10-crop VGG-19 features in a single archive:

| File | URL |
| --- | --- |
| Features + split + captions (~1 GB) | http://www.cs.toronto.edu/~vendrov/order/coco.zip |

Extracting yields:

```
data/coco/
├── train.txt          # 566,435 lines = 5 captions x 113,287 images
├── val.txt            #  25,000 lines = 5 captions x   5,000 images
├── test.txt           #  25,000 lines = 5 captions x   5,000 images
└── images/
    └── 10crop/
        ├── train.npy  # (113287, 4096) float32
        ├── val.npy    # (  5000, 4096) float32
        └── test.npy   # (  5000, 4096) float32
```

Notes:

- Features are already in the model-facing orientation, `(num_images, 4096)`.
  Load with `numpy.load` — no transpose required.
- Captions are stored **one per line**, with **5 consecutive lines per image**.
  There is no filename or image ID column: lines `5i` through `5i+4` are the
  five captions for row `i` of the corresponding `.npy` array. Caption-to-image
  mapping is therefore **positional only**.
- Because the text files contain no filenames, displaying the actual image for a
  retrieved row requires an external ID source (see section 2).

## 2. Raw images (only needed for visualization)

Features are precomputed, so images are **not** required to train or evaluate.
Download them only to display retrieved pictures in the demo.

| File | URL |
| --- | --- |
| Val images | http://images.cocodataset.org/zips/val2014.zip |
| Train images | http://images.cocodataset.org/zips/train2014.zip |

Download page: https://cocodataset.org/#download

## 3. Pretrained word embeddings

- GloVe 300d (used to initialize caption embeddings):
  https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
- Project page: https://nlp.stanford.edu/projects/glove/
