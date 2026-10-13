# Data Sources

This project trains on **Microsoft COCO 2017** (https://cocodataset.org/#download),
with image features we compute ourselves from the raw JPEGs, and **GloVe**
word vectors.

## 1. COCO 2017 captions and images

| File | URL | Used for |
| --- | --- | --- |
| Annotations (252 MB; we need only `captions_*2017.json`) | http://images.cocodataset.org/annotations/annotations_trainval2017.zip | captions |
| Train images (19.3 GB, 118,287 images) | http://images.cocodataset.org/zips/train2017.zip | train + validation features |
| Val images (815 MB, 5,000 images) | http://images.cocodataset.org/zips/val2017.zip | test features + demo |

The image archives do **not** need to be downloaded: `src/extract_features.py`
streams them with HTTP range requests and keeps only the features (it uses the
HTTPS form of the same bucket, `https://s3.us-east-1.amazonaws.com/images.cocodataset.org/zips/…`,
because some networks stall plain-HTTP downloads from `images.cocodataset.org`).

### Split

The official 2017 test images have no public captions, so we use:

| Split | Source | Images | Captions |
| --- | --- | --- | --- |
| train | `train2017` minus the held-out 5,000 | 113,287 | 566,435 |
| val | 5,000 `train2017` images, random (seed 0) | 5,000 | 25,000 |
| test | all of `val2017` | 5,000 | 25,000 |

These are the sizes of the Karpathy & Fei-Fei (2015) split. Every image keeps
its first five captions by annotation id (a few have six or seven), lower-cased
with punctuation stripped.

### Image features

VGG-19 (ImageNet weights from torchvision) `fc7` activations after ReLU,
4,096-d: resize the short side to 256, take the 224 × 224 centre crop
(`--crops 10` averages the Vendrov 10 crops instead), and L2-normalise each
vector. Without normalisation (raw norms 19–82) the GRU model does not
generalise.

### On-disk layout

After `extract_features.py` and `prepare_coco2017.py` (see the top-level
README):

```
data/coco2017/
├── annotations/captions_{train,val}2017.json
├── shards/{train2017,val2017}/shard_*.npz   # raw extractor output (resumable)
├── val2017/*.jpg                           # test images, for the demo
├── {train,val,test}.txt                    # 5 captions per image, one per line
├── {train,val,test}_path.txt               # COCO filename per image
└── images/1crop/{train,val,test}.npy       # (num_images, 4096) float32
```

Lines `5i`–`5i+4` of a `.txt` file are the five captions of row `i` of the
matching `.npy` array, and line `i` of `_path.txt` is its filename.

## 2. Pretrained word embeddings

- GloVe 300d (used to initialise caption embeddings), saved as
  `data/glove/glove.6B.300d.txt`:
  https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
  (Stanford's server can be very slow; the identical zip is mirrored at
  https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip)
- Project page: https://nlp.stanford.edu/projects/glove/

## 3. Earlier release (no longer used)

The first version of this project used the Vendrov et al. (2015) release
(https://www.cs.toronto.edu/~vendrov/order/coco.zip): precomputed 10-crop
VGG-19 features for COCO 2014 images on the Karpathy split, trained on a
5,000-image subset. `data.py` still reads that layout (`images/10crop/`), and
`src/make_paths.py` recovers its filenames from Karpathy's `dataset_coco.json`.
