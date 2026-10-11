# Data Sources

This project uses the **Microsoft COCO** dataset with the
**Karpathy & Fei-Fei (2015)** train/val/test split.

## 1. The dataset (images + annotations)

### COCO 2014 downloads

| File | URL |
| --- | --- |
| Train images | http://images.cocodataset.org/zips/train2014.zip |
| Val images | http://images.cocodataset.org/zips/val2014.zip |
| Annotations | http://images.cocodataset.org/annotations/annotations_trainval2014.zip |

Download page: https://cocodataset.org/#download

### Karpathy & Fei-Fei (2015) split

The split (train 113,287 / val 5,000 / test 5,000) is a re-split of the COCO
2014 train+val sets. It is a separate JSON file, not part of the zips above.

| File | URL |
| --- | --- |
| `dataset_coco.json` (JSON only) | https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip |
| JSON + VGG CNN features | https://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip |

Source page: https://cs.stanford.edu/people/karpathy/deepimagesent/

### Expected layout after download & extraction

```
data/
├── train2014/
├── val2014/
├── annotations/
│   ├── captions_train2014.json
│   └── captions_val2014.json
└── dataset_coco.json        # Karpathy split
```



## 2. COCO API / tooling (pycocotools)

Repository: https://github.com/cocodataset/cocoapi

- Annotation parser: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
- Demo notebook: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb

Install via pip (see ../requirements.txt):

```bash
pip install pycocotools
```


## 3. Pretrained features / embeddings

- VGG-19 (fc7 image features), paper: https://arxiv.org/abs/1409.1556
  - torchvision: `torchvision.models.vgg19(pretrained=True)`
- GloVe word vectors (300d): https://nlp.stanford.edu/projects/glove/
  - Direct: https://nlp.stanford.edu/data/glove.6B.zip
