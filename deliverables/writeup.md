# Caption-Image Retrieval

**UE24CS352A Machine Learning — Mini-Project**

Vishanth (PES1UG24AM323) · Kshitij Gupta (PES1UG24AM915)

## 1. Problem Statement

Given a collection of images and a natural-language query describing a desired image, return a ranked list of the top-K most relevant images. We frame this as learning a *shared embedding space* in which captions and images are aligned, so that nearest-neighbour search for a query's embedding returns the correct image. Quality is measured by **Recall@K** (R@K) — the percentage of query captions whose ground-truth image is ranked in the top K — and the **mean rank** of the ground-truth image.

## 2. Dataset

Microsoft COCO 2017 (Lin et al., 2014), downloaded from cocodataset.org: 118,287 `train2017` and 5,000 `val2017` images, each with at least five human-written captions (we keep the first five). The official test images have no public captions, so `val2017` is our **test** set and 5,000 randomly held-out `train2017` images (seed 0) are our **validation** set for model selection — the same split sizes as Karpathy & Fei-Fei (2015). Captions are lower-cased with punctuation removed.

| Split | Source | Images | Captions | Words per caption (mean · median · max) |
|---|---|---|---|---|
| Train | `train2017` | 113,287 | 566,435 | 10.5 · 10 · 50 |
| Validation | `train2017` (held out) | 5,000 | 25,000 | 10.5 · 10 · 47 |
| Test | `val2017` | 5,000 | 25,000 | 10.5 · 10 · 49 |

95% of captions have at most 15 words; the training captions use 26,351 distinct words (10,417 seen only once); GloVe (Pennington et al., 2014) 300-d vectors cover 82.9% of these word types and 99.87% of word occurrences. Only 0.20% of validation and test tokens are unseen in training.

**Image features.** We extract our own from the raw JPEGs (the 19.3 GB `train2017.zip` is streamed with HTTP range requests, never stored): resize to a short side of 256, centre-crop 224 × 224, and take the L2-normalised 4,096-d `fc7` activation (after ReLU) of an ImageNet VGG-19. Normalisation is essential: in a 3,000-image pilot the GRU on raw `fc7` (norms 19–82) memorised its training pairs and stayed at chance (validation R@10 2% vs 49% normalised).

## 3. Approach

Both models map captions and images into one shared space and retrieve by nearest neighbour.

**Baseline — multi-response linear regression.** A caption is the mean of its words' GloVe vectors. A weight matrix `W` (300 × 4,096) is fit in closed form so that `X @ W` approximates the image features `Y` in a least-squares sense, one row per caption; solving the normal equations from per-image caption sums avoids building the 566,435 × 4,096 (9.3 GB) `Y`. A query is projected through `W` and matched to the nearest image features by squared Euclidean distance.

**GRU + GloVe.** Following Vendrov et al. (2015), an image is projected by a learned linear map from 4,096 to 1,024 dimensions, and a caption is encoded by a GRU: a 26,353 × 300 word-embedding table initialised from GloVe (other words random, all fine-tuned), a single-layer GRU with 1,024 hidden units, and a linear projection of its final state to 1,024 dimensions — 17.2 M parameters in total. Both encoders are trained with a margin ranking loss: with `s(i, c)` the negative squared Euclidean distance between L2-normalised embeddings, each correct pair `(i, c)` in a batch costs `Σ_c' max(0, α − s(i,c) + s(i,c')) + Σ_i' max(0, α − s(i,c) + s(i',c))` over the wrong captions `c'` and wrong images `i'` in that batch, with margin α = 0.1.

## 4. Implementation

Code is in `src/` (NumPy for the baseline; PyTorch for feature extraction and the GRU, on an Apple M3 Pro GPU): `extract_features.py` streams the COCO archives and computes features, `prepare_coco2017.py` builds the splits, `run_baseline.py` and `train_rnn.py` train and evaluate, and `demo.py` answers interactive queries.

Captions are padded or truncated to 20 tokens (truncating 0.6%); unseen words map to `<unk>`. Each epoch shuffles the 113,287 training images and pairs each with one of its five captions at random; batches of 128 (886 steps per epoch) give every positive pair 127 in-batch negatives in each direction. We use Adam with learning rate 0.001 for 20 epochs (about 1.4 min each) and keep the checkpoint with the best validation R@10, which is then scored once on the test set.

## 5. Results

Test set (`val2017`). **5K**: each of the 25,000 test captions ranks all 5,000 test images. **1K**: the reference protocol — five disjoint 1,000-image folds, averaged (chance R@10 1.0%).

| Model | Pool | R@1 | R@5 | R@10 | Mean rank |
|---|---|---|---|---|---|
| Random ranking (chance) | 5K | 0.02 | 0.10 | 0.20 | 2,500.5 |
| Baseline | 5K | 2.8 | 10.7 | 17.5 | 156.7 |
| GRU + GloVe | 5K | 10.4 | 30.6 | 43.8 | 48.1 |
| Baseline | 1K | 9.7 | 31.6 | 47.7 | 32.2 |
| GRU + GloVe | 1K | 26.9 | 62.4 | 77.9 | 10.4 |
| *Reference report — GRU + GloVe* | *1K* | *37.0* | *—* | *86.8* | *7.3* |
| *Vendrov et al. — GRU* | *1K* | *36.3* | *—* | *85.8* | *7.6* |

Both models train on all 113,287 training images. The GRU beats the baseline on every metric and pool: on the 5K pool R@10 rises from 17.5% to 43.8% (2.5×) and R@1 from 2.8% to 10.4% (3.7×), and the mean rank of the correct image falls from 157 to 48 out of 5,000. On the 1K protocol the GRU reaches R@10 77.9% and a mean rank of 10.4 — within 9 points of the reference report (on different test images: `val2017` vs Karpathy's test split). Our earlier run on a 5,000-image subset of the Vendrov features reached 28.3% validation R@10 on a 5K pool; training on the full COCO 2017 split adds 15.5 points (43.8%). Validation R@10 was still rising at epoch 20 (23.2%, 38.4%, 43.8% after epochs 1, 10, 20). In the demo, the query "a cycle" returns a bicycle or motorcycle in all ten of its top results on unseen test images, while the compositional query "two dogs playing in the snow" returns snow scenes and dogs, but only one image of two dogs in the snow.

## 6. Conclusions

Scaling the training data from 5,000 to 113,287 images was the single largest improvement: with the same model and loss, the GRU now ranks the correct image at 48 of 5,000 on average and puts it in the top 10 for 43.8% of test captions, against 17.5% for the baseline. The baseline's mean-pooling discards word order; the GRU reads words in sequence. Computing our own features exposed a detail that precomputed features hide: VGG `fc7` vectors must be L2-normalised, or the model memorises the training set. The remaining gap to the reference (77.9% vs 86.8% 1K R@10) is plausibly due to 20 rather than up to 50 epochs, with validation recall still climbing, and to single centre-crop rather than 10-crop features, which we chose to fit our time budget. Limitations: one global image vector, so counting and composition ("two dogs … in the snow") fail; a sum-of-hinges loss over all in-batch negatives. Future work: train longer with 10-crop features, try a hardest-negative loss (VSE++), attend over image regions, and compare with the order-embedding objective of Vendrov et al.

## References

Lin et al. (2014), *Microsoft COCO: Common Objects in Context* · Karpathy & Fei-Fei (2015), *Deep Visual-Semantic Alignments for Generating Image Descriptions* · Vendrov, Kiros, Fidler & Urtasun (2015), *Order-Embeddings of Images and Language* · Simonyan & Zisserman (2015), *Very Deep Convolutional Networks for Large-Scale Image Recognition* · Pennington, Socher & Manning (2014), *GloVe: Global Vectors for Word Representation* · Qian & Lamberti, *Neural Caption-Image Retrieval*, Stanford CS229 project report (`docs/report.pdf`).
