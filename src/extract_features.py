"""Extract 10-crop VGG-19 fc7 features from a COCO 2017 image archive.

Reproduces the image representation of Vendrov et al. (2015) on the raw COCO
2017 images: each image is resized so its short side is 256, cut into ten
224x224 crops (four corners + centre, and their mirror images), and the
4096-d ``fc7`` activations (after ReLU) of an ImageNet VGG-19 are averaged
over the crops.

The archive is streamed over HTTP range requests (see ``remote_zip.py``), so
the 19.3 GB ``train2017.zip`` never has to be stored. Work is split into
chunks of about ``--chunk-mb`` of the archive; each finished chunk is written
as one shard ``<out-dir>/shard_<k>.npz`` holding ``names`` and ``feats``, so
an interrupted run resumes where it stopped. ``prepare_coco2017.py``
assembles the shards into split ``.npy`` files.

Usage
-----
    python3 src/extract_features.py --archive val2017 \
        --out-dir data/coco2017/shards/val2017 \
        --save-images data/coco2017/val2017
    python3 src/extract_features.py --archive train2017 \
        --out-dir data/coco2017/shards/train2017
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import remote_zip  # noqa: E402

# Same bucket as http://images.cocodataset.org/zips/, addressed over HTTPS.
# Some networks stall plain-HTTP downloads from the cocodataset.org alias.
ZIP_URL = "https://s3.us-east-1.amazonaws.com/images.cocodataset.org/zips/{}.zip"

CENTER_CROP = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.Lambda(lambda x: x.unsqueeze(0)),          # (1, 3, 224, 224)
])

TEN_CROP = T.Compose([
    T.Resize(256),
    T.TenCrop(224),
    T.Lambda(lambda crops: torch.stack([T.functional.to_tensor(c) for c in crops])),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="10-crop VGG-19 fc7 features")
    p.add_argument("--archive", required=True, choices=["train2017", "val2017"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--save-images", default="",
                   help="also write the JPEGs here (e.g. val2017, for the demo)")
    p.add_argument("--crops", type=int, default=10, choices=[1, 10],
                   help="10 = Vendrov 10-crop average; 1 = centre crop only "
                        "(10x less GPU work)")
    p.add_argument("--chunk-mb", type=int, default=64)
    p.add_argument("--workers", type=int, default=16,
                   help="parallel range downloads")
    p.add_argument("--batch-images", type=int, default=16,
                   help="images per forward pass (x crops)")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def vgg19_fc7(device: torch.device) -> torch.nn.Module:
    """VGG-19 truncated after the fc7 ReLU (classifier layers 0..4)."""
    vgg = torchvision.models.vgg19(weights="IMAGENET1K_V1").eval()
    head = list(vgg.classifier.children())[:5]   # fc6, relu, dropout, fc7, relu
    model = torch.nn.Sequential(
        vgg.features, vgg.avgpool, torch.nn.Flatten(), *head
    )
    return model.to(device).eval()


def decode(name_bytes, transform=TEN_CROP):
    """JPEG bytes -> (name, (crops, 3, 224, 224) tensor). Runs in worker threads."""
    name, raw = name_bytes
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return os.path.basename(name), transform(img)


def fetch(url, chunk, size, save_dir):
    """Download a chunk's raw JPEG bytes (decoding happens later, per batch,
    so only compressed bytes are held for the chunks in flight)."""
    items = remote_zip.fetch_chunk(url, chunk, size)
    if save_dir:
        for name, raw in items:
            with open(os.path.join(save_dir, os.path.basename(name)), "wb") as f:
                f.write(raw)
    return items


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(args.save_images, exist_ok=True)
    url = ZIP_URL.format(args.archive)
    transform = TEN_CROP if args.crops == 10 else CENTER_CROP

    print(f"[1/3] reading central directory of {url} ...")
    size = remote_zip.remote_size(url)
    members = remote_zip.list_members(url)
    chunks = remote_zip.chunk_members(members, args.chunk_mb << 20)
    shard = lambda k: os.path.join(args.out_dir, f"shard_{k:05d}.npz")  # noqa: E731
    todo = [k for k in range(len(chunks)) if not os.path.exists(shard(k))]
    print(f"      {len(members)} images in {len(chunks)} chunks; "
          f"{len(chunks) - len(todo)} already done; device={device}")

    print("[2/3] loading VGG-19 ...")
    model = vgg19_fc7(device)

    print("[3/3] extracting ...")
    t0 = time.time()
    done_imgs = 0
    done_shards = 0
    todo_imgs = sum(len(chunks[k]) for k in todo)
    decoder = ThreadPoolExecutor(4)
    with ThreadPoolExecutor(args.workers) as pool:
        # keep at most `workers` chunks in flight so memory stays bounded
        pending = {}
        queue = list(todo)
        while queue or pending:
            while queue and len(pending) < args.workers:
                k = queue.pop(0)
                pending[k] = pool.submit(
                    fetch, url, chunks[k], size, args.save_images
                )
            # take whichever chunk finished first: shards are independent
            # files, so one slow download must not hold up the others
            finished, _ = wait(pending.values(), return_when=FIRST_COMPLETED)
            k = next(j for j, f in pending.items() if f in finished)
            items = pending.pop(k).result()

            bs = args.batch_images
            batches = [items[s : s + bs] for s in range(0, len(items), bs)]
            names = []
            feats = np.empty((len(items), 4096), dtype=np.float32)
            # decode batch i+1 on the CPU while batch i runs on the GPU
            nxt = decoder.submit(lambda b: [decode(x, transform) for x in b], batches[0])
            with torch.no_grad():
                for i in range(len(batches)):
                    decoded = nxt.result()
                    if i + 1 < len(batches):
                        nxt = decoder.submit(
                            lambda b: [decode(x, transform) for x in b], batches[i + 1]
                        )
                    crops = torch.stack([c for _, c in decoded])
                    b = crops.shape[0]
                    out = model(crops.view(-1, 3, 224, 224).to(device))
                    feats[i * bs : i * bs + b] = (
                        out.view(b, args.crops, -1).mean(dim=1).cpu().numpy()
                    )
                    names.extend(n for n, _ in decoded)
            np.savez(shard(k), names=np.asarray(names), feats=feats)

            done_imgs += len(names)
            done_shards += 1
            rate = done_imgs / (time.time() - t0)
            eta = (todo_imgs - done_imgs) / max(rate, 1e-9) / 60
            print(f"      shard {len(chunks) - len(todo) + done_shards}/{len(chunks)} | {done_imgs}/{todo_imgs} "
                  f"images | {rate:5.1f} img/s | eta {eta:5.1f} min", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
