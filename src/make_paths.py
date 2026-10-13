"""Build ``<split>_path.txt``: the COCO filename for each row of a split.

The Vendrov caption files have no filename column (see ../data/README.md), so
the demo cannot show pictures without an external ID source. This script
recovers one from the Karpathy split file ``dataset_coco.json``
(https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).

Rather than trusting that both releases list images in the same order, every
row is matched by its captions: each of the row's five captions is looked up
in a caption -> filename index built from the JSON, and the majority filename
wins. Rows that cannot be matched are reported and written as empty lines.

Usage
-----
    python3 src/make_paths.py \
        --data-dir data/coco \
        --karpathy-json /path/to/dataset_coco.json \
        --split val
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as data_mod  # noqa: E402

_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize(caption: str) -> str:
    """Lowercase and strip punctuation so tokenized and raw captions agree."""
    return " ".join(_NON_WORD.sub(" ", caption.lower()).split())


def caption_index(karpathy: dict) -> Dict[str, str]:
    """Map every normalized caption in the Karpathy JSON to its filename."""
    index: Dict[str, str] = {}
    for img in karpathy["images"]:
        for sent in img["sentences"]:
            text = sent.get("raw") or " ".join(sent["tokens"])
            index.setdefault(normalize(text), img["filename"])
    return index


def match_rows(caps: List[str], index: Dict[str, str]) -> List[str]:
    """Return one filename per image row ("" when no caption matches)."""
    n = len(caps) // data_mod.CAPTIONS_PER_IMAGE
    out = []
    for row in range(n):
        votes = Counter(
            index[k]
            for k in map(normalize, data_mod.captions_for_image(caps, row))
            if k in index
        )
        out.append(votes.most_common(1)[0][0] if votes else "")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build <split>_path.txt")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--karpathy-json", required=True,
                   help="dataset_coco.json from Karpathy's caption_datasets.zip")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = p.parse_args()

    cap_path = os.path.join(args.data_dir, f"{args.split}.txt")
    with open(cap_path, encoding="utf-8") as f:
        caps = [line.strip() for line in f]
    with open(args.karpathy_json, encoding="utf-8") as f:
        index = caption_index(json.load(f))

    names = match_rows(caps, index)
    missing = sum(1 for n in names if not n)

    out = os.path.join(args.data_dir, f"{args.split}_path.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")
    print(f"wrote {out}: {len(names) - missing}/{len(names)} rows matched")
    if missing:
        print(f"WARNING: {missing} rows had no matching caption; "
              "their filenames are blank")


if __name__ == "__main__":
    main()
