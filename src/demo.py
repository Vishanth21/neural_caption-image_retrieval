"""Interactive caption -> image retrieval demo.

Usage
-----
    python3 src/demo.py \
        --data-dir data/coco \
        --images /home/ubuntu/work/val2014 \
        --checkpoint rnn_checkpoint.pt

or, for the baseline (needs a saved W, see run_baseline.py):

    python3 src/demo.py --backend baseline --data-dir data/coco \
        --images /home/ubuntu/work/val2014 --W W.npy \
        --glove /home/ubuntu/work/glove.6B.300d.txt

Type a caption at the prompt; the top-K matches are printed, their filenames
resolved via ``val_path.txt``, and (if matplotlib is available) shown as a
montage. Type an empty line to quit.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import retrieve  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive retrieval demo")
    p.add_argument("--backend", choices=["rnn", "baseline"], default="rnn")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--images", default=None,
                   help="directory containing the val2014 .jpg files")
    p.add_argument("--split", default="val")
    p.add_argument("--k", type=int, default=10)
    # rnn
    p.add_argument("--checkpoint", default="rnn_checkpoint.pt")
    # baseline
    p.add_argument("--W", default=None)
    p.add_argument("--glove", default=None)
    p.add_argument("--no-display", action="store_true",
                   help="print filenames only; do not render images")
    return p.parse_args()


def build_retriever(args: argparse.Namespace):
    if args.backend == "rnn":
        return retrieve.load_rnn(
            args.data_dir, args.checkpoint, split=args.split
        )
    if not args.W or not args.glove:
        raise SystemExit("baseline needs --W and --glove")
    return retrieve.load_baseline(
        args.data_dir, args.glove, args.W, split=args.split
    )


def show_montage(paths, rows=None, cols=None):
    """Render a grid of images. No-op if matplotlib or files are missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")          # headless: render to file, not screen
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
    except Exception as e:  # pragma: no cover
        print(f"      (matplotlib unavailable: {e})")
        return None

    existing = [p for p in paths if p and os.path.exists(p)]
    if not existing:
        print("      (no image files found to display)")
        return None

    n = len(existing)
    cols = cols or min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, p in zip(axes, existing):
        try:
            ax.imshow(mpimg.imread(p))
            ax.set_title(os.path.basename(p), fontsize=7)
        except Exception:
            ax.axis("off")
    out = "demo_montage.png"
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print(f"      montage written to {out}")
    return out


def main() -> None:
    args = parse_args()
    print(f"loading {args.backend} retriever ...")
    r = build_retriever(args)
    print(f"  {r.num_images} images indexed")
    if r.paths is None:
        print("  WARNING: no *_path.txt found; filenames will be blank.")

    print("\nType a caption and press Enter. Empty line to quit.\n")
    while True:
        try:
            query = input("caption> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            break

        results = r.retrieve(query, k=args.k)
        print(f"\ntop {args.k} for: {query!r}")
        full_paths = []
        for rank, (row, cap, name) in enumerate(results, 1):
            print(f"  {rank:2d}. row {row:5d}  {name or '(no filename)'}")
            print(f"      caption: {cap}")
            if args.images and name:
                full_paths.append(os.path.join(args.images, name))
        if not args.no_display and full_paths:
            show_montage(full_paths)
        print()


if __name__ == "__main__":
    main()
