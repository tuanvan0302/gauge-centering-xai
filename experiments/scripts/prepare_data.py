"""Arrange ImageNet val and ImageNet-S-50 masks in the layout the Hidden Monotonicity code expects.

Target layout under DATA_ROOT:
    imagenet/meta/imagenet_2012_validation_synset_labels.txt   line i = wnid of ILSVRC2012_val_{i:08d}
    imagenet/val/<wnid>/ILSVRC2012_val_*.JPEG                    torchvision class-folder layout
    imagenet-s/ImageNetS50/validation-segmentation/<wnid>/*.png  + the ImageNet-S mapping files

The class-folder layout matters: HM's loader finds an image either flat in val/ or in
val/<mask wnid>/, so with class folders the 136 masks whose ImageNet label differs from the
ImageNet-S folder are dropped, leaving 616 images (50 val + 566 test), as in the paper.

Usage:
    python prepare_data.py --data-root ../../../data --flat-src ../../../data/ILSVRC2012_img_val \
        --masks-src ../../../experiments/data/ImageNetS50
Every step is idempotent; rerunning skips what is already in place.
"""

import argparse
import csv
import hashlib
import shutil
import urllib.request
from pathlib import Path

LABELS_URL = ("https://raw.githubusercontent.com/tensorflow/models/master/"
              "research/slim/datasets/imagenet_2012_validation_synset_labels.txt")
LABELS_SHA256 = "313fc3f23c864dde9183c9a368809065765dd1b657a6904fded44e3e46d34604"


def get_labels(meta: Path) -> list[str]:
    path = meta / "imagenet_2012_validation_synset_labels.txt"
    if not path.exists():
        meta.mkdir(parents=True, exist_ok=True)
        print(f"downloading {LABELS_URL}")
        urllib.request.urlretrieve(LABELS_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == LABELS_SHA256, f"label file sha256 mismatch: {digest}"
    labels = path.read_text().split()
    assert len(labels) == 50000 and len(set(labels)) == 1000
    return labels


def arrange_val(flat_src: Path, val: Path, labels: list[str], log_path: Path) -> None:
    """Move flat ILSVRC2012_val_*.JPEG files into val/<wnid>/ (a move, not a copy)."""
    if flat_src.exists() and flat_src.resolve() != val.resolve():
        files = sorted(flat_src.glob("ILSVRC2012_val_*.JPEG"))
        print(f"moving {len(files)} images from {flat_src} to {val}/<wnid>/")
        new_log = not log_path.exists()
        with open(log_path, "a", newline="") as fh:
            writer = csv.writer(fh)
            if new_log:
                writer.writerow(["source", "destination"])
            for f in files:
                wnid = labels[int(f.stem.split("_")[-1]) - 1]
                dst = val / wnid / f.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                f.replace(dst)
                writer.writerow([str(f), str(dst)])
        if not any(flat_src.iterdir()):
            flat_src.rmdir()
    n = sum(1 for _ in val.glob("*/*.JPEG"))
    n_dirs = sum(1 for d in val.iterdir() if d.is_dir())
    assert n == 50000 and n_dirs == 1000, f"val has {n} images in {n_dirs} folders"
    print(f"val ok: {n} images in {n_dirs} class folders")


def copy_masks(masks_src: Path, dst: Path) -> None:
    if not dst.exists():
        print(f"copying {masks_src} to {dst}")
        shutil.copytree(masks_src, dst, ignore=shutil.ignore_patterns("validation"))
    n = sum(1 for _ in (dst / "validation-segmentation").glob("*/*.png"))
    assert n == 752, f"expected 752 masks, found {n}"
    print(f"masks ok: {n}")


def check_overlap(data_root: Path) -> None:
    masks = sorted((data_root / "imagenet-s/ImageNetS50/validation-segmentation").glob("*/*.png"))
    val = data_root / "imagenet" / "val"
    found = sum((val / m.parent.name / (m.stem + ".JPEG")).exists() for m in masks)
    print(f"masks with image in val/<mask wnid>/: {found} of {len(masks)} (paper: 616)")
    assert found == 616


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--flat-src", type=Path, required=True, help="folder with the 50k flat val JPEGs")
    ap.add_argument("--masks-src", type=Path, required=True, help="ImageNetS50 folder from LUSSeg/ImageNet-S")
    args = ap.parse_args()

    meta = args.data_root / "imagenet" / "meta"
    labels = get_labels(meta)
    arrange_val(args.flat_src, args.data_root / "imagenet" / "val", labels, meta / "arrange_val_log.csv")
    copy_masks(args.masks_src, args.data_root / "imagenet-s" / "ImageNetS50")
    check_overlap(args.data_root)


if __name__ == "__main__":
    main()
