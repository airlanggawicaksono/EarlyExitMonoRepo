"""Download CIFAR-10/100 + SVHN via torchvision; Tiny-ImageNet via HF.

Usage:
    python prepare_data.py
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

import config as C   # type: ignore


def prepare_cifar10():
    from torchvision import datasets
    out = C.DATA_DIR / "cifar10"
    out.mkdir(parents=True, exist_ok=True)
    datasets.CIFAR10(root=str(out), train=True,  download=True)
    datasets.CIFAR10(root=str(out), train=False, download=True)
    print(f"[prepare_data] CIFAR-10 ready at {out}")


def prepare_cifar100():
    from torchvision import datasets
    out = C.DATA_DIR / "cifar100"
    out.mkdir(parents=True, exist_ok=True)
    datasets.CIFAR100(root=str(out), train=True,  download=True)
    datasets.CIFAR100(root=str(out), train=False, download=True)
    print(f"[prepare_data] CIFAR-100 ready at {out}")


def prepare_svhn():
    from torchvision import datasets
    out = C.DATA_DIR / "svhn"
    out.mkdir(parents=True, exist_ok=True)
    datasets.SVHN(root=str(out), split='train', download=True)
    datasets.SVHN(root=str(out), split='test',  download=True)
    print(f"[prepare_data] SVHN ready at {out}")


def prepare_tinyimagenet():
    """Tiny-ImageNet from Stanford. ~250 MB. Auto-extracts to ImageFolder layout."""
    out = C.DATA_DIR / "tinyimagenet"
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "tiny-imagenet-200.zip"
    extract_dir = out / "tiny-imagenet-200"

    if not extract_dir.exists():
        if not zip_path.exists():
            url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            print(f"[prepare_data] downloading {url}")
            urllib.request.urlretrieve(url, zip_path)
        print(f"[prepare_data] extracting {zip_path}")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out)

    # Restructure val/ from "val/images/*.JPEG + val_annotations.txt" -> "val/<class>/*.JPEG"
    val_dir = extract_dir / "val"
    annot   = val_dir / "val_annotations.txt"
    if annot.exists():
        print("[prepare_data] restructuring Tiny-ImageNet val/ for ImageFolder")
        with open(annot) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                fname, cls = parts[0], parts[1]
                src = val_dir / "images" / fname
                cls_dir = val_dir / cls
                cls_dir.mkdir(exist_ok=True)
                if src.exists():
                    src.rename(cls_dir / fname)
        annot.unlink()
        img_dir = val_dir / "images"
        if img_dir.exists() and not any(img_dir.iterdir()):
            img_dir.rmdir()

    # Symlink/copy expected paths (data_root expected = .../tinyimagenet, with train/ + val/)
    for sub in ("train", "val"):
        src = extract_dir / sub
        dst = out / sub
        if src.exists() and not dst.exists():
            try:
                dst.symlink_to(src.resolve(), target_is_directory=True)
            except (OSError, NotImplementedError):
                import shutil
                shutil.copytree(src, dst)
    print(f"[prepare_data] Tiny-ImageNet ready at {out}")


def prepare_imagenet():
    out = C.DATA_DIR / "ImageNet"
    print("[prepare_data] ImageNet must be downloaded manually:")
    print("  1) Register at https://image-net.org")
    print("  2) Download ILSVRC2012 train + val tarballs")
    print(f"  3) Extract to {out}/train/ and {out}/val/ (class-named subdirs)")


def prepare_all():
    prepare_cifar10()
    prepare_cifar100()
    prepare_svhn()
    prepare_tinyimagenet()
    prepare_imagenet()
    print(f"\nAll done. Data at: {C.DATA_DIR}")


if __name__ == "__main__":
    prepare_all()
