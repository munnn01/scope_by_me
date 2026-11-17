"""Detect mounted Kaggle MOT17 / SFU-HW datasets or build small synthetic CI fixtures."""
from pathlib import Path
from typing import Optional


KAGGLE_SEARCH_PATHS = [
    "/kaggle/input/mot17challenge/MOT17/train",
    "/kaggle/input/mot17challenge",
    "/kaggle/input/mot-17/MOT17/train",
    "/kaggle/input/mot-17",
    "/kaggle/input/mot17",
    "/kaggle/input/motchallenge-mot17",
    "/kaggle/input/sfu-hw-objects-v1",
    "/kaggle/input/bdd100k",
]


def find_mounted_kaggle_dataset() -> Optional[Path]:
    """Recursively detects MOT17 or SFU-HW-Objects-v1 inside /kaggle/input."""
    kaggle_in = Path("/kaggle/input")
    if kaggle_in.exists():
        for gt_file in kaggle_in.rglob("gt.txt"):
            seq_dir = gt_file.parent.parent
            if (seq_dir / "img1").is_dir():
                train_root = seq_dir.parent
                print(f"[Dataset Auto-Detect] Found real MOT train root at: {train_root}")
                return train_root
        for yuv_file in kaggle_in.rglob("*.yuv"):
            print(f"[Dataset Auto-Detect] Found raw YUV root at: {yuv_file.parent}")
            return yuv_file.parent

    for p_str in KAGGLE_SEARCH_PATHS:
        p = Path(p_str)
        if p.exists():
            print(f"[Dataset Auto-Detect] Found mounted Kaggle dataset at: {p}")
            return p
    return None
