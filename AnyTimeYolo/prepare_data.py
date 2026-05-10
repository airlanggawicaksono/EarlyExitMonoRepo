"""Download YOLO datasets via Roboflow API or use local.

Usage:
    python prepare_data.py
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

import config as C  # type: ignore


def prepare_roboflow(dataset: str = "coco") -> Path:
    """Pull dataset from Roboflow. Requires RF_API_KEY in .env."""
    if not C.RF_API_KEY:
        print("[prepare_data] RF_API_KEY missing in .env. Add it or use local data.")
        return C.DATA_DIR / dataset

    if dataset not in C.ROBOFLOW_PROJECTS:
        raise ValueError(f"Unknown dataset {dataset}, add to config.ROBOFLOW_PROJECTS")
    workspace, project, version = C.ROBOFLOW_PROJECTS[dataset]

    from roboflow import Roboflow

    rf = Roboflow(api_key=C.RF_API_KEY)
    proj = rf.workspace(workspace).project(project)
    out = C.DATA_DIR / dataset
    out.mkdir(parents=True, exist_ok=True)

    print(f"[prepare_data] downloading {workspace}/{project} v{version} -> {out}")
    ds = proj.version(version).download("yolov9", location=str(out))
    print(f"  data.yaml at: {ds.location}/data.yaml")
    return Path(ds.location)


def prepare_all():
    for ds in C.DATASETS:
        prepare_roboflow(ds)
    print("\nAll done.")


if __name__ == "__main__":
    prepare_all()
