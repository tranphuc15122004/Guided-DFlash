#!/usr/bin/env python3
import os
from datasets import load_dataset, DownloadConfig

DATASETS = [
    ("metamath", {"path": "meta-math/MetaMathQA", "split": "train"}),
    ("magicoder", {"path": "ise-uiuc/Magicoder-Evol-Instruct-110K", "split": "train"}),
    ("math_instruct", {"path": "TIGER-Lab/MathInstruct", "split": "train"}),
]


def main() -> None:
    hf_home = os.environ.get("HF_HOME", "")
    print(f"[CACHE] HF_HOME={hf_home}")
    print(
        "[CACHE] OFFLINE FLAGS "
        f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', 'unset')} "
        f"HF_DATASETS_OFFLINE={os.environ.get('HF_DATASETS_OFFLINE', 'unset')} "
        f"HF_DATASETS_OFFLINE_MODE={os.environ.get('HF_DATASETS_OFFLINE_MODE', 'unset')}"
    )

    for name, kwargs in DATASETS:
        print(f"[CACHE] loading {name}: {kwargs['path']} split={kwargs['split']}")
        ds = load_dataset(
            path=kwargs["path"],
            split=kwargs["split"],
            streaming=False,
            download_config=DownloadConfig(local_files_only=False),
        )
        print(f"[CACHE] done {name}: rows={len(ds):,}")

    print("[CACHE] all requested datasets are cached")


if __name__ == "__main__":
    main()
