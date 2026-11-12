"""Download CORD v2 (CC BY 4.0) from Hugging Face into .cache/data. About 2.3 GB.

    python -m fieldgate.fetch
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

API = "https://huggingface.co/api/datasets/naver-clova-ix/cord-v2"
BASE = "https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/main/"


def main(target: str = ".cache/data") -> int:
    out = Path(target)
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(API) as response:
        files = sorted(s["rfilename"] for s in json.load(response)["siblings"] if s["rfilename"].endswith(".parquet"))
    for name in files:
        split, index = Path(name).name.split("-")[0], int(Path(name).name.split("-")[1])
        dest = out / (f"cord-train-{index}.parquet" if split == "train" else f"cord-{split}.parquet")
        if dest.exists():
            print(f"have {dest}")
            continue
        print(f"downloading {name} -> {dest}", flush=True)
        urllib.request.urlretrieve(BASE + name, dest)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
