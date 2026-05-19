"""
Setup helper – install CUDA submodules for 3D Gaussian Splatting.

Usage (from the Gaussians/ directory):
    pip install submodules/diff-gaussian-rasterization
    pip install submodules/simple-knn

Or run this script directly:
    python setup.py
"""

import subprocess
import sys
import os

SUBMODULES = [
    "submodules/diff-gaussian-rasterization",
    "submodules/simple-knn",
]


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    for sub in SUBMODULES:
        path = os.path.join(root, sub)
        if not os.path.isdir(path):
            print(f"WARNING: {path} not found – skipping.")
            print(f"  Clone it first:  git clone <repo-url> {path}")
            continue
        print(f"\n{'='*60}")
        print(f"Installing {sub} ...")
        print(f"{'='*60}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-e", path]
        )
    print("\nAll CUDA submodules installed.")


if __name__ == "__main__":
    main()
