#!/usr/bin/env python3
"""Convert LFW pairs.txt into face_benchmark pair-manifest format."""

import argparse
from pathlib import Path


def lfw_image(root: Path, person: str, index: str) -> Path:
    return root / person / f"{person}_{int(index):04d}.jpg"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lfw-root", required=True, help="Path to lfw or lfw-deepfunneled image directory")
    parser.add_argument("--pairs", required=True, help="Path to LFW pairs.txt")
    parser.add_argument("--output", required=True, help="Output manifest path")
    args = parser.parse_args()

    root = Path(args.lfw_root)
    pairs_path = Path(args.pairs)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    missing = 0
    with pairs_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        header = True
        for raw in src:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if header:
                header = False
                if len(parts) in (1, 2) and all(p.isdigit() for p in parts):
                    continue

            if len(parts) == 3:
                img_a = lfw_image(root, parts[0], parts[1])
                img_b = lfw_image(root, parts[0], parts[2])
                label = "same"
            elif len(parts) == 4:
                img_a = lfw_image(root, parts[0], parts[1])
                img_b = lfw_image(root, parts[2], parts[3])
                label = "different"
            else:
                continue

            if not img_a.exists() or not img_b.exists():
                missing += 1
            dst.write(f"{img_a} {img_b} {label}\n")
            written += 1

    print(f"wrote {written} pairs to {out_path}")
    if missing:
        print(f"warning: {missing} pairs reference missing image files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
