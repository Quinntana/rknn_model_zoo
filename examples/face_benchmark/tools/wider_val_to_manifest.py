#!/usr/bin/env python3
"""Convert WIDER FACE bbox annotations into face_benchmark detection manifest."""

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", required=True, help="Path to WIDER_val/images or WIDER_train/images")
    parser.add_argument("--annotation", required=True, help="Path to wider_face_val_bbx_gt.txt or train annotation")
    parser.add_argument("--output", required=True, help="Output manifest path")
    parser.add_argument("--keep-invalid", action="store_true", help="Do not skip WIDER invalid boxes")
    args = parser.parse_args()

    images_root = Path(args.images_root)
    ann_path = Path(args.annotation)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ann_path.read_text(encoding="utf-8").splitlines()
    i = 0
    images = 0
    boxes_written = 0
    with out_path.open("w", encoding="utf-8") as out:
        while i < len(lines):
            rel_image = lines[i].strip()
            i += 1
            if not rel_image:
                continue
            if i >= len(lines):
                break
            try:
                count = int(lines[i].strip())
            except ValueError:
                break
            i += 1

            boxes = []
            for _ in range(count):
                if i >= len(lines):
                    break
                parts = lines[i].strip().split()
                i += 1
                if len(parts) < 4:
                    continue
                x, y, w, h = [float(v) for v in parts[:4]]
                invalid = len(parts) > 7 and parts[7] == "1"
                if (invalid and not args.keep_invalid) or w <= 0 or h <= 0:
                    continue
                boxes.append(f"{x:.1f},{y:.1f},{x + w - 1:.1f},{y + h - 1:.1f}")

            if boxes:
                out.write(f"{images_root / rel_image} {' '.join(boxes)}\n")
                images += 1
                boxes_written += len(boxes)

    print(f"wrote {images} images and {boxes_written} boxes to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
