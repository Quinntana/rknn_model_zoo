#!/usr/bin/env python3
"""Convert SCface landmark text into a face_benchmark detection manifest.

SCface publishes eye/nose/mouth landmarks, not official detection boxes. This
tool derives approximate boxes from those landmarks, so the resulting AP is an
internal close/medium/far detector sanity check, not an official SCface metric.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


def parse_csv(value: Optional[str]) -> Optional[set[str]]:
    if value is None or value.strip().lower() in ("", "all"):
        return None
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def classify_stem(stem: str) -> Tuple[str, str]:
    parts = stem.lower().split("_")
    if len(parts) >= 3 and parts[1].startswith("cam"):
        return parts[1], parts[2]
    if len(parts) >= 2 and parts[1] == "frontal":
        return "frontal", "gallery"
    if len(parts) >= 2 and parts[1] == "cam8":
        return "cam8", "ir_gallery"
    return "pose", "pose"


def build_image_index(root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            out.setdefault(path.stem.lower(), path)
    return out


def find_image(root: Path, index: Dict[str, Path], stem: str) -> Optional[Path]:
    direct = root / f"{stem}.jpg"
    if direct.exists():
        return direct
    return index.get(stem.lower())


def derive_box(points: Iterable[Tuple[float, float]], x_scale: float, y_scale: float) -> Tuple[float, float, float, float]:
    pts = list(points)
    min_x = min(p[0] for p in pts)
    max_x = max(p[0] for p in pts)
    min_y = min(p[1] for p in pts)
    max_y = max(p[1] for p in pts)
    width = max(1.0, max_x - min_x)
    height = max(1.0, max_y - min_y)
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    box_w = width * x_scale
    box_h = height * y_scale
    return cx - box_w * 0.5, cy - box_h * 0.5, cx + box_w * 0.5, cy + box_h * 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", required=True, help="SCface image root")
    parser.add_argument("--landmarks", required=True, help="SCface all.txt landmark file")
    parser.add_argument("--output", required=True, help="Output face_benchmark detection manifest")
    parser.add_argument("--cameras", default="cam1,cam2,cam3,cam4,cam5",
                        help="Comma-separated cameras, or all. Default is visible daytime cam1..cam5")
    parser.add_argument("--distances", default="all",
                        help="Comma-separated distance labels: 1=far, 2=medium, 3=close, or all")
    parser.add_argument("--include-frontal", action="store_true",
                        help="Include visible frontal gallery images")
    parser.add_argument("--include-ir", action="store_true",
                        help="Include IR cam6/cam7/cam8 images")
    parser.add_argument("--x-scale", type=float, default=2.2,
                        help="Horizontal expansion applied to landmark bbox")
    parser.add_argument("--y-scale", type=float, default=2.2,
                        help="Vertical expansion applied to landmark bbox")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Skip missing images instead of failing")
    args = parser.parse_args()

    images_root = Path(args.images_root)
    landmarks = Path(args.landmarks)
    output = Path(args.output)
    cameras = parse_csv(args.cameras)
    distances = parse_csv(args.distances)

    image_index = build_image_index(images_root)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    missing = []
    seen = Counter()
    kept = Counter()
    with landmarks.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            if len(parts) != 9:
                continue
            stem = parts[0]
            camera, distance = classify_stem(stem)
            seen[(camera, distance)] += 1

            is_ir = camera in {"cam6", "cam7", "cam8"}
            is_frontal = camera == "frontal"
            if is_frontal and not args.include_frontal:
                continue
            if is_ir and not args.include_ir:
                continue
            if cameras is not None and camera not in cameras:
                continue
            if distances is not None and distance not in distances:
                continue

            image = find_image(images_root, image_index, stem)
            if image is None:
                missing.append(stem)
                continue

            values = [float(v) for v in parts[1:]]
            points = [(values[i], values[i + 1]) for i in range(0, len(values), 2)]
            x1, y1, x2, y2 = derive_box(points, args.x_scale, args.y_scale)
            lines.append(f"{image.resolve()} {x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}")
            kept[(camera, distance)] += 1

    if missing and not args.allow_missing:
        preview = ", ".join(missing[:10])
        raise SystemExit(f"missing {len(missing)} images under {images_root}; first missing: {preview}")

    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"wrote {len(lines)} images to {output}")
    if missing:
        print(f"skipped missing images: {len(missing)}")
    for key in sorted(kept):
        print(f"kept {key[0]} distance={key[1]} count={kept[key]}")
    if not lines:
        print("no manifest rows written; check filters and image root")


if __name__ == "__main__":
    main()
