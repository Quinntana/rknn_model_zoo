#!/usr/bin/env python3
"""Create eye/nose/mouth-center aligned SCface pair manifests.

ArcFace/InsightFace-style recognition usually aligns faces by estimating a
similarity transform from facial landmarks to a canonical 112x112 template.
SCface provides four landmarks in its text files:

    left_eye, right_eye, nose, mouth_center

The common ArcFace template has five points, including separate left/right
mouth corners. This tool maps SCface's mouth-center to the midpoint of the two
ArcFace mouth-corner template points and estimates one similarity transform
from the four available correspondences.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


ARCFACE_5PT = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)
ARCFACE_4PT = np.vstack([ARCFACE_5PT[:3], ARCFACE_5PT[3:5].mean(axis=0)])


def parse_csv(value: str) -> Optional[set[str]]:
    if value.strip().lower() in ("", "all"):
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
    return "other", "other"


def subject_id(stem: str) -> int:
    return int(stem.split("_", 1)[0])


def read_landmarks(path: Path) -> Dict[str, np.ndarray]:
    landmarks: Dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            parts = raw.split()
            if len(parts) != 9:
                continue
            values = np.array([float(v) for v in parts[1:]], dtype=np.float64).reshape(4, 2)
            landmarks[parts[0].lower()] = values
    return landmarks


def build_fallback_index(root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            out.setdefault(path.stem.lower(), path)
    return out


def first_existing(candidates: Sequence[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def find_image(root: Path, fallback: Dict[str, Path], stem: str) -> Optional[Path]:
    stem_l = stem.lower()
    camera, distance = classify_stem(stem_l)
    candidates: List[Path] = []
    if camera == "frontal":
        candidates.extend(
            [
                root / "mugshot_frontal_cropped_all" / f"{stem}.JPG",
                root / "mugshot_frontal_cropped_all" / f"{stem}.jpg",
                root / "mugshot_frontal_original_all" / f"{stem}.jpg",
                root / "mugshot_rotation_all" / f"{stem}.jpg",
            ]
        )
    elif camera.startswith("cam") and distance in {"1", "2", "3"}:
        cam_num = camera.replace("cam", "")
        candidates.extend(
            [
                root / f"surveillance_cameras_distance_{distance}" / f"cam_{cam_num}" / f"{stem}.jpg",
                root / "surveillance_cameras_all" / f"{stem}.jpg",
            ]
        )
    elif camera == "cam8":
        candidates.append(root / "surveillance_cameras_IR_cam8" / f"{stem}.jpg")
    return first_existing(candidates) or fallback.get(stem_l)


def estimate_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return 2x3 affine matrix mapping src points to dst points."""
    rows = []
    targets = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, -y, 1.0, 0.0])
        targets.append(u)
        rows.append([y, x, 0.0, 1.0])
        targets.append(v)
    params, _, _, _ = np.linalg.lstsq(np.array(rows), np.array(targets), rcond=None)
    a, b, tx, ty = params
    return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float64)


def pil_inverse_coeff(forward: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    full = np.eye(3, dtype=np.float64)
    full[:2, :] = forward
    inv = np.linalg.inv(full)[:2, :]
    return tuple(float(v) for v in inv.reshape(-1))


def align_image(src_path: Path, src_landmarks: np.ndarray, out_path: Path, image_size: int) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    dst_landmarks = ARCFACE_4PT * (float(image_size) / 112.0)
    forward = estimate_similarity(src_landmarks, dst_landmarks)
    coeff = pil_inverse_coeff(forward)
    with Image.open(src_path) as image:
        aligned = image.convert("RGB").transform(
            (image_size, image_size),
            Image.Transform.AFFINE,
            coeff,
            resample=Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0),
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        aligned.save(out_path, quality=95)


def selected_probes(landmarks: Dict[str, np.ndarray],
                    cameras: Optional[set[str]],
                    distances: Optional[set[str]]) -> List[str]:
    out = []
    for stem in landmarks:
        camera, distance = classify_stem(stem)
        if camera == "frontal":
            continue
        if cameras is not None and camera not in cameras:
            continue
        if distances is not None and distance not in distances:
            continue
        out.append(stem)
    return sorted(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", required=True, help="SCface_database root")
    parser.add_argument("--landmarks", required=True, help="SCface all.txt")
    parser.add_argument("--aligned-root", required=True, help="Directory for generated aligned crops")
    parser.add_argument("--output", required=True, help="Output pair manifest")
    parser.add_argument("--cameras", default="cam1,cam2,cam3,cam4,cam5")
    parser.add_argument("--distances", default="all")
    parser.add_argument("--image-size", type=int, default=112)
    args = parser.parse_args()

    images_root = Path(args.images_root)
    aligned_root = Path(args.aligned_root)
    output = Path(args.output)
    landmarks = read_landmarks(Path(args.landmarks))
    fallback = build_fallback_index(images_root)
    cameras = parse_csv(args.cameras)
    distances = parse_csv(args.distances)

    probes = selected_probes(landmarks, cameras, distances)
    needed = set(probes)
    needed.update(f"{i:03d}_frontal" for i in range(1, 131))

    missing = []
    aligned_count = 0
    for stem in sorted(needed):
        src_path = find_image(images_root, fallback, stem)
        src_landmarks = landmarks.get(stem)
        if src_path is None or src_landmarks is None:
            missing.append(stem)
            continue
        align_image(src_path, src_landmarks, aligned_root / f"{stem}.jpg", args.image_size)
        aligned_count += 1
    if missing:
        preview = ", ".join(missing[:10])
        raise SystemExit(f"missing {len(missing)} SCface images/landmarks; first missing: {preview}")

    lines = []
    for probe in probes:
        sid = subject_id(probe)
        neg_sid = 1 if sid == 130 else sid + 1
        fold = min((sid - 1) // 13, 9)
        probe_path = aligned_root / f"{probe}.jpg"
        same_path = aligned_root / f"{sid:03d}_frontal.jpg"
        diff_path = aligned_root / f"{neg_sid:03d}_frontal.jpg"
        lines.append(f"{probe_path.resolve()} {same_path.resolve()} same {fold}")
        lines.append(f"{probe_path.resolve()} {diff_path.resolve()} different {fold}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"aligned_or_verified={aligned_count} aligned_root={aligned_root}")
    print(f"wrote {output} pairs={len(lines)} positives={len(lines)//2} negatives={len(lines)//2}")


if __name__ == "__main__":
    main()
