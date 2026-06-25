#!/usr/bin/env python3
"""Align an identity-labeled image tree with RetinaFace RKNN landmarks.

The tool expects an image tree shaped like:

    images_root / identity / image.jpg

It detects faces with RetinaFace, selects the highest-score face by default,
warps the image to ArcFace-style five-point templates, and optionally rewrites
existing pair manifests so they point to the aligned crops.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


try:
    from rknnlite.api import RKNNLite
except Exception:  # pragma: no cover - depends on board image
    RKNNLite = None  # type: ignore


ARCFACE_5PT_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass
class Detection:
    bbox: np.ndarray
    score: float
    landmarks: np.ndarray


def image_paths(root: Path) -> List[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def letterbox_resize(image: np.ndarray, size: Tuple[int, int], bg_color: int) -> Tuple[np.ndarray, float, int, int]:
    target_width, target_height = size
    image_height, image_width = image.shape[:2]
    ratio = min(target_width / image_width, target_height / image_height)
    new_width = int(round(image_width * ratio))
    new_height = int(round(image_height * ratio))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    out = np.full((target_height, target_width, 3), bg_color, dtype=np.uint8)
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    out[offset_y : offset_y + new_height, offset_x : offset_x + new_width] = resized
    return out, ratio, offset_x, offset_y


def prior_box(image_size: Tuple[int, int]) -> np.ndarray:
    anchors: List[float] = []
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    feature_maps = [[math.ceil(image_size[0] / step), math.ceil(image_size[1] / step)] for step in steps]
    for k, f in enumerate(feature_maps):
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes[k]:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [(j + 0.5) * steps[k] / image_size[1]]
                dense_cy = [(i + 0.5) * steps[k] / image_size[0]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors.extend([cx, cy, s_kx, s_ky])
    return np.array(anchors, dtype=np.float32).reshape(-1, 4)


def box_decode(loc: np.ndarray, priors: np.ndarray) -> np.ndarray:
    variances = [0.1, 0.2]
    boxes = np.concatenate(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]),
        ),
        axis=1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landmarks(pre: np.ndarray, priors: np.ndarray) -> np.ndarray:
    variances = [0.1, 0.2]
    parts = []
    for i in range(5):
        parts.append(priors[:, :2] + pre[:, 2 * i : 2 * i + 2] * variances[0] * priors[:, 2:])
    return np.concatenate(parts, axis=1)


def nms(dets: np.ndarray, threshold: float) -> List[int]:
    if dets.size == 0:
        return []
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= threshold)[0] + 1]
    return keep


class RetinaFaceRknn:
    def __init__(self, model_path: Path, model_size: int, core: int) -> None:
        if RKNNLite is None:
            raise RuntimeError("rknnlite.api is not available")
        self.model_size = model_size
        self.priors = prior_box((model_size, model_size))
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {model_path} ret={ret}")
        core_mask = {
            0: RKNNLite.NPU_CORE_0,
            1: RKNNLite.NPU_CORE_1,
            2: RKNNLite.NPU_CORE_2,
        }.get(core, RKNNLite.NPU_CORE_0)
        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed ret={ret}")

    def release(self) -> None:
        self.rknn.release()

    def detect(self, bgr: np.ndarray, conf_threshold: float, nms_threshold: float) -> List[Detection]:
        img_h, img_w = bgr.shape[:2]
        letterbox, ratio, offset_x, offset_y = letterbox_resize(bgr, (self.model_size, self.model_size), 114)
        rgb = letterbox[:, :, ::-1]
        outputs = self.rknn.inference(inputs=[rgb])
        if outputs is None or len(outputs) < 3:
            raise RuntimeError("RetinaFace inference returned no outputs")
        loc, conf, landm = outputs[:3]
        loc = np.asarray(loc).squeeze(0)
        conf = np.asarray(conf).squeeze(0)
        landm = np.asarray(landm).squeeze(0)

        boxes = box_decode(loc, self.priors)
        boxes *= np.array([self.model_size, self.model_size, self.model_size, self.model_size], dtype=np.float32)
        boxes[:, 0::2] = np.clip((boxes[:, 0::2] - offset_x) / ratio, 0, img_w)
        boxes[:, 1::2] = np.clip((boxes[:, 1::2] - offset_y) / ratio, 0, img_h)

        landmarks = decode_landmarks(landm, self.priors)
        landmarks *= np.array([self.model_size, self.model_size] * 5, dtype=np.float32)
        landmarks[:, 0::2] = np.clip((landmarks[:, 0::2] - offset_x) / ratio, 0, img_w)
        landmarks[:, 1::2] = np.clip((landmarks[:, 1::2] - offset_y) / ratio, 0, img_h)

        scores = conf[:, 1]
        keep = np.where(scores >= conf_threshold)[0]
        if keep.size == 0:
            return []
        boxes = boxes[keep]
        landmarks = landmarks[keep]
        scores = scores[keep]
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]
        dets = np.hstack([boxes, scores[:, None]]).astype(np.float32, copy=False)
        nms_keep = nms(dets, nms_threshold)
        out: List[Detection] = []
        for i in nms_keep:
            out.append(
                Detection(
                    bbox=dets[i, :4].copy(),
                    score=float(dets[i, 4]),
                    landmarks=landmarks[i].reshape(5, 2).astype(np.float32),
                )
            )
        return out


def align_face(bgr: np.ndarray, landmarks: np.ndarray, image_size: int) -> np.ndarray:
    dst = ARCFACE_5PT_112 * (float(image_size) / 112.0)
    transform, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), dst, method=cv2.LMEDS)
    if transform is None:
        raise RuntimeError("estimateAffinePartial2D failed")
    return cv2.warpAffine(bgr, transform, (image_size, image_size), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def output_path_for(src: Path, images_root: Path, aligned_root: Path) -> Path:
    rel = src.relative_to(images_root)
    return (aligned_root / rel).with_suffix(".jpg")


def choose_detection(dets: Sequence[Detection], mode: str, image_shape: Tuple[int, int]) -> Detection:
    if not dets:
        raise ValueError("no detections")
    if mode == "largest":
        return max(dets, key=lambda d: max(0.0, d.bbox[2] - d.bbox[0]) * max(0.0, d.bbox[3] - d.bbox[1]))
    if mode == "center":
        h, w = image_shape
        cx, cy = w * 0.5, h * 0.5
        return min(dets, key=lambda d: ((d.bbox[0] + d.bbox[2]) * 0.5 - cx) ** 2 + ((d.bbox[1] + d.bbox[3]) * 0.5 - cy) ** 2)
    return max(dets, key=lambda d: d.score)


def rewrite_manifest(input_manifest: Path,
                     output_manifest: Path,
                     images_root: Path,
                     aligned_root: Path,
                     ok_map: Dict[Path, Path]) -> Tuple[int, int]:
    kept = 0
    dropped = 0
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with input_manifest.open("r", encoding="utf-8") as src, output_manifest.open("w", encoding="utf-8") as dst:
        for raw in src:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                dropped += 1
                continue
            p1 = Path(parts[0]).resolve()
            p2 = Path(parts[1]).resolve()
            a1 = ok_map.get(p1)
            a2 = ok_map.get(p2)
            if a1 is None or a2 is None:
                dropped += 1
                continue
            rest = " ".join(parts[2:])
            dst.write(f"{a1.resolve()} {a2.resolve()} {rest}\n")
            kept += 1
    return kept, dropped


def parse_manifest_arg(value: str) -> Tuple[Path, Path]:
    if ":" not in value:
        src = Path(value)
        return src, src.with_name(src.stem + "_aligned.txt")
    left, right = value.split(":", 1)
    return Path(left), Path(right)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="RetinaFace RKNN path")
    parser.add_argument("--images-root", required=True, help="Identity-labeled image root")
    parser.add_argument("--aligned-root-112", required=True, help="Output root for 112x112 aligned crops")
    parser.add_argument("--aligned-root-160", help="Optional output root for 160x160 aligned crops")
    parser.add_argument("--metadata", required=True, help="CSV metadata output")
    parser.add_argument("--failed", required=True, help="Failed image list")
    parser.add_argument("--manifest", action="append", default=[],
                        help="Rewrite manifest, format input.txt:output.txt. Can repeat.")
    parser.add_argument("--model-size", type=int, default=320)
    parser.add_argument("--core", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--nms-threshold", type=float, default=0.4)
    parser.add_argument("--choose", choices=["score", "largest", "center"], default="score")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()

    model_path = Path(args.model)
    images_root = Path(args.images_root).resolve()
    aligned_root_112 = Path(args.aligned_root_112).resolve()
    aligned_root_160 = Path(args.aligned_root_160).resolve() if args.aligned_root_160 else None
    metadata_path = Path(args.metadata)
    failed_path = Path(args.failed)

    paths = image_paths(images_root)
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        print(f"no images found under {images_root}", file=sys.stderr)
        return 2

    detector = RetinaFaceRknn(model_path, args.model_size, args.core)
    ok_map_112: Dict[Path, Path] = {}
    failed: List[Tuple[Path, str]] = []
    start = time.time()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("w", encoding="utf-8", newline="") as meta_file:
        writer = csv.writer(meta_file)
        writer.writerow([
            "source", "status", "score", "num_faces", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y", "nose_x", "nose_y",
            "left_mouth_x", "left_mouth_y", "right_mouth_x", "right_mouth_y", "aligned112", "aligned160",
        ])
        for idx, src_path in enumerate(paths, 1):
            src_resolved = src_path.resolve()
            out112 = output_path_for(src_path, images_root, aligned_root_112)
            out160 = output_path_for(src_path, images_root, aligned_root_160) if aligned_root_160 else None
            try:
                if not args.overwrite and out112.exists() and out112.stat().st_size > 0 and (
                    out160 is None or (out160.exists() and out160.stat().st_size > 0)
                ):
                    ok_map_112[src_resolved] = out112
                    writer.writerow([src_resolved, "existing", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", out112, out160 or ""])
                else:
                    image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
                    if image is None:
                        raise RuntimeError("cv2.imread failed")
                    dets = detector.detect(image, args.conf_threshold, args.nms_threshold)
                    if not dets:
                        raise RuntimeError("no_face")
                    det = choose_detection(dets, args.choose, image.shape[:2])
                    aligned112 = align_face(image, det.landmarks, 112)
                    out112.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(out112), aligned112, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                        raise RuntimeError("write_112_failed")
                    if out160 is not None:
                        aligned160 = align_face(image, det.landmarks, 160)
                        out160.parent.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(out160), aligned160, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                            raise RuntimeError("write_160_failed")
                    ok_map_112[src_resolved] = out112
                    row = [
                        src_resolved, "ok", f"{det.score:.6f}", len(dets),
                        *[f"{v:.3f}" for v in det.bbox],
                        *[f"{v:.3f}" for v in det.landmarks.reshape(-1)],
                        out112, out160 or "",
                    ]
                    writer.writerow(row)
            except Exception as exc:  # keep processing the rest
                failed.append((src_resolved, str(exc)))
                writer.writerow([src_resolved, f"failed:{exc}", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])
            if args.progress_every > 0 and (idx == 1 or idx % args.progress_every == 0 or idx == len(paths)):
                elapsed = max(1e-9, time.time() - start)
                print(
                    f"[align] {idx}/{len(paths)} ok={len(ok_map_112)} failed={len(failed)} "
                    f"rate={idx / elapsed:.2f} img/s",
                    flush=True,
                )

    with failed_path.open("w", encoding="utf-8") as handle:
        for path, reason in failed:
            handle.write(f"{path}\t{reason}\n")

    for item in args.manifest:
        input_manifest, output_manifest = parse_manifest_arg(item)
        kept, dropped = rewrite_manifest(input_manifest, output_manifest, images_root, aligned_root_112, ok_map_112)
        print(f"[manifest] {input_manifest} -> {output_manifest} kept={kept} dropped={dropped}")

    detector.release()
    print(f"[done] images={len(paths)} ok={len(ok_map_112)} failed={len(failed)} metadata={metadata_path} failed_list={failed_path}")
    if aligned_root_160:
        print(f"[done] aligned112={aligned_root_112} aligned160={aligned_root_160}")
    else:
        print(f"[done] aligned112={aligned_root_112}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
