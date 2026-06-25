#!/usr/bin/env python3
"""Convert RFW pair protocols into face_benchmark pair manifests."""

import argparse
from pathlib import Path


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
SAME_LABELS = {"1", "same", "match", "positive", "genuine", "true"}
DIFF_LABELS = {"0", "different", "diff", "nonmatch", "negative", "impostor", "false"}


def build_image_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        index.setdefault(path.name.lower(), path)
        index.setdefault(path.stem.lower(), path)
    return index


def resolve_image(root: Path, index: dict[str, Path], token: str) -> Path | None:
    token = token.strip().strip("'\"")
    path = Path(token)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(root / path)
        candidates.append(root / path.name)

    if path.suffix.lower() not in IMAGE_SUFFIXES:
        for suffix in IMAGE_SUFFIXES:
            if path.is_absolute():
                candidates.append(Path(f"{path}{suffix}"))
            else:
                candidates.append(root / f"{token}{suffix}")
                candidates.append(root / f"{path.name}{suffix}")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return index.get(token.lower()) or index.get(path.name.lower()) or index.get(path.stem.lower())


def numbered_image(root: Path, index: dict[str, Path], person: str, image_id: str) -> Path | None:
    try:
        number = int(image_id)
    except ValueError:
        return None

    patterns = [
        f"{person}/{person}_{number:04d}",
        f"{person}_{number:04d}",
        f"{person}/{person}_{number}",
        f"{person}_{number}",
        f"{person}{number:04d}",
    ]
    for pattern in patterns:
        image = resolve_image(root, index, pattern)
        if image:
            return image
    return None


def is_numeric_header(parts: list[str]) -> bool:
    return len(parts) in (1, 2) and all(part.isdigit() for part in parts)


def parse_pair(
    root: Path,
    index: dict[str, Path],
    parts: list[str],
) -> tuple[Path | None, Path | None, str | None]:
    last = parts[-1].lower()
    if len(parts) >= 3 and last in SAME_LABELS | DIFF_LABELS:
        label = "same" if last in SAME_LABELS else "different"
        return resolve_image(root, index, parts[0]), resolve_image(root, index, parts[1]), label

    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return numbered_image(root, index, parts[0], parts[1]), numbered_image(root, index, parts[0], parts[2]), "same"

    if len(parts) == 4 and parts[1].isdigit() and parts[3].isdigit():
        return numbered_image(root, index, parts[0], parts[1]), numbered_image(root, index, parts[2], parts[3]), "different"

    return None, None, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", required=True, help="RFW Asian image root or pre-aligned crop folder")
    parser.add_argument("--pairs", required=True, help="Path to Asian_pairs.txt")
    parser.add_argument("--output", required=True, help="Output face_benchmark pair manifest")
    parser.add_argument("--pairs-per-fold", type=int, default=600, help="RFW uses 600 pairs per split")
    args = parser.parse_args()

    root = Path(args.images_root)
    pairs_path = Path(args.pairs)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    index = build_image_index(root)
    written = 0
    skipped = 0
    pair_index = 0

    with pairs_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        header = True
        for raw in src:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if header:
                header = False
                if is_numeric_header(parts):
                    continue

            img_a, img_b, label = parse_pair(root, index, parts)
            if not img_a or not img_b or not label:
                skipped += 1
                continue

            fold = pair_index // max(args.pairs_per_fold, 1)
            dst.write(f"{img_a} {img_b} {label} {fold}\n")
            written += 1
            pair_index += 1

    print(f"wrote {written} pairs to {out_path}")
    if skipped:
        print(f"warning: skipped {skipped} pairs that could not be resolved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
