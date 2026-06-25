#!/usr/bin/env python3
"""Convert CelebA-Spoof label JSON into face_benchmark liveness manifest."""

import argparse
import json
from pathlib import Path


def read_label(meta, label_index):
    if isinstance(meta, dict):
        for key in ("live_spoof", "live", "label", "labels"):
            if key in meta:
                value = meta[key]
                if isinstance(value, list):
                    return value[label_index]
                return value
    if isinstance(meta, list):
        return meta[label_index]
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="CelebA-Spoof image root")
    parser.add_argument("--label-json", required=True, help="Path to test_label.json/train_label.json")
    parser.add_argument("--output", required=True, help="Output manifest path")
    parser.add_argument("--label-index", type=int, default=43, help="Index of live/spoof label in CelebA-Spoof metadata")
    parser.add_argument("--live-value", default="1", help="Metadata value to treat as live")
    parser.add_argument("--spoof-value", default="0", help="Metadata value to treat as spoof")
    args = parser.parse_args()

    root = Path(args.root)
    label_path = Path(args.label_json)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(label_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        items = []
        for row in data:
            if isinstance(row, dict):
                image = row.get("path") or row.get("image") or row.get("img")
                items.append((image, row))
            elif isinstance(row, list) and len(row) >= 2:
                items.append((row[0], row[1:]))
            else:
                continue
    elif isinstance(data, dict):
        items = data.items()
    else:
        raise ValueError("unsupported label JSON structure")

    written = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as out:
        for rel_path, meta in items:
            if not rel_path:
                skipped += 1
                continue
            value = str(read_label(meta, args.label_index))
            if value == args.live_value:
                label = "live"
            elif value == args.spoof_value:
                label = "spoof"
            else:
                skipped += 1
                continue
            out.write(f"{root / rel_path} {label}\n")
            written += 1

    print(f"wrote {written} liveness entries to {out_path}")
    if skipped:
        print(f"warning: skipped {skipped} entries with unknown/missing labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
