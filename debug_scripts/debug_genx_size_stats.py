from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


CLASS_NAMES = {
    "gen1": {
        0: "car",
        1: "pedestrian",
    },
    "gen4": {
        0: "pedestrian",
        1: "two-wheeler",
        2: "car",
    },
}


ORIGINAL_HW = {
    "gen1": (240, 304),
    "gen4": (720, 1280),
}


def size_category(w: np.ndarray, h: np.ndarray) -> np.ndarray:
    area = w * h

    cats = np.empty(area.shape[0], dtype=object)
    cats[area < 32 ** 2] = "S"
    cats[(area >= 32 ** 2) & (area < 96 ** 2)] = "M"
    cats[area >= 96 ** 2] = "L"
    return cats


def evaluator_filter_mask(
    w: np.ndarray,
    h: np.ndarray,
    dataset: str,
) -> np.ndarray:
    if len(w) == 0:
        return np.zeros((0,), dtype=bool)

    # This follows SMamba Prophesee-style filter.
    if dataset in {"gen4", "etram"}:
        min_box_diag = 60.0
        min_box_side = 20.0
    else:
        min_box_diag = 30.0
        min_box_side = 10.0

    diag = np.sqrt(w ** 2 + h ** 2)
    return (diag >= min_box_diag) & (w >= min_box_side) & (h >= min_box_side)


def update_counts(
    counts: Dict[str, int],
    cats: np.ndarray,
) -> None:
    for cat in cats:
        counts[cat] += 1
        counts["total"] += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="gen1", choices=["gen1", "gen4"])
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--out_dir", type=str, default="debug_scripts/debug_genx_size_stats")
    parser.add_argument("--downsample_by_factor_2", action="store_true")
    parser.add_argument("--max_sequences", type=int, default=-1)
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    split_dir = data_root / args.split

    assert data_root.is_dir(), f"data_dir not found: {data_root}"
    assert split_dir.is_dir(), f"split_dir not found: {split_dir}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = CLASS_NAMES[args.dataset]

    input_hw = ORIGINAL_HW[args.dataset]
    output_hw = input_hw

    if args.downsample_by_factor_2:
        output_hw = tuple(x // 2 for x in output_hw)

    scale_y = output_hw[0] / input_hw[0]
    scale_x = output_hw[1] / input_hw[1]

    print("dataset:", args.dataset)
    print("split:", args.split)
    print("input_hw:", input_hw)
    print("output_hw:", output_hw)
    print("scale_x:", scale_x)
    print("scale_y:", scale_y)

    seq_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])

    if args.max_sequences > 0:
        seq_dirs = seq_dirs[: args.max_sequences]

    print("num sequences:", len(seq_dirs))

    total_counts = {"S": 0, "M": 0, "L": 0, "total": 0}
    total_counts_filtered = {"S": 0, "M": 0, "L": 0, "total": 0}

    per_class_counts = defaultdict(lambda: {"S": 0, "M": 0, "L": 0, "total": 0})
    per_class_counts_filtered = defaultdict(lambda: {"S": 0, "M": 0, "L": 0, "total": 0})

    per_sequence_rows = []
    raw_class_ids = defaultdict(int)

    for seq_dir in seq_dirs:
        labels_path = seq_dir / "labels_v2" / "labels.npz"

        if not labels_path.is_file():
            print(f"[WARNING] missing labels: {labels_path}")
            continue

        label_data = np.load(str(labels_path))
        labels = label_data["labels"]

        if len(labels) == 0:
            per_sequence_rows.append(
                {
                    "sequence": seq_dir.name,
                    "num_boxes": 0,
                    "S": 0,
                    "M": 0,
                    "L": 0,
                    "num_boxes_after_filter": 0,
                    "S_after_filter": 0,
                    "M_after_filter": 0,
                    "L_after_filter": 0,
                }
            )
            continue

        x = labels["x"].astype(np.float32) * scale_x
        y = labels["y"].astype(np.float32) * scale_y
        w = labels["w"].astype(np.float32) * scale_x
        h = labels["h"].astype(np.float32) * scale_y
        class_id = labels["class_id"].astype(np.int64)

        # Optional clamp similar to label handling.
        out_h, out_w = output_hw
        x0 = np.clip(x, 0, out_w - 1)
        y0 = np.clip(y, 0, out_h - 1)
        x1 = np.clip(x + w, 0, out_w - 1)
        y1 = np.clip(y + h, 0, out_h - 1)

        w = x1 - x0
        h = y1 - y0

        keep = (w > 0) & (h > 0)

        x0 = x0[keep]
        y0 = y0[keep]
        w = w[keep]
        h = h[keep]
        class_id = class_id[keep]

        for cid in class_id:
            raw_class_ids[int(cid)] += 1

        cats = size_category(w, h)
        filter_mask = evaluator_filter_mask(w, h, dataset=args.dataset)
        cats_filtered = cats[filter_mask]

        update_counts(total_counts, cats)
        update_counts(total_counts_filtered, cats_filtered)

        for cid, cat in zip(class_id, cats):
            cls_name = class_names.get(int(cid), str(int(cid)))
            per_class_counts[cls_name][cat] += 1
            per_class_counts[cls_name]["total"] += 1

        for cid, cat, keep_filter in zip(class_id, cats, filter_mask):
            if not keep_filter:
                continue
            cls_name = class_names.get(int(cid), str(int(cid)))
            per_class_counts_filtered[cls_name][cat] += 1
            per_class_counts_filtered[cls_name]["total"] += 1

        per_sequence_rows.append(
            {
                "sequence": seq_dir.name,
                "num_boxes": int(len(cats)),
                "S": int((cats == "S").sum()),
                "M": int((cats == "M").sum()),
                "L": int((cats == "L").sum()),
                "num_boxes_after_filter": int(filter_mask.sum()),
                "S_after_filter": int((cats_filtered == "S").sum()),
                "M_after_filter": int((cats_filtered == "M").sum()),
                "L_after_filter": int((cats_filtered == "L").sum()),
            }
        )

    summary_df = pd.DataFrame(
        [
            {"mode": "raw", **total_counts},
            {"mode": "after_evaluator_filter", **total_counts_filtered},
        ]
    )

    per_class_df = pd.DataFrame(
        [{"class_name": k, **v} for k, v in per_class_counts.items()],
        columns=["class_name", "S", "M", "L", "total"],
    )
    if not per_class_df.empty:
        per_class_df = per_class_df.sort_values("class_name")

    per_class_filtered_df = pd.DataFrame(
        [{"class_name": k, **v} for k, v in per_class_counts_filtered.items()],
        columns=["class_name", "S", "M", "L", "total"],
    )
    if not per_class_filtered_df.empty:
        per_class_filtered_df = per_class_filtered_df.sort_values("class_name")

    per_sequence_df = pd.DataFrame(per_sequence_rows)
    if not per_sequence_df.empty:
        per_sequence_df = per_sequence_df.sort_values("sequence")

    summary_path = out_dir / f"{args.dataset}_{args.split}_SML_summary.csv"
    per_class_path = out_dir / f"{args.dataset}_{args.split}_per_class_SML.csv"
    per_class_filtered_path = out_dir / f"{args.dataset}_{args.split}_per_class_SML_after_filter.csv"
    per_sequence_path = out_dir / f"{args.dataset}_{args.split}_per_sequence_SML.csv"

    summary_df.to_csv(summary_path, index=False)
    per_class_df.to_csv(per_class_path, index=False)
    per_class_filtered_df.to_csv(per_class_filtered_path, index=False)
    per_sequence_df.to_csv(per_sequence_path, index=False)

    print("\n=== S/M/L summary ===")
    print(summary_df)

    print("\n=== Per-class S/M/L ===")
    print(per_class_df)

    print("\n=== Per-class S/M/L after evaluator filter ===")
    print(per_class_filtered_df)

    print("\nraw class ids:", dict(sorted(raw_class_ids.items())))

    print("\nSaved:")
    print(summary_path)
    print(per_class_path)
    print(per_class_filtered_path)
    print(per_sequence_path)


if __name__ == "__main__":
    main()
