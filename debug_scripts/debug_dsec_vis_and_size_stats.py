from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from dsec_det.dataset import DSECDet
from data.utils.representations import StackedHistogram


DSEC_CLASS_NAMES = {
    0: "pedestrian",
    1: "rider",
    2: "car",
    3: "bus",
    4: "truck",
    5: "bicycle",
    6: "motorcycle",
    7: "train",
}


def seq_name_from_directory(directory: Any) -> str:
    return directory.root.name if hasattr(directory, "root") else str(directory)


def get_event_field(events: Any, key: str) -> Any:
    if isinstance(events, dict):
        return events[key]
    if isinstance(events, np.ndarray) and events.dtype.names is not None:
        return events[key]
    raise TypeError(f"Unsupported events type: {type(events)}")


def to_long_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.long()

    x_np = np.asarray(x)

    # torch.from_numpy does not support uint16 directly.
    if np.issubdtype(x_np.dtype, np.unsignedinteger):
        x_np = x_np.astype(np.int64, copy=False)

    return torch.from_numpy(x_np).long()


def resize_chw_tensor(
    x: torch.Tensor,
    resize_hw: Optional[Tuple[int, int]],
    mode: str = "nearest-exact",
) -> torch.Tensor:
    if resize_hw is None:
        return x

    if tuple(x.shape[-2:]) == tuple(resize_hw):
        return x

    x = x[None].float()

    if mode in ("linear", "bilinear", "bicubic", "trilinear"):
        x = F.interpolate(x, size=resize_hw, mode=mode, align_corners=False)
    else:
        x = F.interpolate(x, size=resize_hw, mode=mode)

    return x[0].round().clamp(0, 255).to(torch.uint8)


def build_event_repr(
    events: Any,
    representation: StackedHistogram,
    input_hw: Tuple[int, int],
    output_hw: Tuple[int, int],
) -> torch.Tensor:
    x = to_long_tensor(get_event_field(events, "x"))
    y = to_long_tensor(get_event_field(events, "y"))
    p = to_long_tensor(get_event_field(events, "p"))
    t = to_long_tensor(get_event_field(events, "t"))

    p = (p > 0).long()

    in_h, in_w = input_hw
    keep = (x >= 0) & (x < in_w) & (y >= 0) & (y < in_h)

    x = x[keep]
    y = y[keep]
    p = p[keep]
    t = t[keep]

    ev_repr = representation.construct(x=x, y=y, pol=p, time=t)
    ev_repr = resize_chw_tensor(ev_repr, output_hw, mode="nearest-exact")
    return ev_repr


def convert_tracks_to_boxes(
    tracks: np.ndarray,
    input_hw: Tuple[int, int],
    output_hw: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    if tracks is None or len(tracks) == 0:
        return {
            "x": np.array([], dtype=np.float32),
            "y": np.array([], dtype=np.float32),
            "w": np.array([], dtype=np.float32),
            "h": np.array([], dtype=np.float32),
            "class_id": np.array([], dtype=np.int64),
        }

    in_h, in_w = input_hw
    out_h, out_w = output_hw

    scale_x = float(out_w) / float(in_w)
    scale_y = float(out_h) / float(in_h)

    x0 = tracks["x"].astype(np.float32) * scale_x
    y0 = tracks["y"].astype(np.float32) * scale_y
    w = tracks["w"].astype(np.float32) * scale_x
    h = tracks["h"].astype(np.float32) * scale_y

    x1 = x0 + w
    y1 = y0 + h

    x0 = np.clip(x0, 0, out_w - 1)
    y0 = np.clip(y0, 0, out_h - 1)
    x1 = np.clip(x1, 0, out_w - 1)
    y1 = np.clip(y1, 0, out_h - 1)

    w = x1 - x0
    h = y1 - y0

    class_id = tracks["class_id"].astype(np.int64)

    keep = (w > 0) & (h > 0) & (class_id >= 0) & (class_id < 8)

    return {
        "x": x0[keep],
        "y": y0[keep],
        "w": w[keep],
        "h": h[keep],
        "class_id": class_id[keep],
    }


def size_category(w: np.ndarray, h: np.ndarray) -> np.ndarray:
    area = w * h

    cats = np.empty(area.shape[0], dtype=object)
    cats[area < 32 ** 2] = "S"
    cats[(area >= 32 ** 2) & (area < 96 ** 2)] = "M"
    cats[area >= 96 ** 2] = "L"
    return cats


def evaluator_filter_mask(
    boxes: Dict[str, np.ndarray],
    min_box_diag: float = 30.0,
    min_box_side: float = 10.0,
) -> np.ndarray:
    w = boxes["w"]
    h = boxes["h"]

    if len(w) == 0:
        return np.zeros((0,), dtype=bool)

    diag = np.sqrt(w ** 2 + h ** 2)
    return (diag >= min_box_diag) & (w >= min_box_side) & (h >= min_box_side)


def event_repr_to_rgb(ev_repr: torch.Tensor, bins: int = 10) -> np.ndarray:
    ev = ev_repr.detach().cpu().float()

    neg = ev[:bins].sum(dim=0).numpy()
    pos = ev[bins: 2 * bins].sum(dim=0).numpy()

    def normalize(x: np.ndarray) -> np.ndarray:
        if x.max() <= 0:
            return np.zeros_like(x, dtype=np.uint8)
        x = x / x.max()
        return (x * 255).astype(np.uint8)

    neg = normalize(neg)
    pos = normalize(pos)

    h, w = pos.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # red for positive, blue for negative
    rgb[..., 0] = pos
    rgb[..., 2] = neg
    rgb[..., 1] = np.maximum(pos, neg) // 4

    return rgb


def draw_boxes(
    rgb: np.ndarray,
    boxes: Dict[str, np.ndarray],
    save_path: Path,
    title: str,
) -> None:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)

    for x, y, w, h, cls_id in zip(
        boxes["x"],
        boxes["y"],
        boxes["w"],
        boxes["h"],
        boxes["class_id"],
    ):
        x0 = float(x)
        y0 = float(y)
        x1 = float(x + w)
        y1 = float(y + h)

        cls_name = DSEC_CLASS_NAMES.get(int(cls_id), str(cls_id))

        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=2)
        draw.text((x0, max(0, y0 - 10)), cls_name, fill=(255, 255, 0))

    draw.text((5, 5), title, fill=(255, 255, 255))
    image.save(save_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, default="config/dataset/dsec.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--out_dir", type=str, default="outputs/debug_dsec_boxes")
    parser.add_argument("--num_vis", type=int, default=20)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--sequence", type=str, default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.dataset_config)

    input_hw = tuple(cfg.get("input_hw", [480, 640]))
    output_hw = tuple(cfg.get("resolution_hw", [480, 640]))

    if bool(cfg.get("downsample_by_factor_2", False)):
        output_hw = tuple(x // 2 for x in output_hw)

    bins = int(cfg.get("bins", 10))
    count_cutoff = int(cfg.get("count_cutoff", 10))
    fastmode = bool(cfg.get("fastmode", True))
    split_config = cfg.get("train_val_test_split", None)

    print("input_hw:", input_hw)
    print("output_hw:", output_hw)
    print("bins:", bins)
    print("count_cutoff:", count_cutoff)
    print("split:", args.split)

    dsec = DSECDet(
        root=Path(args.data_dir),
        split=args.split,
        sync="back",
        debug=False,
        split_config=split_config,
    )
    print("len(dsec):", len(dsec))
    if len(dsec) > 0:
        rel_idx, _, directory = dsec.rel_index(0)
        seq_name = seq_name_from_directory(directory)
        tracks = dsec.get_tracks(rel_idx, directory_name=seq_name)
        print("first seq_name:", seq_name)
        print("first rel_idx:", rel_idx)
        print("first tracks type:", type(tracks))
        print("first tracks len:", 0 if tracks is None else len(tracks))
        if tracks is not None:
            print("first tracks dtype:", tracks.dtype)
            print("first tracks fields:", tracks.dtype.names)


    representation = StackedHistogram(
        bins=bins,
        height=input_hw[0],
        width=input_hw[1],
        count_cutoff=count_cutoff,
        fastmode=fastmode,
    )

    rows = []
    per_class_counts = defaultdict(lambda: {"S": 0, "M": 0, "L": 0, "total": 0})
    per_class_counts_filtered = defaultdict(lambda: {"S": 0, "M": 0, "L": 0, "total": 0})

    total_counts = {"S": 0, "M": 0, "L": 0, "total": 0}
    total_counts_filtered = {"S": 0, "M": 0, "L": 0, "total": 0}

    vis_count = 0
    n = len(dsec) if args.max_frames < 0 else min(len(dsec), args.max_frames)

    for global_idx in range(n):
        rel_idx, _img_idx_to_track_idx, directory = dsec.rel_index(global_idx)
        seq_name = seq_name_from_directory(directory)

        if args.sequence and args.sequence != seq_name:
            continue

        tracks = dsec.get_tracks(rel_idx, directory_name=seq_name)
        boxes = convert_tracks_to_boxes(
            tracks=tracks,
            input_hw=input_hw,
            output_hw=output_hw,
        )

        cats = size_category(boxes["w"], boxes["h"])
        filtered_mask = evaluator_filter_mask(
            boxes,
            min_box_diag=30.0,
            min_box_side=10.0,
        )
        cats_filtered = cats[filtered_mask]

        for cat in cats:
            total_counts[cat] += 1
            total_counts["total"] += 1

        for cls_id, cat in zip(boxes["class_id"], cats):
            cls_name = DSEC_CLASS_NAMES.get(int(cls_id), str(cls_id))
            per_class_counts[cls_name][cat] += 1
            per_class_counts[cls_name]["total"] += 1

        for cat in cats_filtered:
            total_counts_filtered[cat] += 1
            total_counts_filtered["total"] += 1

        for cls_id, cat, keep in zip(boxes["class_id"], cats, filtered_mask):
            if not keep:
                continue
            cls_name = DSEC_CLASS_NAMES.get(int(cls_id), str(cls_id))
            per_class_counts_filtered[cls_name][cat] += 1
            per_class_counts_filtered[cls_name]["total"] += 1

        rows.append(
            {
                "global_idx": global_idx,
                "seq_name": seq_name,
                "rel_idx": int(rel_idx),
                "num_boxes": int(len(cats)),
                "S": int((cats == "S").sum()),
                "M": int((cats == "M").sum()),
                "L": int((cats == "L").sum()),
                "num_boxes_after_filter": int(filtered_mask.sum()),
                "S_after_filter": int((cats_filtered == "S").sum()),
                "M_after_filter": int((cats_filtered == "M").sum()),
                "L_after_filter": int((cats_filtered == "L").sum()),
            }
        )

        if vis_count < args.num_vis and len(cats) > 0:
            events = dsec.get_events(rel_idx, directory_name=seq_name)
            ev_repr = build_event_repr(
                events=events,
                representation=representation,
                input_hw=input_hw,
                output_hw=output_hw,
            )

            rgb = event_repr_to_rgb(ev_repr, bins=bins)

            save_path = out_dir / f"{vis_count:04d}_{seq_name}_rel{int(rel_idx)}.png"
            title = f"{seq_name}, rel_idx={int(rel_idx)}, boxes={len(cats)}"
            draw_boxes(rgb, boxes, save_path, title)

            vis_count += 1

    frame_df = pd.DataFrame(rows)
    frame_df.to_csv(out_dir / f"dsec_{args.split}_frame_size_counts.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {"mode": "raw", **total_counts},
            {"mode": "after_evaluator_filter", **total_counts_filtered},
        ]
    )
    summary_df.to_csv(out_dir / f"dsec_{args.split}_SML_summary.csv", index=False)

    per_class_df = pd.DataFrame(
        [{"class_name": k, **v} for k, v in per_class_counts.items()]
    )
    per_class_df = per_class_df.sort_values("class_name")
    per_class_df.to_csv(out_dir / f"dsec_{args.split}_per_class_SML.csv", index=False)

    per_class_filtered_df = pd.DataFrame(
        [{"class_name": k, **v} for k, v in per_class_counts_filtered.items()]
    )
    per_class_filtered_df = per_class_filtered_df.sort_values("class_name")
    per_class_filtered_df.to_csv(
        out_dir / f"dsec_{args.split}_per_class_SML_after_filter.csv",
        index=False,
    )

    print("\n=== S/M/L summary ===")
    print(summary_df)

    print("\n=== Per-class S/M/L ===")
    print(per_class_df)

    print(f"\nSaved visualizations and CSVs to: {out_dir}")


if __name__ == "__main__":
    main()
