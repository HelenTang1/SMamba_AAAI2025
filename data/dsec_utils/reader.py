from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from dsec_det.dataset import DSECDet

from data.genx_utils.labels import ObjectLabels, SparselyBatchedObjectLabels
from data.utils.representations import StackedHistogram
from data.utils.types import DataType, DatasetMode, LoaderDataDictGenX


def dataset_mode_to_dsec_split(dataset_mode: DatasetMode) -> str:
    if dataset_mode == DatasetMode.TRAIN:
        return "train"
    if dataset_mode == DatasetMode.VALIDATION:
        return "val"
    if dataset_mode == DatasetMode.TESTING:
        return "test"
    raise NotImplementedError(dataset_mode)


def get_dsec_output_hw(dataset_config: DictConfig) -> Tuple[int, int]:
    """
    Return the actual H, W seen by SMamba after DSEC adapter preprocessing.

    This mirrors the GenX logic:
      GenX downsample_by_factor_2=True -> read event_representations_ds2_nearest.h5
      DSEC downsample_by_factor_2=True -> resize on-the-fly event repr to resolution_hw // 2

    Labels are scaled to this same output size inside DSECReader.build_object_labels().
    """
    output_hw = tuple(dataset_config.resolution_hw)
    assert len(output_hw) == 2

    if bool(dataset_config.downsample_by_factor_2):
        output_hw = tuple(x // 2 for x in output_hw)

    return output_hw


def make_dsec_reader(dataset_mode: DatasetMode, dataset_config: DictConfig) -> "DSECReader":
    split = dataset_mode_to_dsec_split(dataset_mode)
    input_hw = tuple(dataset_config.get("input_hw", [480, 640]))
    output_hw = get_dsec_output_hw(dataset_config)
    split_config = dataset_config.get("train_val_test_split", None)

    return DSECReader(
        root=dataset_config.path,
        split=split,
        sequence_length=int(dataset_config.sequence_length),
        bins=int(dataset_config.get("bins", 10)),
        input_hw=input_hw,
        resize_hw=output_hw,
        count_cutoff=int(dataset_config.get("count_cutoff", 10)),
        fastmode=bool(dataset_config.get("fastmode", True)),
        split_config=split_config,
    )


class DSECReader:
    """
    Shared DSEC reader for random and streaming datasets.

    It uses dsec_det.dataset.DSECDet to read raw events and tracks, then converts them
    to the SMamba recurrent detection format:
        EV_REPR: List[Tensor], each [2 * bins, H, W]
        OBJLABELS_SEQ: SparselyBatchedObjectLabels
        IS_FIRST_SAMPLE: bool
        IS_PADDED_MASK: List[bool]
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        sequence_length: int,
        bins: int = 10,
        input_hw: Tuple[int, int] = (480, 640),
        resize_hw: Optional[Tuple[int, int]] = None,
        count_cutoff: int = 10,
        fastmode: bool = True,
        split_config: Optional[Any] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.sequence_length = int(sequence_length)
        assert self.sequence_length > 0
        assert self.root.is_dir(), str(self.root)

        self.input_hw = tuple(input_hw)
        assert len(self.input_hw) == 2

        self.resize_hw = tuple(resize_hw) if resize_hw is not None else None
        if self.resize_hw is not None:
            assert len(self.resize_hw) == 2

        self.output_hw = self.resize_hw if self.resize_hw is not None else self.input_hw
        self.split_config = split_config

        self.representation = StackedHistogram(
            bins=int(bins),
            height=self.input_hw[0],
            width=self.input_hw[1],
            count_cutoff=int(count_cutoff),
            fastmode=bool(fastmode),
        )

        # Keep this lazy, so DataLoader workers open their own DSECDet object.
        self._dsec: Optional[DSECDet] = None

    @property
    def dsec(self) -> DSECDet:
        if self._dsec is None:
            self._dsec = DSECDet(
                root=self.root,
                split=self.split,
                sync="back",
                debug=False,
                split_config=self.split_config,
            )
        return self._dsec

    def reset_lazy_dataset(self) -> None:
        self._dsec = None

    @staticmethod
    def _seq_name_from_directory(directory: Any) -> str:
        return directory.root.name if hasattr(directory, "root") else str(directory)

    def build_seq_to_rel_indices(self) -> Dict[str, List[int]]:
        """
        Returns a mapping from DSEC sequence name to relative indices inside that sequence.
        """
        seq_to_rel_indices: Dict[str, List[int]] = {}

        for global_idx in range(len(self.dsec)):
            rel_idx, _img_idx_to_track_idx, directory = self.dsec.rel_index(global_idx)
            seq_name = self._seq_name_from_directory(directory)
            seq_to_rel_indices.setdefault(seq_name, []).append(int(rel_idx))

        for seq_name, rel_indices in seq_to_rel_indices.items():
            seq_to_rel_indices[seq_name] = sorted(set(rel_indices))

        # Important when this reader later gets copied into DataLoader workers.
        self.reset_lazy_dataset()
        return seq_to_rel_indices

    @staticmethod
    def _get_event_field(events: Any, key: str) -> Any:
        if isinstance(events, dict):
            return events[key]
        if isinstance(events, np.ndarray) and events.dtype.names is not None:
            return events[key]
        raise TypeError(f"Unsupported events type: {type(events)}")

    @staticmethod
    def _to_long_tensor(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.long()
        return torch.from_numpy(np.asarray(x)).long()

    @staticmethod
    def _resize_chw_tensor(
        x: torch.Tensor,
        resize_hw: Optional[Tuple[int, int]],
        mode: str = "nearest-exact",
    ) -> torch.Tensor:
        if resize_hw is None:
            return x

        resize_hw = tuple(resize_hw)
        if tuple(x.shape[-2:]) == resize_hw:
            return x

        original_dtype = x.dtype
        x = x[None].float()
        if mode in ("linear", "bilinear", "bicubic", "trilinear"):
            x = F.interpolate(x, size=resize_hw, mode=mode, align_corners=False)
        else:
            x = F.interpolate(x, size=resize_hw, mode=mode)
        x = x[0]

        # GenX's ds2 representation is nearest-downsampled and remains uint8.
        if not torch.is_floating_point(torch.empty((), dtype=original_dtype)):
            x = x.round().to(original_dtype)
        return x

    def build_event_repr(self, events: Any) -> torch.Tensor:
        x = self._to_long_tensor(self._get_event_field(events, "x"))
        y = self._to_long_tensor(self._get_event_field(events, "y"))
        p = self._to_long_tensor(self._get_event_field(events, "p"))
        t = self._to_long_tensor(self._get_event_field(events, "t"))

        # DSEC polarity may be bool, 0/1, or -1/1 depending on preprocessing.
        p = (p > 0).long()

        keep = (
            (x >= 0) & (x < self.input_hw[1]) &
            (y >= 0) & (y < self.input_hw[0])
        )
        x = x[keep]
        y = y[keep]
        p = p[keep]
        t = t[keep]

        ev_repr = self.representation.construct(x=x, y=y, pol=p, time=t)
        ev_repr = self._resize_chw_tensor(ev_repr, self.resize_hw, mode="nearest-exact")
        return ev_repr

    def build_object_labels(self, tracks: Optional[np.ndarray]) -> Optional[ObjectLabels]:
        if tracks is None or len(tracks) == 0:
            return None

        in_h, in_w = self.input_hw
        out_h, out_w = self.output_hw
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

        class_id = tracks["class_id"].astype(np.float32)
        class_confidence = tracks["class_confidence"].astype(np.float32)
        t = tracks["t"].astype(np.float32)

        keep = (w > 0) & (h > 0) & (class_id >= 0) & (class_id < 8)
        if keep.sum() == 0:
            return None

        labels_np = np.stack(
            [
                t[keep],
                x0[keep],
                y0[keep],
                w[keep],
                h[keep],
                class_id[keep],
                class_confidence[keep],
            ],
            axis=1,
        ).astype(np.float32)

        labels = torch.from_numpy(labels_np).float()
        obj_labels = ObjectLabels(object_labels=labels, input_size_hw=self.output_hw)
        return obj_labels if len(obj_labels) > 0 else None

    def get_padding_representation(self) -> torch.Tensor:
        c = self.representation.get_shape()[0]
        h, w = self.output_hw
        return torch.zeros((c, h, w), dtype=self.representation.get_torch_dtype())

    def build_sample_from_rel_indices(
        self,
        seq_name: str,
        rel_indices: List[int],
        is_first_sample: bool,
        pad_to_sequence_length: bool,
        only_load_end_labels: bool = False,
    ) -> LoaderDataDictGenX:
        ev_repr_seq: List[torch.Tensor] = []
        obj_labels_seq: List[Optional[ObjectLabels]] = []
        is_padded_mask: List[bool] = []

        for tidx, rel_idx in enumerate(rel_indices):
            events = self.dsec.get_events(rel_idx, directory_name=seq_name)
            tracks = self.dsec.get_tracks(rel_idx, directory_name=seq_name)

            ev_repr_seq.append(self.build_event_repr(events))

            if only_load_end_labels and tidx < len(rel_indices) - 1:
                obj_labels_seq.append(None)
            else:
                obj_labels_seq.append(self.build_object_labels(tracks))

            is_padded_mask.append(False)

        if pad_to_sequence_length and len(ev_repr_seq) < self.sequence_length:
            pad_len = self.sequence_length - len(ev_repr_seq)
            padding_repr = torch.zeros_like(ev_repr_seq[0]) if len(ev_repr_seq) > 0 else self.get_padding_representation()
            ev_repr_seq.extend([padding_repr] * pad_len)
            obj_labels_seq.extend([None] * pad_len)
            is_padded_mask.extend([True] * pad_len)

        sparse_labels = SparselyBatchedObjectLabels(sparse_object_labels_batch=obj_labels_seq)

        return {
            DataType.EV_REPR: ev_repr_seq,
            DataType.OBJLABELS_SEQ: sparse_labels,
            DataType.IS_FIRST_SAMPLE: is_first_sample,
            DataType.IS_PADDED_MASK: is_padded_mask,
        }
