from __future__ import annotations

from typing import List, Optional, Tuple

from omegaconf import DictConfig
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.map import MapDataPipe

from data.dsec_utils.reader import DSECReader, get_dsec_output_hw
from data.genx_utils.labels import SparselyBatchedObjectLabels
from data.utils.augmentor import RandomSpatialAugmentorGenX
from data.utils.types import DataType, LoaderDataDictGenX


def _get_position_range_indices(label_positions: List[int], max_len: int) -> List[Tuple[int, int]]:
    """Split train-stream ranges using positions of frames that actually have labels.

    This mirrors GenX _get_ev_repr_range_indices(), but DSEC on-the-fly data
    uses rel_indices instead of precomputed repr_idx. The input must therefore
    be positions in the sorted rel_indices list for frames with valid boxes.
    """
    if len(label_positions) == 0:
        return []

    label_positions = sorted(label_positions)
    out: List[Tuple[int, int]] = []

    meta_start = 0
    for i in range(1, len(label_positions)):
        if label_positions[i] - label_positions[i - 1] > max_len:
            start = max(label_positions[meta_start] - max_len + 1, 0)
            stop = label_positions[i - 1] + 1
            out.append((start, stop))
            meta_start = i

    start = max(label_positions[meta_start] - max_len + 1, 0)
    stop = label_positions[-1] + 1
    out.append((start, stop))
    return out


class DSECSequenceForIter(MapDataPipe):
    """
    Streaming sequence for DSEC.

    The sequence is split into consecutive chunks of sequence_length. Only the
    first chunk of a sequence has IS_FIRST_SAMPLE=True, so SMamba can keep its
    recurrent state across later chunks.
    """

    def __init__(
        self,
        reader: DSECReader,
        seq_name: str,
        rel_indices: List[int],
        range_positions: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()

        self.reader = reader
        self.seq_name = seq_name
        self.rel_indices_full = sorted(rel_indices)
        self.seq_len = self.reader.sequence_length

        if range_positions is None:
            rel_pos_start = 0
            rel_pos_stop = len(self.rel_indices_full)
        else:
            rel_pos_start, rel_pos_stop = range_positions

        assert 0 <= rel_pos_start < rel_pos_stop <= len(self.rel_indices_full), (
            f"{rel_pos_start=}, {rel_pos_stop=}, {len(self.rel_indices_full)=}, {seq_name=}"
        )

        self.rel_indices = self.rel_indices_full[rel_pos_start:rel_pos_stop]
        self.start_positions = list(range(0, len(self.rel_indices), self.seq_len))
        self.stop_positions = self.start_positions[1:] + [len(self.rel_indices)]
        self.length = len(self.start_positions)
        assert self.length > 0

        self._padding_representation = None

    @staticmethod
    def get_sequences_with_guaranteed_labels(
        reader: DSECReader,
        seq_name: str,
        rel_indices: List[int],
    ) -> List["DSECSequenceForIter"]:
        rel_indices = sorted(rel_indices)
        valid_label_rel_indices = set(
            reader.build_valid_label_rel_indices(seq_name=seq_name, rel_indices=rel_indices)
        )
        label_positions = [
            pos for pos, rel_idx in enumerate(rel_indices)
            if rel_idx in valid_label_rel_indices
        ]
        range_positions_list = _get_position_range_indices(
            label_positions=label_positions,
            max_len=reader.sequence_length,
        )

        sequence_list = []
        for range_positions in range_positions_list:
            sequence_list.append(
                DSECSequenceForIter(
                    reader=reader,
                    seq_name=seq_name,
                    rel_indices=rel_indices,
                    range_positions=range_positions,
                )
            )
        return sequence_list

    @property
    def padding_representation(self):
        if self._padding_representation is None:
            self._padding_representation = self.reader.get_padding_representation()
        return self._padding_representation

    def get_fully_padded_sample(self) -> LoaderDataDictGenX:
        ev_repr = [self.padding_representation] * self.seq_len
        labels = [None] * self.seq_len
        sparse_labels = SparselyBatchedObjectLabels(sparse_object_labels_batch=labels)
        return {
            DataType.EV_REPR: ev_repr,
            DataType.OBJLABELS_SEQ: sparse_labels,
            DataType.IS_FIRST_SAMPLE: False,
            DataType.IS_PADDED_MASK: [True] * self.seq_len,
        }

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> LoaderDataDictGenX:
        start_pos = self.start_positions[index]
        stop_pos = self.stop_positions[index]

        sample_len = stop_pos - start_pos
        assert self.seq_len >= sample_len > 0, (
            f"{self.seq_len=}, {sample_len=}, {start_pos=}, {stop_pos=}, {self.seq_name=}"
        )

        window_rel_indices = self.rel_indices[start_pos:stop_pos]
        return self.reader.build_sample_from_rel_indices(
            seq_name=self.seq_name,
            rel_indices=window_rel_indices,
            is_first_sample=(index == 0),
            pad_to_sequence_length=True,
            only_load_end_labels=False,
        )


class RandAugmentDSECIterDataPipe(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe, dataset_config: DictConfig):
        super().__init__()
        self.source_dp = source_dp

        resolution_hw = get_dsec_output_hw(dataset_config)

        self.spatial_augmentor = RandomSpatialAugmentorGenX(
            dataset_hw=resolution_hw,
            automatic_randomization=False,
            augm_config=dataset_config.data_augmentation.stream,
        )

    def __iter__(self):
        self.spatial_augmentor.randomize_augmentation()
        for x in self.source_dp:
            yield self.spatial_augmentor(x)
