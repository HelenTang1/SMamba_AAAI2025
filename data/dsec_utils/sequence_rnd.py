from __future__ import annotations

from typing import Dict, List, Tuple

from data.dsec_utils.reader import DSECReader
from data.genx_utils.labels import SparselyBatchedObjectLabels
from data.utils.types import DataType, LoaderDataDictGenX


class DSECSequenceForRandomAccess:
    """
    Random access sequence for DSEC.

    Like SMamba's GenX SequenceForRandomAccess, each item ends at a labeled
    DSEC index and takes sequence_length windows before it. Since samples are
    randomly loaded, IS_FIRST_SAMPLE is always True.
    """

    def __init__(
        self,
        reader: DSECReader,
        seq_to_rel_indices: Dict[str, List[int]],
        only_load_end_labels: bool,
    ) -> None:
        self.reader = reader
        self.seq_to_rel_indices = seq_to_rel_indices
        self.only_load_end_labels = bool(only_load_end_labels)
        self.seq_len = self.reader.sequence_length

        self.samples = self._build_endpoint_samples()
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No DSEC random samples found. sequence_length={self.seq_len}"
            )

        # Used by weighted sampler. Same interface as GenX random sequence.
        self._only_load_labels = False

    def _build_endpoint_samples(self) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        for seq_name, rel_indices in self.seq_to_rel_indices.items():
            if len(rel_indices) < self.seq_len:
                continue
            for end_pos in range(self.seq_len - 1, len(rel_indices)):
                samples.append((seq_name, end_pos))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> LoaderDataDictGenX:
        seq_name, end_pos = self.samples[index]
        rel_indices = self.seq_to_rel_indices[seq_name]

        start_pos = end_pos - self.seq_len + 1
        window_rel_indices = rel_indices[start_pos:end_pos + 1]
        assert len(window_rel_indices) == self.seq_len

        if self._only_load_labels:
            labels = []
            for tidx, rel_idx in enumerate(window_rel_indices):
                if self.only_load_end_labels and tidx < len(window_rel_indices) - 1:
                    labels.append(None)
                    continue
                tracks = self.reader.dsec.get_tracks(rel_idx, directory_name=seq_name)
                labels.append(self.reader.build_object_labels(tracks))
            sparse_labels = SparselyBatchedObjectLabels(sparse_object_labels_batch=labels)
            return {DataType.OBJLABELS_SEQ: sparse_labels}

        return self.reader.build_sample_from_rel_indices(
            seq_name=seq_name,
            rel_indices=window_rel_indices,
            is_first_sample=True,
            pad_to_sequence_length=False,
            only_load_end_labels=self.only_load_end_labels,
        )

    def is_only_loading_labels(self) -> bool:
        return self._only_load_labels

    def only_load_labels(self):
        self._only_load_labels = True

    def load_everything(self):
        self._only_load_labels = False
