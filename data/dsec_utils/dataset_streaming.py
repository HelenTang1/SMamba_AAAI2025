from __future__ import annotations

from functools import partialmethod
from pathlib import Path
from typing import List, Union

from omegaconf import DictConfig
from torchdata.datapipes.map import MapDataPipe
from tqdm import tqdm

from data.dsec_utils.reader import dataset_mode_to_dsec_split, make_dsec_reader
from data.dsec_utils.sequence_for_streaming import DSECSequenceForIter, RandAugmentDSECIterDataPipe
from data.utils.stream_concat_datapipe import ConcatStreamingDataPipe
from data.utils.stream_sharded_datapipe import ShardedStreamingDataPipe
from data.utils.types import DatasetMode


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


def build_dsec_streaming_dataset(
    dataset_mode: DatasetMode,
    dataset_config: DictConfig,
    batch_size: int,
    num_workers: int,
) -> Union[ConcatStreamingDataPipe, ShardedStreamingDataPipe]:
    dataset_path = Path(dataset_config.path)
    assert dataset_path.is_dir(), str(dataset_path)

    reader = make_dsec_reader(dataset_mode=dataset_mode, dataset_config=dataset_config)
    seq_to_rel_indices = reader.build_seq_to_rel_indices()

    datapipes: List[MapDataPipe] = []
    num_full_sequences = 0
    num_splits = 0
    num_split_sequences = 0
    guarantee_labels = dataset_mode == DatasetMode.TRAIN
    split_name = dataset_mode_to_dsec_split(dataset_mode)

    for seq_name, rel_indices in tqdm(
        seq_to_rel_indices.items(),
        desc=f"creating DSEC streaming {split_name} datasets",
    ):
        if len(rel_indices) == 0:
            continue

        # Give each sequence its own reader object. This is closer to GenX's per-sequence datapipe behavior.
        seq_reader = make_dsec_reader(dataset_mode=dataset_mode, dataset_config=dataset_config)

        if guarantee_labels:
            new_datapipes = DSECSequenceForIter.get_sequences_with_guaranteed_labels(
                reader=seq_reader,
                seq_name=seq_name,
                rel_indices=rel_indices,
            )
        else:
            new_datapipes = [
                DSECSequenceForIter(
                    reader=seq_reader,
                    seq_name=seq_name,
                    rel_indices=rel_indices,
                )
            ]

        if len(new_datapipes) == 1:
            num_full_sequences += 1
        else:
            num_splits += 1
            num_split_sequences += len(new_datapipes)
        datapipes.extend(new_datapipes)

    print(f"{num_full_sequences=}\n{num_splits=}\n{num_split_sequences=}")
    assert len(datapipes) > 0

    if dataset_mode == DatasetMode.TRAIN:
        augmentation_datapipe_type = partialclass(
            RandAugmentDSECIterDataPipe,
            dataset_config=dataset_config,
        )
        return ConcatStreamingDataPipe(
            datapipe_list=datapipes,
            batch_size=batch_size,
            num_workers=num_workers,
            augmentation_pipeline=augmentation_datapipe_type,
            print_seed_debug=False,
        )

    if dataset_mode in (DatasetMode.VALIDATION, DatasetMode.TESTING):
        fill_value = datapipes[0].get_fully_padded_sample()
        return ShardedStreamingDataPipe(
            datapipe_list=datapipes,
            batch_size=batch_size,
            fill_value=fill_value,
        )

    raise NotImplementedError(dataset_mode)
