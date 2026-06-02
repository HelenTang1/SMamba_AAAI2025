from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterable
from pathlib import Path
from typing import List

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

from data.dsec_utils.reader import DSECReader, get_dsec_output_hw, make_dsec_reader
from data.dsec_utils.sequence_rnd import DSECSequenceForRandomAccess
from data.genx_utils.labels import SparselyBatchedObjectLabels
from data.utils.augmentor import RandomSpatialAugmentorGenX
from data.utils.types import DataType, DatasetMode, LoaderDataDictGenX


def _mode_to_split(dataset_mode: DatasetMode) -> str:
    if dataset_mode == DatasetMode.TRAIN:
        return "train"
    if dataset_mode == DatasetMode.VALIDATION:
        return "val"
    if dataset_mode == DatasetMode.TESTING:
        return "test"
    raise NotImplementedError(dataset_mode)


class DSECSequenceDataset(Dataset):
    def __init__(
        self,
        root: Path,
        dataset_mode: DatasetMode,
        dataset_config: DictConfig,
    ) -> None:
        assert root.is_dir(), str(root)

        sequence_length = int(dataset_config.sequence_length)
        assert sequence_length > 0
        self.output_seq_len = sequence_length

        self.reader = make_dsec_reader(
            dataset_mode=dataset_mode,
            dataset_config=dataset_config,
        )

        seq_to_rel_indices = self.reader.build_seq_to_rel_indices()
        self.sequence = DSECSequenceForRandomAccess(
            reader=self.reader,
            seq_to_rel_indices=seq_to_rel_indices,
            only_load_end_labels=bool(dataset_config.only_load_end_labels),
        )

        self.spatial_augmentor = None
        if dataset_mode == DatasetMode.TRAIN:
            resolution_hw = get_dsec_output_hw(dataset_config)
            self.spatial_augmentor = RandomSpatialAugmentorGenX(
                dataset_hw=resolution_hw,
                automatic_randomization=True,
                augm_config=dataset_config.data_augmentation.random,
            )

    def only_load_labels(self):
        self.sequence.only_load_labels()

    def load_everything(self):
        self.sequence.load_everything()

    def __len__(self) -> int:
        return len(self.sequence)

    def __getitem__(self, index: int) -> LoaderDataDictGenX:
        item = self.sequence[index]

        if self.spatial_augmentor is not None and not self.sequence.is_only_loading_labels():
            item = self.spatial_augmentor(item)

        return item


class CustomConcatDataset(ConcatDataset):
    datasets: List[DSECSequenceDataset]

    def __init__(self, datasets: Iterable[DSECSequenceDataset]):
        super().__init__(datasets=datasets)

    def only_load_labels(self):
        for idx, dataset in enumerate(self.datasets):
            self.datasets[idx].only_load_labels()

    def load_everything(self):
        for idx, dataset in enumerate(self.datasets):
            self.datasets[idx].load_everything()


def build_dsec_random_access_dataset(
    dataset_mode: DatasetMode,
    dataset_config: DictConfig,
) -> CustomConcatDataset:
    dataset_path = Path(dataset_config.path)
    assert dataset_path.is_dir(), str(dataset_path)

    # DSECDet itself handles train, val, test by split_config.
    # Keep one dataset object here instead of iterating root/train folders like GenX.
    seq_datasets = [
        DSECSequenceDataset(
            root=dataset_path,
            dataset_mode=dataset_mode,
            dataset_config=dataset_config,
        )
    ]
    return CustomConcatDataset(seq_datasets)


def get_dsec_weighted_random_sampler(dataset: CustomConcatDataset) -> WeightedRandomSampler:
    class2count = dict()
    ClassAndCount = namedtuple("ClassAndCount", ["class_ids", "counts"])
    classandcount_list = list()

    print("--- START generating DSEC weighted random sampler ---")
    dataset.only_load_labels()

    for data in tqdm(dataset, desc="iterate through DSEC random dataset"):
        labels: SparselyBatchedObjectLabels = data[DataType.OBJLABELS_SEQ]
        label_list, _valid_batch_indices = labels.get_valid_labels_and_batch_indices()
        if len(label_list) == 0:
            classandcount_list.append(ClassAndCount(class_ids=np.array([], dtype=np.int32), counts=np.array([])))
            continue

        class_ids_seq = []
        for label in label_list:
            class_ids_numpy = np.asarray(label.class_id.numpy(), dtype="int32")
            if len(class_ids_numpy) > 0:
                class_ids_seq.append(class_ids_numpy)

        if len(class_ids_seq) == 0:
            classandcount_list.append(ClassAndCount(class_ids=np.array([], dtype=np.int32), counts=np.array([])))
            continue

        class_ids_seq, counts_seq = np.unique(np.concatenate(class_ids_seq), return_counts=True)
        for class_id, count in zip(class_ids_seq, counts_seq):
            class2count[class_id] = class2count.get(class_id, 0) + count
        classandcount_list.append(ClassAndCount(class_ids=class_ids_seq, counts=counts_seq))

    dataset.load_everything()

    class2weight = {}
    for class_id, count in class2count.items():
        class2weight[class_id] = 1 / max(count, 1)

    weights = []
    for classandcount in classandcount_list:
        weight = 0
        for class_id, count in zip(classandcount.class_ids, classandcount.counts):
            weight += class2weight[class_id] * count
        weights.append(weight if weight > 0 else 1e-12)

    print("--- DONE generating DSEC weighted random sampler ---")
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
