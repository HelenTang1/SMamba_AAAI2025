from functools import partial
from typing import Any, Dict, Optional, Union

import math
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from data.dsec_utils.dataset_rnd import (
    CustomConcatDataset,
    build_dsec_random_access_dataset,
    get_dsec_weighted_random_sampler,
)
from data.dsec_utils.dataset_streaming import build_dsec_streaming_dataset
from data.genx_utils.collate import custom_collate_rnd, custom_collate_streaming
from data.utils.spatial import get_dataloading_hw
from data.utils.types import DatasetMode, DatasetSamplingMode


def get_dataloader_kwargs(
    dataset: Union[Dataset, CustomConcatDataset],
    sampling_mode: DatasetSamplingMode,
    dataset_mode: DatasetMode,
    dataset_config: DictConfig,
    batch_size: int,
    num_workers: int,
) -> Dict[str, Any]:
    if dataset_mode == DatasetMode.TRAIN:
        if sampling_mode == DatasetSamplingMode.STREAM:
            return dict(
                dataset=dataset,
                batch_size=None,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
                drop_last=False,
                collate_fn=custom_collate_streaming,
            )

        if sampling_mode == DatasetSamplingMode.RANDOM:
            use_weighted_rnd_sampling = dataset_config.train.random.weighted_sampling
            sampler = get_dsec_weighted_random_sampler(dataset) if use_weighted_rnd_sampling else None
            return dict(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=sampler is None,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=False,
                drop_last=True,
                collate_fn=custom_collate_rnd,
            )

        raise NotImplementedError(sampling_mode)

    if dataset_mode in (DatasetMode.VALIDATION, DatasetMode.TESTING):
        if sampling_mode == DatasetSamplingMode.STREAM:
            return dict(
                dataset=dataset,
                batch_size=None,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
                drop_last=False,
                collate_fn=custom_collate_streaming,
            )

        if sampling_mode == DatasetSamplingMode.RANDOM:
            return dict(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
                drop_last=True,
                collate_fn=custom_collate_rnd,
            )

        raise NotImplementedError(sampling_mode)

    raise NotImplementedError(dataset_mode)


class DataModule(pl.LightningDataModule):
    """
    DSEC DataModule that follows modules/data/genx.py.

    mixed training is created by returning two dataloaders:
        DatasetSamplingMode.RANDOM -> random access loader
        DatasetSamplingMode.STREAM -> streaming loader
    """

    def __init__(
        self,
        dataset_config: DictConfig,
        num_workers_train: int,
        num_workers_eval: int,
        batch_size_train: int,
        batch_size_eval: int,
    ):
        super().__init__()
        assert num_workers_train >= 0
        assert num_workers_eval >= 0
        assert batch_size_train >= 1
        assert batch_size_eval >= 1

        self.dataset_config = dataset_config
        self.train_sampling_mode = dataset_config.train.sampling
        self.eval_sampling_mode = dataset_config.eval.sampling

        assert self.train_sampling_mode in iter(DatasetSamplingMode)
        assert self.eval_sampling_mode in (DatasetSamplingMode.STREAM, DatasetSamplingMode.RANDOM)

        self.overall_batch_size_train = batch_size_train
        self.overall_batch_size_eval = batch_size_eval
        self.overall_num_workers_train = num_workers_train
        self.overall_num_workers_eval = num_workers_eval

        if self.eval_sampling_mode == DatasetSamplingMode.STREAM:
            self.build_eval_dataset = partial(
                build_dsec_streaming_dataset,
                batch_size=self.overall_batch_size_eval,
                num_workers=self.overall_num_workers_eval,
            )
        elif self.eval_sampling_mode == DatasetSamplingMode.RANDOM:
            self.build_eval_dataset = build_dsec_random_access_dataset
        else:
            raise NotImplementedError(self.eval_sampling_mode)

        self.sampling_mode_2_dataset = dict()
        self.sampling_mode_2_train_workers = dict()
        self.sampling_mode_2_train_batch_size = dict()
        self.validation_dataset = None
        self.test_dataset = None

    def get_dataloading_hw(self):
        return get_dataloading_hw(dataset_config=self.dataset_config)

    def set_mixed_sampling_mode_variables_for_train(self):
        assert self.overall_batch_size_train >= 2, "Cannot use mixed mode with batch size smaller than 2"
        assert self.overall_num_workers_train >= 2, "Cannot use mixed mode with num workers smaller than 2"

        weight_random = self.dataset_config.train.mixed.w_random
        weight_stream = self.dataset_config.train.mixed.w_stream
        assert weight_random > 0
        assert weight_stream > 0

        bs_rnd = min(
            round(self.overall_batch_size_train * weight_random / (weight_stream + weight_random)),
            self.overall_batch_size_train - 1,
        )
        bs_str = self.overall_batch_size_train - bs_rnd
        self.sampling_mode_2_train_batch_size[DatasetSamplingMode.RANDOM] = bs_rnd
        self.sampling_mode_2_train_batch_size[DatasetSamplingMode.STREAM] = bs_str

        workers_rnd = min(
            math.ceil(self.overall_num_workers_train * bs_rnd / self.overall_batch_size_train),
            self.overall_num_workers_train - 1,
        )
        workers_str = self.overall_num_workers_train - workers_rnd
        self.sampling_mode_2_train_workers[DatasetSamplingMode.RANDOM] = workers_rnd
        self.sampling_mode_2_train_workers[DatasetSamplingMode.STREAM] = workers_str

        print(
            f"[Train] Local batch size for:\n"
            f"stream sampling:\t{bs_str}\n"
            f"random sampling:\t{bs_rnd}\n"
            f"[Train] Local num workers for:\n"
            f"stream sampling:\t{workers_str}\n"
            f"random sampling:\t{workers_rnd}"
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            if self.train_sampling_mode == DatasetSamplingMode.MIXED:
                self.set_mixed_sampling_mode_variables_for_train()
            else:
                self.sampling_mode_2_train_workers[self.train_sampling_mode] = self.overall_num_workers_train
                self.sampling_mode_2_train_batch_size[self.train_sampling_mode] = self.overall_batch_size_train

            if self.train_sampling_mode in (DatasetSamplingMode.RANDOM, DatasetSamplingMode.MIXED):
                self.sampling_mode_2_dataset[DatasetSamplingMode.RANDOM] = build_dsec_random_access_dataset(
                    dataset_mode=DatasetMode.TRAIN,
                    dataset_config=self.dataset_config,
                )

            if self.train_sampling_mode in (DatasetSamplingMode.STREAM, DatasetSamplingMode.MIXED):
                self.sampling_mode_2_dataset[DatasetSamplingMode.STREAM] = build_dsec_streaming_dataset(
                    dataset_mode=DatasetMode.TRAIN,
                    dataset_config=self.dataset_config,
                    batch_size=self.sampling_mode_2_train_batch_size[DatasetSamplingMode.STREAM],
                    num_workers=self.sampling_mode_2_train_workers[DatasetSamplingMode.STREAM],
                )

            self.validation_dataset = self.build_eval_dataset(
                dataset_mode=DatasetMode.VALIDATION,
                dataset_config=self.dataset_config,
            )

        elif stage == "validate":
            self.validation_dataset = self.build_eval_dataset(
                dataset_mode=DatasetMode.VALIDATION,
                dataset_config=self.dataset_config,
            )

        elif stage == "test":
            self.test_dataset = self.build_eval_dataset(
                dataset_mode=DatasetMode.TESTING,
                dataset_config=self.dataset_config,
            )

        else:
            raise NotImplementedError(stage)

    def train_dataloader(self):
        train_loaders = dict()

        for sampling_mode, dataset in self.sampling_mode_2_dataset.items():
            train_loaders[sampling_mode] = DataLoader(
                **get_dataloader_kwargs(
                    dataset=dataset,
                    sampling_mode=sampling_mode,
                    dataset_mode=DatasetMode.TRAIN,
                    dataset_config=self.dataset_config,
                    batch_size=self.sampling_mode_2_train_batch_size[sampling_mode],
                    num_workers=self.sampling_mode_2_train_workers[sampling_mode],
                )
            )

        if len(train_loaders) == 1:
            return next(iter(train_loaders.values()))

        assert len(train_loaders) == 2
        return train_loaders

    def val_dataloader(self):
        return DataLoader(
            **get_dataloader_kwargs(
                dataset=self.validation_dataset,
                sampling_mode=self.eval_sampling_mode,
                dataset_mode=DatasetMode.VALIDATION,
                dataset_config=self.dataset_config,
                batch_size=self.overall_batch_size_eval,
                num_workers=self.overall_num_workers_eval,
            )
        )

    def test_dataloader(self):
        return DataLoader(
            **get_dataloader_kwargs(
                dataset=self.test_dataset,
                sampling_mode=self.eval_sampling_mode,
                dataset_mode=DatasetMode.TESTING,
                dataset_config=self.dataset_config,
                batch_size=self.overall_batch_size_eval,
                num_workers=self.overall_num_workers_eval,
            )
        )
