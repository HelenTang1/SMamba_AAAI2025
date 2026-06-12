from typing import Any, Optional, Tuple, Union, Dict
from warnings import warn

import numpy as np
import pytorch_lightning as pl
import torch
import torch as th
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.types import STEP_OUTPUT

from data.genx_utils.labels import ObjectLabels
from data.utils.types import DataType, LstmStates, ObjDetOutput, DatasetSamplingMode, MambaStates
from models.detection.yolox.utils.boxes import postprocess
from models.detection.yolox_extension.models.detector import YoloXDetector
from models.detection.recurrent_backbone.eventddt_encoding import EventDDTEncodingBridge
from utils.evaluation.prophesee.evaluator import PropheseeEvaluator
from utils.evaluation.prophesee.io.box_loading import to_prophesee
from utils.padding import InputPadderFromShape
from .utils.detection import BackboneFeatureSelector, EventReprSelector, RNNStates, Mode, mode_2_string, \
    merge_mixed_batches


class Module(pl.LightningModule):
    def __init__(self, full_config: DictConfig):
        super().__init__()

        self.full_config = full_config

        self.mdl_config = full_config.model
        in_res_hw = tuple(self.mdl_config.backbone.in_res_hw)
        self.input_padder = InputPadderFromShape(desired_hw=in_res_hw)

        self.mdl = YoloXDetector(self.mdl_config)

        eventddt_encoding_cfg = self.mdl_config.backbone.get("eventddt_encoding", None)

        if eventddt_encoding_cfg is not None and eventddt_encoding_cfg.get("enable", False):
            self.eventddt_encoder = EventDDTEncodingBridge(
                **OmegaConf.to_container(eventddt_encoding_cfg, resolve=True)
            )
        else:
            self.eventddt_encoder = None

        self._maybe_load_pretrained_smamba()
        self._maybe_setup_adapter_only_training()

        self.mode_2_rnn_states: Dict[Mode, RNNStates] = {
            Mode.TRAIN: RNNStates(),
            Mode.VAL: RNNStates(),
            Mode.TEST: RNNStates(),
        }

    def _set_frozen_modules_eval(self) -> None:
        cfg = self._get_pretrained_smamba_cfg()
        if cfg is None:
            return
        if not cfg.get("train_adapter_only", False):
            return
        if not cfg.get("keep_frozen_eval", True):
            return

        # Put the whole detector into eval first.
        self.mdl.eval()

        # Then put only the adapter back to train mode.
        if hasattr(self.mdl.backbone, "eventddt_priority") and self.mdl.backbone.eventddt_priority is not None:
            self.mdl.backbone.eventddt_priority.train()

        if self.eventddt_encoder is not None:
            self.eventddt_encoder.eval()


    def on_train_epoch_start(self) -> None:
        self._set_frozen_modules_eval()

    def _get_pretrained_smamba_cfg(self):
        cfg = self.mdl_config.get("pretrained_smamba", None)
        if cfg is None:
            return None
        if not cfg.get("enable", False):
            return None
        return cfg


    def _maybe_load_pretrained_smamba(self) -> None:
        cfg = self._get_pretrained_smamba_cfg()
        if cfg is None:
            return

        ckpt_path = cfg.get("ckpt_path", None)
        if ckpt_path is None:
            raise ValueError("model.pretrained_smamba.enable=True, but ckpt_path is null.")

        print(f"[Pretrained SMamba] loading checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

        exclude_keywords = list(cfg.get("exclude_keywords", []))

        current_state = self.mdl.state_dict()
        load_state = {}
        skipped = []

        for key, value in state_dict.items():
            # Lightning checkpoint usually has keys like:
            #   mdl.backbone.xxx
            #   mdl.fpn.xxx
            #   mdl.yolox_head.xxx
            if key.startswith("mdl."):
                new_key = key[len("mdl."):]
            else:
                new_key = key

            if any(ex_kw in new_key for ex_kw in exclude_keywords):
                skipped.append((key, "excluded"))
                continue

            if new_key not in current_state:
                skipped.append((key, "not_in_current_model"))
                continue

            if current_state[new_key].shape != value.shape:
                skipped.append((key, f"shape_mismatch {tuple(value.shape)} vs {tuple(current_state[new_key].shape)}"))
                continue

            load_state[new_key] = value

        missing, unexpected = self.mdl.load_state_dict(load_state, strict=False)

        print(f"[Pretrained SMamba] loaded tensors: {len(load_state)}")
        print(f"[Pretrained SMamba] skipped tensors: {len(skipped)}")
        print(f"[Pretrained SMamba] missing keys: {len(missing)}")
        print(f"[Pretrained SMamba] unexpected keys: {len(unexpected)}")

        # Print only a few examples to avoid flooding stdout.
        if len(missing) > 0:
            print("[Pretrained SMamba] missing examples:")
            for k in list(missing)[:20]:
                print(f"  {k}")

        if len(skipped) > 0:
            print("[Pretrained SMamba] skipped examples:")
            for k, reason in skipped[:20]:
                print(f"  {k}: {reason}")


    def _maybe_setup_adapter_only_training(self) -> None:
        cfg = self._get_pretrained_smamba_cfg()
        if cfg is None:
            return

        if not cfg.get("train_adapter_only", False):
            return

        print("[Adapter-only training] freezing all parameters first.")

        for name, p in self.named_parameters():
            p.requires_grad_(False)

        # EventDDT encoder should remain frozen.
        if self.eventddt_encoder is not None:
            self.eventddt_encoder.eval()
            for p in self.eventddt_encoder.parameters():
                p.requires_grad_(False)

        # Only train EventDDT priority adapter.
        if not hasattr(self.mdl.backbone, "eventddt_priority") or self.mdl.backbone.eventddt_priority is None:
            raise RuntimeError(
                "model.pretrained_smamba.train_adapter_only=True, "
                "but self.mdl.backbone.eventddt_priority is None. "
                "Check model.backbone.eventddt_priority.enable=True."
            )

        for name, p in self.mdl.backbone.eventddt_priority.named_parameters():
            p.requires_grad_(True)
            print(f"[Adapter-only training] trainable: mdl.backbone.eventddt_priority.{name}")

        self._print_trainable_parameter_summary()


    def _print_trainable_parameter_summary(self) -> None:
        total = 0
        trainable = 0

        print("[Trainable parameter summary]")
        for name, p in self.named_parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
                print(f"  trainable: {name}, shape={tuple(p.shape)}, numel={n}")

        print(f"[Trainable parameter summary] trainable={trainable:,} / total={total:,}")

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_name = self.full_config.dataset.name
        self.mode_2_hw: Dict[Mode, Optional[Tuple[int, int]]] = {}
        self.mode_2_batch_size: Dict[Mode, Optional[int]] = {}
        self.mode_2_psee_evaluator: Dict[Mode, Optional[PropheseeEvaluator]] = {}
        self.mode_2_sampling_mode: Dict[Mode, DatasetSamplingMode] = {}

        self.started_training = True

        dataset_train_sampling = self.full_config.dataset.train.sampling
        dataset_eval_sampling = self.full_config.dataset.eval.sampling
        assert dataset_train_sampling in iter(DatasetSamplingMode)
        assert dataset_eval_sampling in (DatasetSamplingMode.STREAM, DatasetSamplingMode.RANDOM)
        if stage == 'fit':  # train + val
            self.train_config = self.full_config.training
            self.train_metrics_config = self.full_config.logging.train.metrics

            if self.train_metrics_config.compute:
                self.mode_2_psee_evaluator[Mode.TRAIN] = PropheseeEvaluator(
                    dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            self.mode_2_psee_evaluator[Mode.VAL] = PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            self.mode_2_sampling_mode[Mode.TRAIN] = dataset_train_sampling
            self.mode_2_sampling_mode[Mode.VAL] = dataset_eval_sampling

            for mode in (Mode.TRAIN, Mode.VAL):
                self.mode_2_hw[mode] = None
                self.mode_2_batch_size[mode] = None
            self.started_training = False
        elif stage == 'validate':
            mode = Mode.VAL
            self.mode_2_psee_evaluator[mode] = PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            self.mode_2_sampling_mode[Mode.VAL] = dataset_eval_sampling
            self.mode_2_hw[mode] = None
            self.mode_2_batch_size[mode] = None
        elif stage == 'test':
            mode = Mode.TEST
            self.mode_2_psee_evaluator[mode] = PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            self.mode_2_sampling_mode[Mode.TEST] = dataset_eval_sampling
            self.mode_2_hw[mode] = None
            self.mode_2_batch_size[mode] = None
        else:
            raise NotImplementedError

    def forward(self,
                event_tensor: th.Tensor,
                previous_states: Optional[MambaStates] = None,
                retrieve_detections: bool = True,
                targets=None) \
            -> Tuple[Union[th.Tensor, None], Union[Dict[str, th.Tensor], None], MambaStates]:
        eventddt_tokens = self._encode_eventddt_tokens(event_tensor)
        return self.mdl(x=event_tensor,
                        previous_states=previous_states,
                        retrieve_detections=retrieve_detections,
                        targets=targets,
                        eventddt_tokens=eventddt_tokens)
    # def forward(self,
    #             event_tensor: th.Tensor,
    #             previous_states: Optional[LstmStates] = None) \
    #         -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    #     output = self.mdl.forward_backbone(x=event_tensor,
    #                     previous_states=previous_states)[0]
    #     output = [output[i] for i in [1, 2, 3, 4]]
    #     return output
    def get_worker_id_from_batch(self, batch: Any) -> int:
        return batch['worker_id']

    def get_data_from_batch(self, batch: Any):
        return batch['data']
    
    def _encode_eventddt_tokens(self, ev_tensors: th.Tensor) -> Optional[th.Tensor]:
        if self.eventddt_encoder is None:
            return None
        return self.eventddt_encoder(ev_tensors)
    
    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        self._set_frozen_modules_eval()
        batch = merge_mixed_batches(batch)
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        mode = Mode.TRAIN
        self.started_training = True
        step = self.trainer.global_step
        ev_tensor_sequence = data[DataType.EV_REPR]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        token_mask_sequence = data.get(DataType.TOKEN_MASK, None)

        self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size
        else:
            assert self.mode_2_batch_size[mode] == batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        backbone_feature_selector = BackboneFeatureSelector()
        ev_repr_selector = EventReprSelector()
        obj_labels = list()
        for tidx in range(sequence_len):
            ev_tensors = ev_tensor_sequence[tidx]
            ev_tensors = ev_tensors.to(dtype=self.dtype)
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)
            if token_mask_sequence is not None:
                token_masks = self.input_padder.pad_token_mask(token_mask=token_mask_sequence[tidx])
            else:
                token_masks = None

            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

            eventddt_tokens = self._encode_eventddt_tokens(ev_tensors)
            backbone_features, states = self.mdl.forward_backbone(x=ev_tensors,
                                                                  previous_states=prev_states,
                                                                  token_mask=token_masks,
                                                                  eventddt_tokens=eventddt_tokens)
            prev_states = states

            current_labels, valid_batch_indices = sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
            # Store backbone features that correspond to the available labels.
            if len(current_labels) > 0:
                backbone_feature_selector.add_backbone_features(backbone_features=backbone_features,
                                                                selected_indices=valid_batch_indices)
                obj_labels.extend(current_labels)
                ev_repr_selector.add_event_representations(event_representations=ev_tensors,
                                                           selected_indices=valid_batch_indices)

        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)
        assert len(obj_labels) > 0
        # Batch the backbone features and labels to parallelize the detection code.
        selected_backbone_features = backbone_feature_selector.get_batched_backbone_features()
        labels_yolox = ObjectLabels.get_labels_as_batched_tensor(obj_label_list=obj_labels, format_='yolox')
        labels_yolox = labels_yolox.to(dtype=self.dtype)

        predictions, losses = self.mdl.forward_detect(backbone_features=selected_backbone_features,
                                                      targets=labels_yolox)

        if self.mode_2_sampling_mode[mode] in (DatasetSamplingMode.MIXED, DatasetSamplingMode.RANDOM):
            # We only want to evaluate the last batch_size samples if we use random sampling (or mixed).
            # This is because otherwise we would mostly evaluate the init phase of the sequence.
            predictions = predictions[-batch_size:]
            obj_labels = obj_labels[-batch_size:]

        pred_processed = postprocess(prediction=predictions,
                                     num_classes=self.mdl_config.head.num_classes,
                                     conf_thre=self.mdl_config.postprocess.confidence_threshold,
                                     nms_thre=self.mdl_config.postprocess.nms_threshold)

        loaded_labels_proph, yolox_preds_proph = to_prophesee(obj_labels, pred_processed)

        assert losses is not None
        assert 'loss' in losses

        # For visualization, we only use the last batch_size items.
        output = {
            ObjDetOutput.LABELS_PROPH: loaded_labels_proph[-batch_size:],
            ObjDetOutput.PRED_PROPH: yolox_preds_proph[-batch_size:],
            ObjDetOutput.EV_REPR: ev_repr_selector.get_event_representations_as_list(start_idx=-batch_size),
            ObjDetOutput.SKIP_VIZ: False,
            'loss': losses['loss']
        }

        # Logging
        prefix = f'{mode_2_string[mode]}/'
        log_dict = {f'{prefix}{k}': v for k, v in losses.items()}
        self.log_dict(log_dict, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)

        if mode in self.mode_2_psee_evaluator:
            self.mode_2_psee_evaluator[mode].add_labels(loaded_labels_proph)
            self.mode_2_psee_evaluator[mode].add_predictions(yolox_preds_proph)
            if self.train_metrics_config.detection_metrics_every_n_steps is not None and \
                    step > 0 and step % self.train_metrics_config.detection_metrics_every_n_steps == 0:
                self.run_psee_evaluator(mode=mode)

        return output

    def _val_test_step_impl(self, batch: Any, mode: Mode) -> Optional[STEP_OUTPUT]:
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        assert mode in (Mode.VAL, Mode.TEST)
        ev_tensor_sequence = data[DataType.EV_REPR]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]

        self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size
        else:
            assert self.mode_2_batch_size[mode] == batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        backbone_feature_selector = BackboneFeatureSelector()
        ev_repr_selector = EventReprSelector()
        obj_labels = list()
        for tidx in range(sequence_len):
            collect_predictions = (tidx == sequence_len - 1) or \
                                  (self.mode_2_sampling_mode[mode] == DatasetSamplingMode.STREAM)
            ev_tensors = ev_tensor_sequence[tidx]
            ev_tensors = ev_tensors.to(dtype=self.dtype)
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)
            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

            eventddt_tokens = self._encode_eventddt_tokens(ev_tensors)
            backbone_features, states = self.mdl.forward_backbone(
                x=ev_tensors,
                previous_states=prev_states,
                eventddt_tokens=eventddt_tokens,
            )
            prev_states = states

            if collect_predictions:
                current_labels, valid_batch_indices = sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
                # Store backbone features that correspond to the available labels.
                if len(current_labels) > 0:
                    backbone_feature_selector.add_backbone_features(backbone_features=backbone_features,
                                                                    selected_indices=valid_batch_indices)

                    obj_labels.extend(current_labels)
                    ev_repr_selector.add_event_representations(event_representations=ev_tensors,
                                                               selected_indices=valid_batch_indices)
        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)
        if len(obj_labels) == 0:
            return {ObjDetOutput.SKIP_VIZ: True}
        selected_backbone_features = backbone_feature_selector.get_batched_backbone_features()
        predictions, _ = self.mdl.forward_detect(backbone_features=selected_backbone_features)

        pred_processed = postprocess(prediction=predictions,
                                     num_classes=self.mdl_config.head.num_classes,
                                     conf_thre=self.mdl_config.postprocess.confidence_threshold,
                                     nms_thre=self.mdl_config.postprocess.nms_threshold)

        loaded_labels_proph, yolox_preds_proph = to_prophesee(obj_labels, pred_processed)

        # For visualization, we only use the last item (per batch).
        output = {
            ObjDetOutput.LABELS_PROPH: loaded_labels_proph[-1],
            ObjDetOutput.PRED_PROPH: yolox_preds_proph[-1],
            ObjDetOutput.EV_REPR: ev_repr_selector.get_event_representations_as_list(start_idx=-1)[0],
            ObjDetOutput.SKIP_VIZ: False,
        }

        if self.started_training:
            self.mode_2_psee_evaluator[mode].add_labels(loaded_labels_proph)
            self.mode_2_psee_evaluator[mode].add_predictions(yolox_preds_proph)

        return output

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._val_test_step_impl(batch=batch, mode=Mode.VAL)

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._val_test_step_impl(batch=batch, mode=Mode.TEST)

    def run_psee_evaluator(self, mode: Mode):
        psee_evaluator = self.mode_2_psee_evaluator[mode]
        batch_size = self.mode_2_batch_size[mode]
        hw_tuple = self.mode_2_hw[mode]
        if psee_evaluator is None:
            warn(f'psee_evaluator is None in {mode=}', UserWarning, stacklevel=2)
            return
        assert batch_size is not None
        assert hw_tuple is not None
        if psee_evaluator.has_data():
            metrics = psee_evaluator.evaluate_buffer(img_height=hw_tuple[0],
                                                     img_width=hw_tuple[1])
            assert metrics is not None

            prefix = f'{mode_2_string[mode]}/'
            step = self.trainer.global_step
            log_dict = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    value = torch.tensor(v)
                elif isinstance(v, np.ndarray):
                    value = torch.from_numpy(v)
                elif isinstance(v, torch.Tensor):
                    value = v
                else:
                    raise NotImplementedError
                assert value.ndim == 0, f'tensor must be a scalar.\n{v=}\n{type(v)=}\n{value=}\n{type(value)=}'
                # put them on the current device to avoid this error: https://github.com/Lightning-AI/lightning/discussions/2529
                log_dict[f'{prefix}{k}'] = value.to(self.device)
            # Somehow self.log does not work when we eval during the training epoch.
            self.log_dict(log_dict, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            if dist.is_available() and dist.is_initialized():
                # We now have to manually sync (average the metrics) across processes in case of distributed training.
                # NOTE: This is necessary to ensure that we have the same numbers for the checkpoint metric (metadata)
                # and wandb metric:
                # - checkpoint callback is using the self.log function which uses global sync (avg across ranks)
                # - wandb uses log_metrics that we reduce manually to global rank 0
                dist.barrier()
                for k, v in log_dict.items():
                    dist.reduce(log_dict[k], dst=0, op=dist.ReduceOp.SUM)
                    if dist.get_rank() == 0:
                        log_dict[k] /= dist.get_world_size()
            if self.trainer.is_global_zero:
                # For some reason we need to increase the step by 2 to enable consistent logging in wandb here.
                # I might not understand wandb login correctly. This works reasonably well for now.
                add_hack = 2
                self.logger.log_metrics(metrics=log_dict, step=step + add_hack)

            psee_evaluator.reset_buffer()
        else:
            warn(f'psee_evaluator has not data in {mode=}', UserWarning, stacklevel=2)

    def on_train_epoch_end(self) -> None:
        mode = Mode.TRAIN
        if mode in self.mode_2_psee_evaluator and \
                self.train_metrics_config.detection_metrics_every_n_steps is None and \
                self.mode_2_hw[mode] is not None:
            # For some reason PL calls this function when resuming.
            # We don't know yet the value of train_height_width, so we skip this
            self.run_psee_evaluator(mode=mode)

    def on_validation_epoch_end(self) -> None:
        mode = Mode.VAL
        if self.started_training:
            assert self.mode_2_psee_evaluator[mode].has_data()
            self.run_psee_evaluator(mode=mode)

    def on_test_epoch_end(self) -> None:
        mode = Mode.TEST
        assert self.mode_2_psee_evaluator[mode].has_data()
        self.run_psee_evaluator(mode=mode)

    def configure_optimizers(self) -> Any:
        lr = self.train_config.learning_rate
        weight_decay = self.train_config.weight_decay

        trainable_params = [p for p in self.parameters() if p.requires_grad]

        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters found. Check adapter-only freeze settings.")

        optimizer = th.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

        scheduler_params = self.train_config.lr_scheduler
        if not scheduler_params.use:
            return optimizer

        total_steps = scheduler_params.total_steps
        assert total_steps is not None
        assert total_steps > 0

        final_div_factor_pytorch = scheduler_params.final_div_factor / scheduler_params.div_factor

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            div_factor=scheduler_params.div_factor,
            final_div_factor=final_div_factor_pytorch,
            total_steps=total_steps,
            pct_start=scheduler_params.pct_start,
            cycle_momentum=False,
            anneal_strategy='linear'
        )

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": 'learning_rate',
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
