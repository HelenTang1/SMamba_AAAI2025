import sys
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from models.detection.recurrent_backbone.event_utils import (
    smamba_ev_to_eventddt_perbin,
)


@contextmanager
def eventddt_import_context(repo_root: str):
    """
    EventDDT and SMamba both use top-level package names like `models`.
    This context temporarily switches import resolution to EventDDT.
    """
    repo_root = str(Path(repo_root).resolve())

    prefixes = ("models", "dataset", "mimogpt")

    old_sys_path = list(sys.path)

    backup_modules = {}
    for name in list(sys.modules.keys()):
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            backup_modules[name] = sys.modules.pop(name)

    sys.path.insert(0, repo_root)

    try:
        yield
    finally:
        for name in list(sys.modules.keys()):
            if any(name == p or name.startswith(p + ".") for p in prefixes):
                sys.modules.pop(name)

        sys.modules.update(backup_modules)
        sys.path[:] = old_sys_path


def _resolve_eventddt_path(repo_root: str, path: Optional[str]) -> Optional[str]:
    if path is None:
        return None

    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)

    return str(Path(repo_root) / path_obj)


def _load_eventddt_cfg(
    repo_root: str,
    tokenizer_config_path: str,
    experiment_config_path: str,
    dataset_config_path: Optional[str] = None,
    eventddt_cfg_override: Optional[Dict[str, Any]] = None,
    disable_decoder_pretrained: bool = True,
):
    tokenizer_config_path = _resolve_eventddt_path(repo_root, tokenizer_config_path)
    experiment_config_path = _resolve_eventddt_path(repo_root, experiment_config_path)
    dataset_config_path = _resolve_eventddt_path(repo_root, dataset_config_path)

    cfg = OmegaConf.load(tokenizer_config_path)

    if dataset_config_path is not None:
        dataset_cfg = OmegaConf.load(dataset_config_path)
        cfg = OmegaConf.merge(cfg, {"dataset": dataset_cfg})

    if experiment_config_path is not None:
        exp_cfg = OmegaConf.load(experiment_config_path)
        cfg = OmegaConf.merge(cfg, exp_cfg)

    if eventddt_cfg_override is not None:
        cfg = OmegaConf.merge(cfg, eventddt_cfg_override)

    # Do not resume EventDDT Lightning training.
    cfg.model.ckpt_path = None

    if disable_decoder_pretrained:
        # We only need EventTokenizer.prepare_tokenizer_input_and_xt + encoding.
        # For exp2_Event_Scratch_rev.yaml, selftok_pretrained_path is already null.
        cfg.tokenizer.selftok_pretrained_path = None
        cfg.tokenizer.image_vae_sd3_path = None

    return cfg


class EventDDTEncodingBridge(nn.Module):
    """
    Use EventDDT EventTokenizer to encode SMamba event tensors.

    SMamba input:
        B, 20, H, W

    EventDDT path:
        B, 20, H, W
            -> perbin / offsets / n_bins
            -> EventTokenizer.prepare_tokenizer_input_and_xt(..., t=tokenizer_t)
            -> EventTokenizer.encoding(...)
            -> tokens, B, K, D
    """

    def __init__(
        self,
        enable: bool = True,
        repo_root: Optional[str] = None,
        tokenizer_config_path: str = "configs/tokenizer_config.yaml",
        experiment_config_path: Optional[str] = None,
        dataset_config_path: Optional[str] = "configs/dataset/DSEC.yaml",
        ckpt_path: Optional[str] = None,
        freeze: bool = True,
        strict_load: bool = True,
        disable_decoder_pretrained: bool = True,
        tokenizer_t: float = 0.5,
        output_dtype: str = "input",
        eventddt_cfg_override: Optional[Dict[str, Any]] = None,
        **unused,
    ):
        super().__init__()

        if not enable:
            raise ValueError("EventDDTEncodingBridge should only be built when enable=True.")

        if repo_root is None:
            raise ValueError("eventddt_encoding.repo_root must be set.")

        if experiment_config_path is None:
            raise ValueError("eventddt_encoding.experiment_config_path must be set.")

        self.repo_root = str(Path(repo_root).resolve())
        self.tokenizer_t = tokenizer_t
        self.output_dtype = output_dtype
        self.freeze = freeze

        cfg = _load_eventddt_cfg(
            repo_root=self.repo_root,
            tokenizer_config_path=tokenizer_config_path,
            experiment_config_path=experiment_config_path,
            dataset_config_path=dataset_config_path,
            eventddt_cfg_override=eventddt_cfg_override,
            disable_decoder_pretrained=disable_decoder_pretrained,
        )

        with eventddt_import_context(self.repo_root):
            from models.build_model import (
                build_tokenizer_model,
                load_tokenizer_encoder_only,
            )
            from models.event_compact_tokenizer import EventCompactTokenizer

            event_tokenizer = build_tokenizer_model(cfg, encoder_only=True)
            if isinstance(event_tokenizer, EventCompactTokenizer):
                include_event_pretrained = True
            else:
                include_event_pretrained = False
            

            if ckpt_path is not None:
                ckpt_path = _resolve_eventddt_path(self.repo_root, ckpt_path)

                load_tokenizer_encoder_only(
                    tokenizer=event_tokenizer,
                    ckpt_path=ckpt_path,
                    strict=strict_load,
                    freeze_encoder=self.freeze,
                    include_event_pretrained=include_event_pretrained,
                )
        self.event_tokenizer = event_tokenizer

        if freeze:
            self.event_tokenizer.eval()
            for p in self.event_tokenizer.parameters():
                p.requires_grad = False

        print("[EventDDTEncodingBridge] ready")
        print(f"  repo_root: {self.repo_root}")
        print(f"  experiment_config_path: {experiment_config_path}")
        print(f"  tokenizer_t: {self.tokenizer_t}")
        print(f"  n_bins: {self.n_bins}")
        print(f"  event_layout: {self.event_layout}")


    def forward(self, ev_tensor: torch.Tensor) -> torch.Tensor:
        input_dtype = ev_tensor.dtype

        perbin, offsets, n_bins = smamba_ev_to_eventddt_perbin(ev_tensor=ev_tensor)

        # EventTokenizer.encoding() itself uses torch.no_grad().
        _, tokenizer_input, _ = self.event_tokenizer.prepare_tokenizer_input_and_xt(
            x0=None, x1=None,
            perbin=perbin, offsets=offsets, n_bins=n_bins,
            t=self.tokenizer_t, average_all_bins=True
        )

        tokens = self.event_tokenizer.encoding(tokenizer_input)

        if self.output_dtype == "input":
            tokens = tokens.to(dtype=input_dtype)
        elif self.output_dtype == "float32":
            tokens = tokens.float()
        else:
            raise ValueError(f"Unknown output_dtype: {self.output_dtype}")

        return tokens