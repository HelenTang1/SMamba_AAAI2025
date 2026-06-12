from typing import Dict, Optional, Tuple
import torch as th
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None
from data.utils.types import FeatureMap, BackboneFeatures, LstmState, LstmStates
from models.layers.rnn import DWSConvLSTM2d
from models.layers.maxvit.maxvit import (
    PartitionAttentionCl,
    nhwC_2_nChw,
    get_downsample_layer_Cf2Cl,
    PartitionType)
from .base import BaseDetector
import copy
from typing import Optional, Callable, Any
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from models.detection.recurrent_backbone.utils import *
from models.detection.recurrent_backbone.eventddt_priority import EventDDTTokenPriorityAdapter

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

class RNNDetector(BaseDetector):
    def __init__(self, mdl_config: DictConfig):
        super().__init__()
        ######## VSS config ########
        ssm_act_layer="silu"
        mlp_act_layer="gelu"
        # ===========================
        drop_path_rate=0.2
        patch_norm=True
        norm_layer="LN2D"
        downsample_version= "v3"
        patchembed_version = "v2"
        use_checkpoint=False  
        # =========================
        posembed=False
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])

        ###### Config ######
        # k in Sparse SS2D
        self.gaussian_kernel = mdl_config.gaussian_kernel
        # beta in STCA
        self.factor = mdl_config.factor
        self.hw = mdl_config.resolution_hw
        ###### Compile if requested ######
        compile_cfg = mdl_config.get('compile', None)
        if compile_cfg is not None:
            compile_mdl = compile_cfg.enable
            if compile_mdl and th_compile is not None:
                compile_args = OmegaConf.to_container(compile_cfg.args, resolve=True, throw_on_missing=True)
                self.forward = th_compile(self.forward, **compile_args)
            elif compile_mdl:
                print('Could not compile backbone because torch.compile is not available')
        ##################################

        patch_size=4
        in_chans=20
        depths=[2, 2, 2, 2]
        num_stages = len(depths)
        assert num_stages == 4
        dims=[64, 128, 256, 512]
        priority_cfg = mdl_config.get("eventddt_priority", None)
        self.use_eventddt_priority = (
            priority_cfg is not None and priority_cfg.get("enable", False)
        )

        if self.use_eventddt_priority:
            self.eventddt_priority = EventDDTTokenPriorityAdapter(
                stage_dims=dims,
                token_dim=priority_cfg.get("token_dim", 512),
                hidden_dim=priority_cfg.get("hidden_dim", 128),
                detach_tokens=priority_cfg.get("detach_tokens", True),
                rank_decay=priority_cfg.get("rank_decay", 3.0),
                temperature=priority_cfg.get("temperature", 0.07),
                first_k_tokens=priority_cfg.get("first_k_tokens", None),
                no_decay_first_ratio=priority_cfg.get("no_decay_first_ratio", 0.125),
                no_decay_first_tokens=priority_cfg.get("no_decay_first_tokens", None),
                gate_strength=priority_cfg.get("gate_strength", 0.1),
            )
        else:
            self.eventddt_priority = None
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_layers = len(depths)
        self.num_features = dims[-1]
        self.dims = dims
        self.stage_dims = dims
        self.strides = [4,8,16,32]
        # AvgPool in STCA
        self.time_embed = self._make_time_embed(patch_size)

        ######## VSS config ########
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        self.pos_embed = self._pos_embed(dims[0], patch_size, self.hw[0], self.hw[1]) if posembed else None
        _make_patch_embed = dict(
            v1=self._make_patch_embed, 
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer, channel_first=self.channel_first)
        
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_layers):
            stage = RNNDetectorStage(i_layer=stage_idx,
                                     hw = self.hw,
                                     norm_layer=norm_layer,
                                     channel_first=self.channel_first,
                                     depths=depths,
                                     use_checkpoint=use_checkpoint,
                                     dims = self.dims,
                                     dpr= self.dpr,
                                     stage_cfg=mdl_config.stage)
            self.stages.append(stage)

        self.num_stages = num_stages
        self.apply(self._init_weights)

        self.index_down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.maxpl = nn.MaxPool2d(kernel_size=4, stride=4)
        self.unspamle= nn.UpsamplingNearest2d(scale_factor=4)
    
    def get_stage_dims(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0, stage_indices
        assert max(stage_indices) < len(self.stages), stage_indices
        return tuple(self.stage_dims[stage_idx] for stage_idx in stage_indices)

    def get_strides(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0, stage_indices
        assert max(stage_indices) < len(self.stages), stage_indices
        return tuple(self.strides[stage_idx] for stage_idx in stage_indices)
    
    @staticmethod
    def _make_time_embed(patch_size=4):
        return nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)
    
    @staticmethod
    def _pos_embed(embed_dims, patch_size, h, w):
        patch_height, patch_width = (h // patch_size, w // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )
    def _apply_eventddt_priority_gate(
        self,
        x: torch.Tensor,
        priority_flat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable path from detection loss to EventDDT priority adapter.

        priority_flat is still used for top-k / sorting, which is not differentiable.
        This gate makes priority_flat also affect the feature values directly.
        """
        if self.eventddt_priority is None:
            return x

        gate_strength = getattr(self.eventddt_priority, "gate_strength", 0.0)

        if gate_strength is None or gate_strength <= 0:
            return x

        B, C, H, W = x.shape

        priority = priority_flat.float()
        priority = priority - priority.mean(dim=1, keepdim=True)
        priority = priority / priority.std(dim=1, keepdim=True).clamp_min(1e-6)

        gate = 1.0 + float(gate_strength) * torch.tanh(priority)
        gate = gate.view(B, 1, H, W).to(dtype=x.dtype, device=x.device)

        return x * gate
    def _make_sparse_scan_indices_from_priority(
        self,
        priority_flat: torch.Tensor,
        sort_priority_flat: torch.Tensor,
    ):
        """
        Convert priority scores into SMamba sparse scanning indices.

        Args:
            priority_flat:
                B, H*W, used to select kept tokens.

            sort_priority_flat:
                B, H*W, used to decide local scan order.

        Returns:
            index_token:
                B, 1, L_keep

            indices:
                B, 1, L_keep
        """
        B, L = priority_flat.shape

        # Same threshold logic as original SMamba.
        # Original:
        # cc = sum(x_time) / (H*W * factor)
        cc = torch.sum(priority_flat, dim=1) / (L * self.factor)

        keep_mask = priority_flat >= cc[:, None]
        K = torch.sum(keep_mask, dim=1).clamp(min=1, max=L)

        max_k = int(K.max().item())

        # Index of kept tokens
        index_token = torch.topk(
            priority_flat,
            k=max_k,
            dim=1,
            largest=True,
            sorted=False,
        )[1]
        index_token.requires_grad_(False)

        # Information-prioritized local sorting
        selected_sort_score = sort_priority_flat.gather(dim=1, index=index_token)
        indices = torch.argsort(selected_sort_score, dim=1)
        indices.requires_grad_(False)

        index_token = index_token.unsqueeze(1)
        indices = indices.unsqueeze(1)

        return index_token, indices

    def forward(self, x: th.Tensor, prev_states: Optional[LstmStates] = None, 
                token_mask: Optional[th.Tensor] = None, 
                eventddt_tokens: Optional[th.Tensor] = None) -> Tuple[BackboneFeatures, LstmStates]:
        # Temporal Continuity Assessment
        # B,C,H,W
        # GENX / SMamba event layout:
        #   [pol0_bin0, ..., pol0_bin9, pol1_bin0, ..., pol1_bin9]
        #
        # Therefore the temporal weights should be:
        #   [1,2,...,10, 1,2,...,10]
        B, C, H, W = x.shape
        assert C % 2 == 0, f"Expected 2 * n_bins channels, got {C}"

        n_bins = C // 2

        temporal_weights = torch.arange(
            1,
            n_bins + 1,
            device=x.device,
            dtype=x.dtype,
        )

        temporal_weights = torch.cat(
            [temporal_weights, temporal_weights],
            dim=0,
        ).view(1, C, 1, 1)

        x_time = (x * temporal_weights).sum(dim=1, keepdim=True)
        # Average pooling
        # B,1,H,W
        x_time = self.time_embed(x_time)
        B, C, H, W = x_time.shape

        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
            x = x + pos_embed

        if prev_states is None:
            prev_states = [None] * self.num_stages
        assert len(prev_states) == self.num_stages
        states: LstmStates = list()
        output: Dict[int, FeatureMap] = {}
        for stage_idx, stage in enumerate(self.stages):
            if stage_idx == 0:
                # Spatial Continuity Assessment
                blur_layer = get_gaussian_kernel(kernel_size=self.gaussian_kernel).to(device=x_time.device, dtype=x_time.dtype)
                x_time = blur_layer(x_time)
            if stage_idx == 0 and not hasattr(self, "_printed_eventddt_debug"):
                print(
                    "[SMamba EventDDT debug]",
                    "use_eventddt_priority=", self.use_eventddt_priority,
                    "eventddt_tokens is None=", eventddt_tokens is None,
                    "eventddt_tokens shape=", None if eventddt_tokens is None else eventddt_tokens.shape,
                )
                self._printed_eventddt_debug = True

            if self.use_eventddt_priority:
                if eventddt_tokens is None:
                    raise RuntimeError(
                        "model.backbone.eventddt_priority.enable=True, "
                        "but eventddt_tokens is None. "
                        "Check model.backbone.eventddt_encoding.enable."
                    )
                # x is current SMamba stage feature map: B, C, H, W
                Bx, Cx, Hx, Wx = x.shape

                # EventDDT-token-guided spatial priority: B, Hx*Wx
                priority_flat = self.eventddt_priority(
                    x=x,
                    event_tokens=eventddt_tokens,
                    stage_idx=stage_idx,
                )

                # Differentiable path for training the EventDDT priority adapter.
                # This does not replace top-k / argsort; it only lets detection loss
                # train the priority adapter through feature modulation.
                x = self._apply_eventddt_priority_gate(
                    x=x,
                    priority_flat=priority_flat,
                )

                priority_map = priority_flat.view(Bx, 1, Hx, Wx)

                if stage_idx in [0, 1]:
                    # Keep SMamba's original local-window sorting idea,
                    # but use EventDDT-guided priority instead of x_time.
                    sort_priority_map = self.maxpl(priority_map)
                    sort_priority_map = self.unspamle(sort_priority_map)

                    # Safety in case H/W is not exactly divisible by 4.
                    if sort_priority_map.shape[-2:] != (Hx, Wx):
                        sort_priority_map = F.interpolate(
                            sort_priority_map,
                            size=(Hx, Wx),
                            mode="nearest",
                        )

                    sort_priority_flat = sort_priority_map.flatten(2, 3).squeeze(1)
                else:
                    sort_priority_flat = priority_flat

                index_token, indices = self._make_sparse_scan_indices_from_priority(
                    priority_flat=priority_flat,
                    sort_priority_flat=sort_priority_flat,
                )

            else:
                # Original SMamba handcrafted priority.
                if stage_idx in [0, 1]:
                    x_time2 = self.maxpl(x_time)
                    x_time2 = self.unspamle(x_time2)
                    x_time2 = x_time2.flatten(2, 3)
                else:
                    x_time2 = x_time.flatten(2, 3)

                x_time_flat = x_time.flatten(2, 3).squeeze(1)

                index_token, indices = self._make_sparse_scan_indices_from_priority(
                    priority_flat=x_time_flat,
                    sort_priority_flat=x_time2.squeeze(1),
                )
            
            x, state, x_fpn = stage(x, index_token, indices, prev_states[stage_idx])

            states.append(state)
            stage_number = stage_idx + 1
            output[stage_number] = x_fpn

            if not self.use_eventddt_priority:
                x_time = self.index_down(x_time)
                B, C, H, W = x_time.shape
        return output, states

class RNNDetectorStage(nn.Module):
    """Operates with NCHW [channel-first] format as input and output.
    """
    def __init__(self,
                 i_layer=None, 
                 hw=None,
                 norm_layer=None,
                 channel_first=None,
                 depths=None,
                 use_checkpoint=None,
                 dims = None,
                 dpr = None,
                 stage_cfg=None): #: DictConfig):
        super().__init__()
        self.num_layers = len(depths)
        self.dims = dims
        self.channel_first = channel_first
        self.dpr = dpr
        ######## VSS config ########
        downsample_version= "v3"
        ssm_d_state=1
        ssm_ratio=2.0
        ssm_dt_rank="auto"
        ssm_act_layer="silu"       
        ssm_conv=3
        ssm_conv_bias=False
        ssm_drop_rate=0.0
        ssm_init="v0"
        forward_type="v05_noz"
        # =========================
        mlp_ratio=4.0
        mlp_act_layer="gelu"
        mlp_drop_rate=0.0
        gmlp=False
        # =========================
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)
        
        _make_downsample = dict(
            v1=PatchMerging2D, 
            v2=self._make_downsample, 
            v3=self._make_downsample_v3, 
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.downsample = _make_downsample(
            self.dims[i_layer], 
            self.dims[i_layer+1], 
            norm_layer=norm_layer,
            channel_first=self.channel_first,
        ) if (i_layer < self.num_layers - 1) else nn.Identity()

        self.smambalayer = self._make_layer(
                    i_layer = i_layer,
                    hw = hw,
                    drop_path = self.dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    use_checkpoint=use_checkpoint,
                    norm_layer=norm_layer,
                    downsample=self.downsample,
                    channel_first=self.channel_first,
                    # =================
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    # =================
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    gmlp=gmlp,
                    )

        ##### LSTM config #####
        lstm_cfg = stage_cfg.lstm
        self.lstm = DWSConvLSTM2d(dim=self.dims[i_layer],
                                  dws_conv=lstm_cfg.dws_conv,
                                  dws_conv_only_hidden=lstm_cfg.dws_conv_only_hidden,
                                  dws_conv_kernel_size=lstm_cfg.dws_conv_kernel_size,
                                  cell_update_dropout=lstm_cfg.get('drop_cell_update', 0))

    def forward(self, x: th.Tensor, index_token: th.Tensor, indices: th.Tensor,
                h_and_c_previous: Optional[LstmState] = None) \
            -> Tuple[FeatureMap, FeatureMap, LstmState]:
        x, index_token, indices = self.smambalayer((x, index_token, indices))
        h_c_tuple = self.lstm(x, h_and_c_previous)
        x_fpn = h_c_tuple[0]
        x = self.downsample(x_fpn)
        return x, h_c_tuple, x_fpn
    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        i_layer=96, 
        hw = None,
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            if i_layer in [0, 1]:
                blocks.append(SSM(
                    idx=i_layer, 
                    drop_path=drop_path[d],
                    norm_layer=norm_layer,
                    channel_first=channel_first,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    gmlp=gmlp,
                    use_checkpoint=use_checkpoint,
                ))
            else:
                blocks.append(SCMM(
                    idx=i_layer,
                    hw = hw, 
                    drop_path=drop_path[d],
                    norm_layer=norm_layer,
                    channel_first=channel_first,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    gmlp=gmlp,
                    use_checkpoint=use_checkpoint,
                ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
        ))

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        if check_name("pos_embed", strict=True):
            srcEmb: torch.Tensor = state_dict[prefix + "pos_embed"]
            state_dict[prefix + "pos_embed"] = F.interpolate(srcEmb.float(), size=self.pos_embed.shape[2:4], align_corners=False, mode="bicubic").to(srcEmb.device)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
