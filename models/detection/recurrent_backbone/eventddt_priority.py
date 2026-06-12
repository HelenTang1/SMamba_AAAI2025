import torch
import torch.nn as nn
import torch.nn.functional as F


class EventDDTTokenPriorityAdapter(nn.Module):
    """
    Convert ordered EventDDT tokens into SMamba patch-level priority scores.

    Assumption:
        Earlier EventDDT tokens are more important.

    Note:
        priority_flat itself is differentiable, but if it is only used for
        top-k / argsort / index selection, detection loss cannot train this
        adapter. To train the adapter, use priority_flat in a differentiable
        path, for example feature gating in SMamba.py.
    """

    def __init__(
        self,
        stage_dims,
        token_dim=512,
        hidden_dim=128,
        detach_tokens=True,
        rank_decay=3.0,
        temperature=0.07,
        first_k_tokens=None,
        no_decay_first_ratio=0.125,
        no_decay_first_tokens=None,
        gate_strength=0.1,
    ):
        super().__init__()

        self.detach_tokens = detach_tokens
        self.rank_decay = rank_decay
        self.temperature = temperature
        self.first_k_tokens = first_k_tokens

        self.no_decay_first_ratio = no_decay_first_ratio
        self.no_decay_first_tokens = no_decay_first_tokens
        self.gate_strength = gate_strength

        self.token_proj = nn.Linear(token_dim, hidden_dim)

        self.stage_proj = nn.ModuleList([
            nn.Conv2d(stage_dim, hidden_dim, kernel_size=1)
            for stage_dim in stage_dims
        ])

        self.stage_scale = nn.Parameter(torch.ones(len(stage_dims)))

    def _make_rank_weights(self, K, device, dtype):
        """
        Make EventDDT token-rank prior.

        If no_decay_first_ratio=0.125 and K=512:
            tokens 0..63 get the same weight
            tokens 64..511 decay exponentially
        """
        if K <= 1:
            return torch.ones(K, device=device, dtype=dtype)

        if self.no_decay_first_tokens is not None:
            no_decay_n = int(self.no_decay_first_tokens)
        else:
            no_decay_n = int(round(K * float(self.no_decay_first_ratio)))

        no_decay_n = max(0, min(no_decay_n, K))

        weights = torch.ones(K, device=device, dtype=dtype)

        if no_decay_n < K:
            tail_len = K - no_decay_n

            if tail_len == 1:
                tail_pos = torch.zeros(1, device=device, dtype=dtype)
            else:
                tail_pos = torch.arange(tail_len, device=device, dtype=dtype)
                tail_pos = tail_pos / (tail_len - 1)

            weights[no_decay_n:] = torch.exp(-self.rank_decay * tail_pos)

        weights = weights / weights.sum().clamp_min(1e-6)
        return weights

    def forward(self, x, event_tokens, stage_idx: int):
        """
        x:
            SMamba stage feature map, B, C, H, W

        event_tokens:
            EventDDT tokens, B, K, D

        return:
            priority_flat, B, H*W
        """
        if event_tokens is None:
            raise RuntimeError(
                "EventDDTTokenPriorityAdapter received event_tokens=None. "
                "Check model.backbone.eventddt_encoding.enable."
            )

        if self.detach_tokens:
            event_tokens = event_tokens.detach()

        if self.first_k_tokens is not None:
            event_tokens = event_tokens[:, : self.first_k_tokens, :]

        B, C, H, W = x.shape
        K = event_tokens.shape[1]

        spatial_feat = self.stage_proj[stage_idx](x)
        spatial_feat = spatial_feat.flatten(2).transpose(1, 2)

        token_feat = self.token_proj(event_tokens)

        spatial_feat = F.normalize(spatial_feat, dim=-1)
        token_feat = F.normalize(token_feat, dim=-1)

        sim = torch.bmm(spatial_feat, token_feat.transpose(1, 2))

        rank_weights = self._make_rank_weights(
            K=K,
            device=sim.device,
            dtype=sim.dtype,
        )

        log_rank_weights = torch.log(rank_weights.clamp_min(1e-6))
        weighted_logits = sim / self.temperature + log_rank_weights.view(1, 1, K)

        priority_flat = torch.logsumexp(weighted_logits, dim=-1)
        priority_flat = F.softplus(priority_flat * self.stage_scale[stage_idx]) + 1e-6

        return priority_flat