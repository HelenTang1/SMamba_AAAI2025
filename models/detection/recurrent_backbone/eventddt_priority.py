import torch
import torch.nn as nn
import torch.nn.functional as F


class EventDDTTokenPriorityAdapter(nn.Module):
    """
    Convert ordered EventDDT encoder output tokens into spatial priority scores
    for SMamba sparse scanning.

    Important assumption:
        Earlier EventDDT tokens are more important.

    x:
        SMamba stage feature map, shape (B, C, H, W)

    event_tokens:
        EventDDT encoder output tokens, shape (B, K, D)

    output:
        priority_flat, shape (B, H*W)
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
    ):
        super().__init__()
        self.detach_tokens = detach_tokens
        self.rank_decay = rank_decay
        self.temperature = temperature
        self.first_k_tokens = first_k_tokens

        self.token_proj = nn.Linear(token_dim, hidden_dim)

        self.stage_proj = nn.ModuleList([
            nn.Conv2d(stage_dim, hidden_dim, kernel_size=1)
            for stage_dim in stage_dims
        ])

        self.stage_scale = nn.Parameter(torch.ones(len(stage_dims)))

    def _make_rank_weights(self, K, device, dtype):
        """
        Larger weight for earlier tokens.

        token 0 gets weight close to 1.
        later tokens decay exponentially.
        """
        if K == 1:
            return torch.ones(1, device=device, dtype=dtype)

        pos = torch.arange(K, device=device, dtype=dtype)
        pos = pos / (K - 1)

        weights = torch.exp(-self.rank_decay * pos)
        weights = weights / weights.sum().clamp_min(1e-6)

        return weights

    def forward(self, x, event_tokens, stage_idx: int):
        """
        x:
            B, C, H, W

        event_tokens:
            B, K, D
        """
        if self.detach_tokens:
            event_tokens = event_tokens.detach()

        if self.first_k_tokens is not None:
            event_tokens = event_tokens[:, : self.first_k_tokens, :]

        B, C, H, W = x.shape
        K = event_tokens.shape[1]

        # B, H*W, hidden_dim
        spatial_feat = self.stage_proj[stage_idx](x)
        spatial_feat = spatial_feat.flatten(2).transpose(1, 2)

        # B, K, hidden_dim
        token_feat = self.token_proj(event_tokens)

        spatial_feat = F.normalize(spatial_feat, dim=-1)
        token_feat = F.normalize(token_feat, dim=-1)

        # B, H*W, K
        sim = torch.bmm(spatial_feat, token_feat.transpose(1, 2))

        # K
        rank_weights = self._make_rank_weights(
            K=K,
            device=sim.device,
            dtype=sim.dtype,
        )

        # Add token-rank prior into similarity.
        # Earlier tokens have larger log prior.
        log_rank_weights = torch.log(rank_weights.clamp_min(1e-6))
        weighted_logits = sim / self.temperature + log_rank_weights.view(1, 1, K)

        # B, H*W
        priority_flat = torch.logsumexp(weighted_logits, dim=-1)

        # Keep it positive for SMamba thresholding.
        priority_flat = F.softplus(priority_flat * self.stage_scale[stage_idx]) + 1e-6

        return priority_flat