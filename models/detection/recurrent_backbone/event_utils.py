import torch


def smamba_ev_to_eventddt_perbin(ev_tensor: torch.Tensor):
    """
    Convert SMamba event tensor to EventDDT perbin format.

    Assumed GENX / SMamba layout:
        ev_tensor: B, 20, H, W
        channels: [pol0_bin0...pol0_bin9, pol1_bin0...pol1_bin9]

    Return:
        perbin: B*10, 2, H, W
        offsets: B+1
        n_bins: B
    """
    B, C, H, W = ev_tensor.shape
    assert C % 2 == 0, f"Expected 2*n_bins channels, got {C}"

    n_bins = C // 2

    # B, 20, H, W
    # -> B, 2, 10, H, W
    # -> B, 10, 2, H, W
    perbin = ev_tensor.float().view(B, 2, n_bins, H, W)
    perbin = perbin.permute(0, 2, 1, 3, 4).contiguous()
    perbin = perbin.view(B * n_bins, 2, H, W)

    offsets = torch.arange(
        0,
        (B + 1) * n_bins,
        step=n_bins,
        device=ev_tensor.device,
        dtype=torch.long,
    )

    n_bins_tensor = torch.full(
        (B,),
        fill_value=n_bins,
        device=ev_tensor.device,
        dtype=torch.long,
    )

    return perbin, offsets, n_bins_tensor