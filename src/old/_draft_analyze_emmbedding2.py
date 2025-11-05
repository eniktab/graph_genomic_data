# ============================================================
# DNA Embedding Path Search (max separation across sequences)
# - Position-aware (not "time")
# - Directional position weighting
# - Pooling along position and embedding channels
# - Per-layer and last-k layer mixes
# - Separation-based ranking (min/mean cosine & L2), RC, locality margin
# - NaN/Inf-safe reductions (no torch.nanmin/torch.nanmax used)
# ============================================================

from typing import List, Dict, Tuple, Optional, Iterable, Union
from pathlib import Path
import math
import csv

import torch
import torch.nn.functional as F

# ------------------------------------------------------------
# NaN/Inf-safe helpers (no torch.nan* APIs required)
# ------------------------------------------------------------

def _to_tensor_f32(x):
    return torch.as_tensor(x, dtype=torch.float32)

def _finite_mask(t: torch.Tensor):
    return torch.isfinite(t)

def _finite_minmax(vals_list):
    """Return (min, max) over finite values; if none finite -> (nan, nan)."""
    t = _to_tensor_f32(vals_list)
    msk = _finite_mask(t)
    if not msk.any():
        return float("nan"), float("nan")
    t = t[msk]
    return float(t.min().item()), float(t.max().item())

def _finite_mean_list(vals_list):
    """Mean over finite values in a python list; if none finite -> nan."""
    t = _to_tensor_f32(vals_list)
    msk = _finite_mask(t)
    if not msk.any():
        return float("nan")
    return float(t[msk].mean().item())

def _safe_finite(x, fallback=-float("inf")):
    """Return x if finite else fallback; useful for sort keys."""
    try:
        xv = float(x)
    except Exception:
        return fallback
    return xv if math.isfinite(xv) else fallback


# ------------------------------------------------------------
# Tokenization & model forward (expects HyenaBackend available)
# ------------------------------------------------------------

@torch.no_grad()
def _tokenize(backend, seq: str, rc: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize one DNA sequence; return input_ids [1, T] and mask [1, T]."""
    if rc:
        tbl = str.maketrans("ACGTNacgtn","TGCANtgcan")
        seq = seq.translate(tbl)[::-1]
    tok = backend.tokenizer(
        [seq],
        padding="longest",
        truncation=True,
        max_length=backend.max_length,
        pad_to_multiple_of=backend._choose_pad_multiple(backend.max_length),
        return_tensors="pt",
    )
    input_ids = tok["input_ids"].to(backend.device, non_blocking=True)
    mask = (input_ids != int(backend.tokenizer.pad_token_id))
    return input_ids, mask


@torch.no_grad()
def _forward_all_layers(backend, input_ids: torch.Tensor, mask: torch.Tensor, rc: bool):
    """
    Run model; return list of hidden states [L][1, T, H] and aligned mask [1, T].
    If rc=True, reverse the position axis so forward and RC align positionally.
    """
    try:
        out = backend.model(input_ids)
    except TypeError:
        out = backend.model(input_ids=input_ids)

    if not (hasattr(out, "hidden_states") and out.hidden_states):
        hs_list = [out.last_hidden_state] if hasattr(out, "last_hidden_state") else []
    else:
        hs_list = list(out.hidden_states)  # includes embedding layer at idx 0 (HF-style)

    if rc:
        hs_list = [torch.flip(hs, dims=[1]) for hs in hs_list]
        mask = torch.flip(mask, dims=[1])

    return hs_list, mask  # [L][1, T, H], [1, T]


# ------------------------------------------------------------
# Position utilities & directional weights
# ------------------------------------------------------------

def _pos_indices(mask_bt: torch.Tensor):
    """Return position index per token (pads→-1) and per-sequence content length."""
    B, T = mask_bt.shape
    c = torch.cumsum(mask_bt.to(torch.int64), dim=1)
    pos_bt = (c - 1).masked_fill(mask_bt == 0, -1)
    len_b = c[:, -1]
    return pos_bt, len_b


def _tile_ranges(L: int, tile_bp: int) -> List[Tuple[int, int]]:
    out, i = [], 0
    while i < L:
        j = min(L, i + tile_bp)
        out.append((i, j)); i = j
    return out


def _make_weights(
    mask_bt: torch.Tensor,
    mode: str,
    *,
    tau: Optional[float] = None,
    tail_k: Optional[int] = None,
    head_k: Optional[int] = None
) -> torch.Tensor:
    """
    Direction-aware position weights [B, T] (non-negative; pads are 0).
    """
    B, T = mask_bt.shape
    pos_bt, len_b = _pos_indices(mask_bt)
    w = torch.zeros_like(mask_bt, dtype=torch.float32)

    if mode == "mean":
        w = mask_bt.to(torch.float32)

    elif mode in ("exp_left","exp_right"):
        if not tau or tau <= 0: raise ValueError("tau>0 required")
        Lbt = torch.gather(len_b[:, None].expand(B, T), 1, (pos_bt.clamp_min(0)))
        if mode == "exp_left":
            expo = -pos_bt.to(torch.float32) / tau
        else:
            expo = -(Lbt.to(torch.float32) - 1 - pos_bt.to(torch.float32)) / tau
        expo = torch.where(pos_bt >= 0, expo, torch.full_like(expo, -1e9))
        w = torch.exp(expo)

    elif mode == "head_k":
        if not head_k or head_k <= 0: raise ValueError("head_k>0 required")
        w = ((pos_bt >= 0) & (pos_bt < head_k)).to(torch.float32)

    elif mode == "tail_k":
        if not tail_k or tail_k <= 0: raise ValueError("tail_k>0 required")
        Lbt = torch.gather(len_b[:, None].expand(B, T), 1, (pos_bt.clamp_min(0)))
        w = ((pos_bt >= (Lbt - tail_k)) & (pos_bt >= 0)).to(torch.float32)

    elif mode == "linear_left":
        Lbt = torch.gather(len_b[:, None].expand(B, T), 1, (pos_bt.clamp_min(0))).to(torch.float32)
        w = torch.where(pos_bt >= 0, (Lbt - pos_bt.to(torch.float32)), torch.zeros_like(Lbt))

    elif mode == "linear_right":
        w = torch.where(pos_bt >= 0, (pos_bt.to(torch.float32) + 1.0), torch.zeros_like(pos_bt, dtype=torch.float32))

    else:
        raise ValueError(f"Unknown weighting mode: {mode}")

    return w * mask_bt.to(torch.float32)


# ------------------------------------------------------------
# Pooling along position & channels (+ layer mixing)
# ------------------------------------------------------------

def _pool_position(hs_bth: torch.Tensor, mask_bt: torch.Tensor, weights_bt: torch.Tensor) -> torch.Tensor:
    """Pool over the **position** axis: [B, T, H] -> [B, H], L2-normalized."""
    w = weights_bt.unsqueeze(-1)
    num = (hs_bth.float() * w).sum(dim=1)
    den = weights_bt.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return F.normalize(num / den, p=2, dim=1).to(torch.float32)


def _mean_layers_token(hs_l_bth: List[torch.Tensor], which: Iterable[int]) -> torch.Tensor:
    """Uniform mean over chosen layers at the **token** level: [B, T, H]."""
    acc = None; c = 0
    for li in which:
        x = hs_l_bth[li].float()
        acc = x if acc is None else (acc + x)
        c += 1
    return acc / max(c, 1)


def _position_then_layers(hs_l_bth: List[torch.Tensor], mask_bt: torch.Tensor, wpos_bt: torch.Tensor, which: Iterable[int]) -> torch.Tensor:
    """Pool each chosen layer over position -> [B, H], then mean across layers -> [B, H]."""
    pooled = [_pool_position(hs_l_bth[li], mask_bt, wpos_bt) for li in which]
    out = torch.stack(pooled, dim=0).mean(dim=0)
    return F.normalize(out, p=2, dim=1).to(torch.float32)


def _chunk_channels(x_bth: torch.Tensor, K: int) -> List[torch.Tensor]:
    """Split the channel (H) dimension into K nearly equal chunks."""
    return torch.chunk(x_bth, chunks=K, dim=-1)


def _pool_channels_groupwise(hs_bth: torch.Tensor, K: int, p: float = 3.0) -> torch.Tensor:
    """
    Groupwise GeM over channels: [B, T, H] -> [B, T, K].
    Each of the K groups is pooled into 1 channel via GeM (power-mean) with exponent p.
    """
    groups = _chunk_channels(hs_bth, K)
    eps = 1e-6
    outs = []
    for g in groups:
        g = g.float().clamp_min(eps)
        out = (g ** p).mean(dim=-1).clamp_min(eps) ** (1.0 / p)  # [B, T]
        outs.append(out.unsqueeze(-1))  # [B, T, 1]
    return torch.cat(outs, dim=-1)  # [B, T, K]


def _pool_channels_after_position(vec_bh: torch.Tensor, K: int, p: float = 3.0) -> torch.Tensor:
    """
    After position pooling to [B, H], pool channels into K dims via groupwise GeM: [B, H] -> [B, K].
    """
    chunks = torch.chunk(vec_bh, chunks=K, dim=-1)
    eps = 1e-6
    outs = []
    for c in chunks:
        c = c.float().clamp_min(eps)
        out = (c ** p).mean(dim=-1, keepdim=True).clamp_min(eps) ** (1.0 / p)  # [B, 1]
        outs.append(out)
    out = torch.cat(outs, dim=-1)  # [B, K]
    return F.normalize(out, p=2, dim=1).to(torch.float32)


# ------------------------------------------------------------
# Between-sequence distances (cosine & L2)
# ------------------------------------------------------------

def _pairwise_nonself_metrics(X: torch.Tensor) -> Dict[str, float]:
    """
    X: [N, D] pooled embeddings for N different sequences (forward only).
    Returns mean/min cosine distance and L2 across i != j, plus nearest-nonself stats.
    """
    N = X.shape[0]
    if N < 2:
        return dict(mean_cos_dist=float("nan"), min_cos_dist=float("nan"),
                    mean_L2=float("nan"), min_L2=float("nan"),
                    nn_cos_sim=float("nan"), nn_L2=float("nan"))
    # Cosine distance = 1 - cosine_similarity
    Xn = F.normalize(X, p=2, dim=1)
    cos_sim = Xn @ Xn.T                      # [N, N]
    cos_dist = 1.0 - cos_sim
    eye = torch.eye(N, dtype=torch.bool, device=X.device)
    cos_dist.masked_fill_(eye, float("inf"))  # exclude self when taking minima

    # nearest non-self stats
    nn_cos_sim = (1.0 - cos_dist.min(dim=1).values).mean().item()

    # L2 distances
    xx = (X**2).sum(dim=1, keepdim=True)     # [N,1]
    L2sq = xx + xx.T - 2.0 * (X @ X.T)
    L2sq = L2sq.clamp_min(0.0)
    L2 = torch.sqrt(L2sq + 1e-12)
    L2.masked_fill_(eye, float("inf"))
    nn_L2 = L2.min(dim=1).values.mean().item()

    mask = ~eye
    cos_vals = cos_dist[mask]
    l2_vals = L2[mask]

    return dict(
        mean_cos_dist=float(cos_vals.mean().item()),
        min_cos_dist=float(cos_vals.min().item()),
        mean_L2=float(l2_vals.mean().item()),
        min_L2=float(l2_vals.min().item()),
        nn_cos_sim=float(nn_cos_sim),
        nn_L2=float(nn_L2)
    )


# ------------------------------------------------------------
# Per-sequence scoring for a single configuration
# ------------------------------------------------------------

@torch.no_grad()
def _score_seq_config(
    backend,
    seq: str,
    *,
    dir_mode: str,
    dir_params: Dict[str, float],
    pooling_axis: str,            # 'position' | 'layers→position' | 'position→layers' | 'channels→position[K]' | 'position→channels[K]'
    layer_spec: Union[int, Tuple[str,int]],
    tile_bp: int,
    far_min_gap_bp: int
) -> Dict[str, float]:
    """Score one configuration on a single sequence (RC, locality, etc.)."""
    ids_f, m_f = _tokenize(backend, seq, rc=False)
    ids_r, m_r = _tokenize(backend, seq, rc=True)

    hs_f_list, m_f = _forward_all_layers(backend, ids_f, m_f, rc=False)
    hs_r_list, m_r = _forward_all_layers(backend, ids_r, m_r, rc=True)

    L_total = len(hs_f_list)

    # choose layers
    if isinstance(layer_spec, int):
        layers = [layer_spec % L_total]
    else:
        mode, k = layer_spec
        assert mode == "last_k"
        layers = list(range(max(0, L_total - k), L_total))

    # position weights
    wpos_f = _make_weights(m_f, dir_mode, **dir_params)
    wpos_r = _make_weights(m_r, dir_mode, **dir_params)

    # parse channel grouping K from axis name
    ch_group_K = None
    if pooling_axis.startswith("channels→position"):
        K = pooling_axis.split("[")[-1].split("]")[0]
        ch_group_K = int(K)
        pa = "channels→position"
    elif pooling_axis.startswith("position→channels"):
        K = pooling_axis.split("[")[-1].split("]")[0]
        ch_group_K = int(K)
        pa = "position→channels"
    else:
        pa = pooling_axis

    # ---- produce one vector for forward and RC ----
    if pa == "position":
        li = layers[0]
        ef = _pool_position(hs_f_list[li], m_f, wpos_f)  # [1, H]
        er = _pool_position(hs_r_list[li], m_r, wpos_r)

    elif pa == "layers→position":
        hf = _mean_layers_token(hs_f_list, layers)               # [1, T, H]
        hr = _mean_layers_token(hs_r_list, layers)
        ef = _pool_position(hf, m_f, wpos_f)                     # [1, H]
        er = _pool_position(hr, m_r, wpos_r)

    elif pa == "position→layers":
        ef = _position_then_layers(hs_f_list, m_f, wpos_f, layers)  # [1, H]
        er = _position_then_layers(hs_r_list, m_r, wpos_r, layers)

    elif pa == "channels→position":
        li = layers[0] if len(layers) == 1 else layers[-1]
        hfK = _pool_channels_groupwise(hs_f_list[li], ch_group_K)  # [1, T, K]
        hrK = _pool_channels_groupwise(hs_r_list[li], ch_group_K)  # [1, T, K]
        ef = _pool_position(hfK, m_f, wpos_f)  # [1, K]
        er = _pool_position(hrK, m_r, wpos_r)  # [1, K]

    elif pa == "position→channels":
        li = layers[0] if len(layers) == 1 else layers[-1]
        efH = _pool_position(hs_f_list[li], m_f, wpos_f)  # [1, H]
        erH = _pool_position(hs_r_list[li], m_r, wpos_r)  # [1, H]
        ef = _pool_channels_after_position(efH, ch_group_K)  # [1, K]
        er = _pool_channels_after_position(erH, ch_group_K)  # [1, K]
    else:
        raise ValueError(f"Unknown pooling_axis: {pooling_axis}")

    rc_sim = F.cosine_similarity(ef, er).item()
    out_dim = ef.shape[-1]

    # ---- locality margin on forward ONLY (tiled along position) ----
    pos, Lb = _pos_indices(m_f); L = int(Lb.item())
    if L == 0:
        return dict(rc_sim=float("nan"), adj_mean=float("nan"), far_mean=float("nan"),
                    margin=float("nan"), out_dim=out_dim)

    first_content = int((pos >= 0).nonzero(as_tuple=False)[0, 1].item())

    # token-level states for tiling
    if pa == "position":
        li = layers[0]
        hs_tok = hs_f_list[li][:, first_content:first_content+L, :]
    elif pa == "layers→position":
        hs_tok = _mean_layers_token(hs_f_list, layers)[:, first_content:first_content+L, :]
    elif pa == "position→layers":
        hs_tok = _mean_layers_token(hs_f_list, layers)[:, first_content:first_content+L, :]
    elif pa == "channels→position":
        li = layers[0] if len(layers) == 1 else layers[-1]
        hs_tok = _pool_channels_groupwise(hs_f_list[li], ch_group_K)[:, first_content:first_content+L, :]
    elif pa == "position→channels":
        li = layers[0] if len(layers) == 1 else layers[-1]
        hs_tok = hs_f_list[li][:, first_content:first_content+L, :]
    else:
        raise AssertionError

    wpos_tok = wpos_f[:, first_content:first_content+L]   # [1, L]
    ranges = _tile_ranges(L, tile_bp=tile_bp)
    tile_embs = []
    for (a, b) in ranges:
        if pa == "position→channels":
            efH_tile = _pool_position(hs_tok[:, a:b, :], torch.ones_like(wpos_tok[:, a:b]), wpos_tok[:, a:b])  # [1, H]
            e_tile = _pool_channels_after_position(efH_tile, ch_group_K)  # [1, K]
        else:
            e_tile = _pool_position(hs_tok[:, a:b, :], torch.ones_like(wpos_tok[:, a:b]), wpos_tok[:, a:b])   # [1, D]
        tile_embs.append(e_tile)
    tile_embs = torch.cat(tile_embs, dim=0)  # [N, D]
    N = tile_embs.shape[0]
    if N < 2:
        return dict(rc_sim=rc_sim, adj_mean=1.0, far_mean=1.0, margin=0.0, out_dim=out_dim)

    # adjacency (neighbor tiles)
    adj = F.cosine_similarity(tile_embs[:-1], tile_embs[1:]).mean().item()

    # "far" by tile index distance (robust)
    min_far_tiles = max(1, math.ceil(far_min_gap_bp / tile_bp))
    far_pairs = [(i, j) for i in range(N) for j in range(i+1, N) if (j - i) >= min_far_tiles]
    if not far_pairs:
        far_mean = adj
    else:
        a = torch.stack([tile_embs[i] for (i, j) in far_pairs], dim=0)
        b = torch.stack([tile_embs[j] for (i, j) in far_pairs], dim=0)
        far_mean = F.cosine_similarity(a, b).mean().item()

    margin = adj - far_mean
    return dict(rc_sim=rc_sim, adj_mean=adj, far_mean=far_mean, margin=margin, out_dim=out_dim)


# ------------------------------------------------------------
# Global search for maximum between-sequence separation
# ------------------------------------------------------------

@torch.no_grad()
def benchmark_for_max_separation(
    backend,
    sequences: List[str],
    *,
    tile_bp: int = 1000,
    far_min_gap_bp: int = 5000,
    dir_candidates: Optional[List[Tuple[str, Dict[str, float]]]] = None,
    channel_group_sizes: Optional[List[int]] = None,
    max_last_single_layers: int = 8,
    last_k_mixes: Tuple[int, ...] = (2, 3, 4, 5)
) -> List[Dict[str, object]]:

    if dir_candidates is None:
        # tuned breadth for ~10kb windows
        dir_candidates = [
            ("mean",        {}),
            ("linear_right",{}),
            ("linear_left", {}),
            ("exp_right",   {"tau": 64.0}),
            ("exp_right",   {"tau": 128.0}),
            ("exp_left",    {"tau": 64.0}),
            ("exp_left",    {"tau": 128.0}),
            ("tail_k",      {"tail_k": 1024}),
            ("tail_k",      {"tail_k": 2048}),
            ("tail_k",      {"tail_k": 4096}),
            ("head_k",      {"head_k": 1024}),
            ("head_k",      {"head_k": 2048}),
            ("head_k",      {"head_k": 4096}),
        ]

    if channel_group_sizes is None:
        channel_group_sizes = [8, 16, 32]  # explore final dims

    # Determine layer count from first sequence
    ids0, m0 = _tokenize(backend, sequences[0], rc=False)
    hs0_list, _ = _forward_all_layers(backend, ids0, m0, rc=False)
    L_total = len(hs0_list)

    # layer candidates
    singles = list(range(-min(L_total, max_last_single_layers), 0))  # last up to N layers
    mixes   = [("last_k", k) for k in last_k_mixes]
    layer_candidates = {"single": singles, "mix": mixes}

    # pooling axes
    pool_axes = ["position", "layers→position", "position→layers"] + \
                [f"channels→position[{K}]" for K in channel_group_sizes] + \
                [f"position→channels[{K}]" for K in channel_group_sizes]

    rows: List[Dict[str, object]] = []

    # helper to min-max normalize a metric across rows (finite only)
    def _norm(key: str) -> List[float]:
        vals_list = [r[key] for r in rows]
        m, M = _finite_minmax(vals_list)
        out = []
        for v in vals_list:
            if not math.isfinite(float(v)) or not math.isfinite(m) or not math.isfinite(M) or (M - m) < 1e-12:
                out.append(float("nan"))
            else:
                out.append((float(v) - m) / (M - m))
        return out

    # main grid
    for dir_mode, dir_params in dir_candidates:
        for pooling_axis in pool_axes:
            layer_space = layer_candidates["single"] if (pooling_axis == "position") else \
                          (list(layer_candidates["single"]) + list(layer_candidates["mix"]))
            for layer_spec in layer_space:
                # aggregate per-config over sequences
                rc_vals, adj_vals, far_vals, mar_vals = [], [], [], []
                ef_list = []  # FORWARD vectors for between-sequence distances

                for s in sequences:
                    # scoring metrics (RC, locality)
                    r = _score_seq_config(
                        backend, s,
                        dir_mode=dir_mode, dir_params=dir_params,
                        pooling_axis=pooling_axis, layer_spec=layer_spec,
                        tile_bp=tile_bp, far_min_gap_bp=far_min_gap_bp
                    )
                    rc_vals.append(r["rc_sim"])
                    adj_vals.append(r["adj_mean"])
                    far_vals.append(r["far_mean"])
                    mar_vals.append(r["margin"])
                    out_dim = r["out_dim"]  # same for all seqs in this config

                    # reconstruct FORWARD vector (consistent with config)
                    ids_f, m_f = _tokenize(backend, s, rc=False)
                    hs_f_list, m_f = _forward_all_layers(backend, ids_f, m_f, rc=False)
                    wpos_f = _make_weights(m_f, dir_mode, **dir_params)

                    pa = pooling_axis
                    chK = None
                    if pa.startswith("channels→position"):
                        chK = int(pa.split("[")[-1].split("]")[0]); pa = "channels→position"
                    elif pa.startswith("position→channels"):
                        chK = int(pa.split("[")[-1].split("]")[0]); pa = "position→channels"

                    if isinstance(layer_spec, int):
                        layers = [layer_spec % L_total]
                    else:
                        modeK, k = layer_spec
                        layers = list(range(max(0, L_total - k), L_total))

                    if pa == "position":
                        li = layers[0]
                        ef = _pool_position(hs_f_list[li], m_f, wpos_f)  # [1, D]
                    elif pa == "layers→position":
                        hf = _mean_layers_token(hs_f_list, layers)
                        ef = _pool_position(hf, m_f, wpos_f)
                    elif pa == "position→layers":
                        ef = _position_then_layers(hs_f_list, m_f, wpos_f, layers)
                    elif pa == "channels→position":
                        li = layers[0] if len(layers) == 1 else layers[-1]
                        hfK = _pool_channels_groupwise(hs_f_list[li], chK)  # [1, T, K]
                        ef = _pool_position(hfK, m_f, wpos_f)               # [1, K]
                    elif pa == "position→channels":
                        li = layers[0] if len(layers) == 1 else layers[-1]
                        efH = _pool_position(hs_f_list[li], m_f, wpos_f)    # [1, H]
                        ef = _pool_channels_after_position(efH, chK)        # [1, K]
                    else:
                        raise AssertionError

                    ef_list.append(ef.squeeze(0))  # [D]

                # Between-sequence distances (FORWARD only)
                X = torch.stack(ef_list, dim=0)  # [N, D]
                sep = _pairwise_nonself_metrics(X)

                row = dict(
                    direction=dir_mode,
                    dir_params=dir_params,
                    pooling_axis=pooling_axis,
                    layer_spec=layer_spec,
                    rc_sim=_finite_mean_list(rc_vals),
                    adj_mean=_finite_mean_list(adj_vals),
                    far_mean=_finite_mean_list(far_vals),
                    margin=_finite_mean_list(mar_vals),
                    out_dim=int(X.shape[-1]),
                    mean_cos_dist=sep["mean_cos_dist"],
                    min_cos_dist=sep["min_cos_dist"],
                    mean_L2=sep["mean_L2"],
                    min_L2=sep["min_L2"],
                    nn_cos_sim=sep["nn_cos_sim"],
                    nn_L2=sep["nn_L2"],
                    total_layers=L_total
                )
                rows.append(row)

    # ---- separation-driven ranking ----
    # Normalize metrics across rows to [0,1] (nan-safe)
    def _norm_all():
        n_min_cos = _norm("min_cos_dist")
        n_min_L2  = _norm("min_L2")
        n_mean_cos= _norm("mean_cos_dist")
        n_mean_L2 = _norm("mean_L2")
        n_margin  = _norm("margin")
        n_rc      = _norm("rc_sim")
        return n_min_cos, n_min_L2, n_mean_cos, n_mean_L2, n_margin, n_rc

    n_min_cos, n_min_L2, n_mean_cos, n_mean_L2, n_margin, n_rc = _norm_all()

    # weights emphasize smallest-pair separation; margin & RC as regularizers
    w_min_cos, w_min_L2, w_mean_cos, w_mean_L2, w_margin, w_rc = 0.35, 0.35, 0.10, 0.10, 0.06, 0.04
    for i, r in enumerate(rows):
        sep_score = 0.0
        for val, w in [(n_min_cos[i], w_min_cos), (n_min_L2[i], w_min_L2),
                       (n_mean_cos[i], w_mean_cos), (n_mean_L2[i], w_mean_L2),
                       (n_margin[i], w_margin), (n_rc[i], w_rc)]:
            if math.isfinite(val):
                sep_score += w * val
        r["separation_score"] = sep_score

    # NaN-proof sort; prefer smaller out_dim at the end as a gentle tiebreak
    def _rank_key(r):
        return (
            -_safe_finite(r.get("separation_score")),
            -_safe_finite(r.get("min_L2")),
            -_safe_finite(r.get("min_cos_dist")),
            -_safe_finite(r.get("margin")),
            -_safe_finite(r.get("rc_sim")),
             _safe_finite(r.get("out_dim"), fallback=1e9)
        )
    rows.sort(key=_rank_key)
    return rows


# ------------------------------------------------------------
# Runner: executes grid, saves CSV & best embeddings
# ------------------------------------------------------------

@torch.no_grad()
def run_chr22_window_search(
    backend,
    seqs: List[str],
    work_dir: Path,
    *,
    tile_bp: int = 1000,
    far_min_gap_bp: int = 5000,
    channel_group_sizes=(8,16,32),
    max_last_single_layers: int = 8,
    last_k_mixes=(2,3,4,5),
    topk_print: int = 10
):
    rows = benchmark_for_max_separation(
        backend, seqs,
        tile_bp=tile_bp,
        far_min_gap_bp=far_min_gap_bp,
        channel_group_sizes=list(channel_group_sizes),
        max_last_single_layers=max_last_single_layers,
        last_k_mixes=tuple(last_k_mixes)
    )

    # Save CSV summary
    out_csv = work_dir / "embedding_path_search_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["rank","direction","dir_params","pooling_axis","layer_spec","out_dim",
                    "separation_score",
                    "rc_sim","adj_mean","far_mean","margin",
                    "mean_cos_dist","min_cos_dist","nn_cos_sim",
                    "mean_L2","min_L2","nn_L2","total_layers"])
        for rank, r in enumerate(rows, 1):
            w.writerow([
                rank, r["direction"], str(r["dir_params"]), r["pooling_axis"], str(r["layer_spec"]), r["out_dim"],
                f"{_safe_finite(r['separation_score'], 0.0):.6f}",
                f"{_safe_finite(r['rc_sim'], 0.0):.6f}",
                f"{_safe_finite(r['adj_mean'], 0.0):.6f}",
                f"{_safe_finite(r['far_mean'], 0.0):.6f}",
                f"{_safe_finite(r['margin'], 0.0):.6f}",
                f"{_safe_finite(r['mean_cos_dist'], 0.0):.6f}",
                f"{_safe_finite(r['min_cos_dist'], 0.0):.6f}",
                f"{_safe_finite(r['nn_cos_sim'], 0.0):.6f}",
                f"{_safe_finite(r['mean_L2'], 0.0):.6f}",
                f"{_safe_finite(r['min_L2'], 0.0):.6f}",
                f"{_safe_finite(r['nn_L2'], 0.0):.6f}",
                r["total_layers"],
            ])

    # Print top-k
    print(f"\nTop {topk_print} configs by separation_score")
    for r in rows[:topk_print]:
        print(f"{r['direction']:12s} {str(r['dir_params']):18s} {r['pooling_axis']:22s} "
              f"layer={str(r['layer_spec']):8s} out_dim={r['out_dim']:>3d} "
              f"sep={_safe_finite(r['separation_score'], 0.0):.4f}  "
              f"minCos={_safe_finite(r['min_cos_dist'], 0.0):.4f} minL2={_safe_finite(r['min_L2'], 0.0):.4f} "
              f"RC={_safe_finite(r['rc_sim'], 0.0):.4f}  margin={_safe_finite(r['margin'], 0.0):.4f}")

    # Recompute and save embeddings for the BEST configuration (FORWARD only; one vector per sequence)
    best = rows[0]
    print("\nBest config:", best)

    # helper to produce FORWARD vector for a given seq and best config
    def _forward_vector_for_best(seq: str) -> torch.Tensor:
        ids_f, m_f = _tokenize(backend, seq, rc=False)
        hs_f_list, m_f = _forward_all_layers(backend, ids_f, m_f, rc=False)
        L_total = len(hs_f_list)
        wpos_f = _make_weights(m_f, best["direction"], **best["dir_params"])

        pa = best["pooling_axis"]
        chK = None
        if pa.startswith("channels→position"):
            chK = int(pa.split("[")[-1].split("]")[0]); pa = "channels→position"
        elif pa.startswith("position→channels"):
            chK = int(pa.split("[")[-1].split("]")[0]); pa = "position→channels"

        layer_spec = best["layer_spec"]
        if isinstance(layer_spec, int):
            layers = [layer_spec % L_total]
        else:
            _, k = layer_spec
            layers = list(range(max(0, L_total - k), L_total))

        if pa == "position":
            li = layers[0]
            ef = _pool_position(hs_f_list[li], m_f, wpos_f)
        elif pa == "layers→position":
            hf = _mean_layers_token(hs_f_list, layers)
            ef = _pool_position(hf, m_f, wpos_f)
        elif pa == "position→layers":
            ef = _position_then_layers(hs_f_list, m_f, wpos_f, layers)
        elif pa == "channels→position":
            li = layers[0] if len(layers) == 1 else layers[-1]
            hfK = _pool_channels_groupwise(hs_f_list[li], chK)
            ef = _pool_position(hfK, m_f, wpos_f)
        elif pa == "position→channels":
            li = layers[0] if len(layers) == 1 else layers[-1]
            efH = _pool_position(hs_f_list[li], m_f, wpos_f)
            ef = _pool_channels_after_position(efH, chK)
        else:
            raise AssertionError
        return ef.squeeze(0)  # [D]

    best_vectors = torch.stack([_forward_vector_for_best(s) for s in seqs], dim=0)  # [N, D]

    # Save tensors
    best_pt = work_dir / "best_sequence_embeddings.pt"
    best_npy = work_dir / "best_sequence_embeddings.npy"
    torch.save({"vectors": best_vectors, "meta": best}, best_pt)
    try:
        import numpy as np
        np.save(best_npy, best_vectors.cpu().numpy())
    except Exception:
        pass

    # Save best config as text
    best_txt = work_dir / "best_config.txt"
    with best_txt.open("w") as f:
        f.write(repr(best) + "\n")

    print(f"\nSaved summary to: {out_csv}")
    print(f"Saved best embeddings to: {best_pt} (and .npy if numpy available)")

    return rows


# ------------------------------------------------------------
# Optional: summary helper (NaN-safe)
# ------------------------------------------------------------

def summarize_best(rows: List[Dict[str, object]]) -> Dict[str, object]:
    """Summarize winners for best overall, best single layer, and best pooling axis family."""
    if not rows:
        return dict(best_overall=None, best_single_layer=None, best_pooling_axis_family_by_avg_margin={})
    best_overall = rows[0]
    singles = [r for r in rows if isinstance(r["layer_spec"], int)]
    best_single = singles[0] if singles else None

    def axis_family(ax: str) -> str:
        if ax.startswith("channels→position"): return "channels→position"
        if ax.startswith("position→channels"): return "position→channels"
        return ax

    fams = sorted(set(axis_family(r["pooling_axis"]) for r in rows))
    fam_scores = {}
    for f in fams:
        vals = [r["margin"] for r in rows if axis_family(r["pooling_axis"]) == f]
        fam_scores[f] = _finite_mean_list(vals)

    # champion by avg margin (finite-safe)
    champion = None
    if fam_scores:
        champion = max(fam_scores.items(), key=lambda kv: _safe_finite(kv[1], fallback=-float("inf")))[0]

    return dict(
        best_overall=best_overall,
        best_single_layer=best_single,
        best_pooling_axis_family_by_avg_margin=dict(champion=champion, axis_avg_margin=fam_scores)
    )


# Assumes you already created `seqs = [chromA[i:i+window] ...]` as given.
backend = HyenaBackend(
    model_name="hyenadna-large-1m-seqlen-hf",
    model_dir="/g/data/te53/en9803/data/hf-cache/models/",
    pooling="mean",
    normalize=True,
    offline=True,
    prefer_cuda=True,
)

out_dir = Path("/g/data/te53/en9803/data/bench")

# 2) Run the search tuned for 10kb windows
rows = run_chr22_window_search(
    backend,
    seqs,
    work_dir=out_dir,
    tile_bp=1000,           # 10 tiles per sequence; stable for locality tests
    far_min_gap_bp=5000,    # "far" = ≥ 5 tiles apart
    channel_group_sizes=(8,16,32),  # test output dims for channel pooling
    max_last_single_layers=8,
    last_k_mixes=(2,3,4,5),
    topk_print=12
)
print(summarize_best(rows))



