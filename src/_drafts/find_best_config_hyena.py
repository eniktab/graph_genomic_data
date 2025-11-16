from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple, Iterable, Optional
import itertools
import json
import contextlib
import inspect

import pysam
import torch

from src.HyenaBackend import HyenaDNAPooler, HyenaBackend
from src.configs import IndexConfig

from typing import Optional, Dict, Any, List, Tuple

# =========================
# Globals / basic settings
# =========================
DEFAULT_BEST_CFG: Dict[str, Any] = {
    "direction": "exp_right",
    "tau": 40.0,
    "pooling_axis": "position→layers",  # accepts "position->layers" too (see _normalize_axis below)
    "layer_spec": -7,
    "rc_average": False,
}


WINDOW = 10_000
cfg_index = IndexConfig(window=WINDOW, stride=5000, rc_index=True)
N_TILES = 100
TARGET_LEN = WINDOW * N_TILES

_RC_MAP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _normalize_axis(axis: str) -> str:
    """Allow ASCII fallback '->' if someone passes it; keep your canonical '→' form."""
    return axis.replace("->", "→")

def _revcomp(seq: str) -> str:
    return seq.translate(_RC_MAP)[::-1]


@torch.inference_mode()
def _l2_normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("Expected [N, D] embeddings.")
    return x / (x.norm(dim=1, keepdim=True) + eps)


@torch.inference_mode()
def _pairwise_stats_from_normalized(X: torch.Tensor) -> Dict[str, float]:
    """
    Compute min/mean pairwise distances from already L2-normalized rows.
    L2 from cosine: ||a-b|| = sqrt(2(1 - cos)).
    """
    assert X.ndim == 2
    N = X.size(0)
    if N < 2:
        return dict(min_L2=0.0, mean_L2=0.0, min_cos=0.0, mean_cos=0.0)

    # Cosine similarity (since rows are L2-normalized)
    cos = X @ X.T
    cos.fill_diagonal_(0.0)  # remove self-sim
    cos_dist = 1.0 - cos
    l2 = torch.clamp(2.0 * (1.0 - cos), min=0.0).sqrt()

    mask = ~torch.eye(N, device=X.device, dtype=torch.bool)
    l2_flat = l2[mask].reshape(N, N - 1)
    cos_flat = cos_dist[mask].reshape(N, N - 1)

    min_L2_vals, _ = l2_flat.min(dim=1)
    min_cos_vals, _ = cos_flat.min(dim=1)

    return dict(
        min_L2=min_L2_vals.mean().item(),
        mean_L2=l2_flat.mean().item(),
        min_cos=min_cos_vals.mean().item(),
        mean_cos=cos_flat.mean().item(),
    )


# =========================================================
# Robust micro-batching with OOM backoff (embed_list/pooler)
# =========================================================
def _estimate_tokens(seqs: List[str]) -> int:
    # Hyena is character-level; tokens ~= bases
    return sum(len(s) for s in seqs)


@contextlib.contextmanager
def _maybe_cuda_empty():
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def _split_by_max_tokens(
    seqs: List[str],
    max_tokens_per_batch: int,
) -> Iterable[List[str]]:
    if max_tokens_per_batch <= 0:
        yield seqs
        return
    cur, cur_tokens = [], 0
    for s in seqs:
        L = len(s)
        if cur and (cur_tokens + L > max_tokens_per_batch):
            yield cur
            cur, cur_tokens = [s], L
        else:
            cur.append(s)
            cur_tokens += L
    if cur:
        yield cur


def _supports_kwarg(fn, kw: str) -> bool:
    try:
        sig = inspect.signature(fn)
        return (kw in sig.parameters) or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    except (ValueError, TypeError):
        # Builtins/C extensions without signature metadata: assume it might accept
        return True


@torch.inference_mode()
def _embed_with_oom_backoff_using_bs(
    call_fn,
    args_tuple,
    kwargs,
    *,
    start_bs: int,
) -> torch.Tensor:
    """
    Generic OOM backoff loop using batch_size halving. Assumes call_fn supports 'batch_size' kw.
    """
    bs = max(1, int(start_bs))
    while True:
        try:
            kwargs["batch_size"] = max(1, bs)
            with _maybe_cuda_empty():
                return call_fn(*args_tuple, **kwargs)
        except RuntimeError as e:
            msg = str(e)
            if "CUDA out of memory" not in msg and "CUBLAS_STATUS_ALLOC_FAILED" not in msg:
                raise
            if bs <= 1:
                raise
            bs = max(1, bs // 2)


@torch.inference_mode()
def _call_no_bs_with_recursive_split(
    call_fn,
    seqs: List[str],
    kwargs: Dict[str, Any],
) -> torch.Tensor:
    """
    For callables that DON'T accept batch_size: if OOM, split the chunk into halves recursively.
    """
    try:
        with _maybe_cuda_empty():
            return call_fn(seqs, **kwargs)
    except RuntimeError as e:
        msg = str(e)
        if ("CUDA out of memory" not in msg and "CUBLAS_STATUS_ALLOC_FAILED" not in msg) or len(seqs) <= 1:
            raise
        mid = len(seqs) // 2
        left = _call_no_bs_with_recursive_split(call_fn, seqs[:mid], kwargs)
        right = _call_no_bs_with_recursive_split(call_fn, seqs[mid:], kwargs)
        return torch.cat((left, right), dim=0)


@torch.inference_mode()
def _embed_with_pooler_batched(
    pooler,
    seqs: List[str],
    *,
    max_tokens_per_batch: int = 200_000,
    start_batch_size: Optional[int] = None,
) -> torch.Tensor:
    """
    pooler.embed may support batch_size. We chunk by tokens and OOM-backoff by bs if supported,
    otherwise recursively split chunks on OOM. Returns a single [N, D] tensor.
    """
    outs = []
    if start_batch_size is None:
        avgL = max(1, _estimate_tokens(seqs) // max(1, len(seqs)))
        start_batch_size = max(1, max_tokens_per_batch // avgL)

    supports_bs = _supports_kwarg(pooler.embed, "batch_size")

    for chunk in _split_by_max_tokens(seqs, max_tokens_per_batch):
        if supports_bs:
            X = _embed_with_oom_backoff_using_bs(
                pooler.embed,
                (chunk,),
                dict(),
                start_bs=start_batch_size,
            )
        else:
            X = _call_no_bs_with_recursive_split(pooler.embed, chunk, {})
        outs.append(X)
    return torch.cat(outs, dim=0)


@torch.inference_mode()
def _baseline_embed_list(
    backend: HyenaBackend,
    seqs: List[str],
    pooling: str = "mean",
    rc_invariant: bool = False,
    *,
    max_tokens_per_batch: int = 200_000,
    start_batch_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Robust micro-batched embed_list. If backend.embed_list supports batch_size, use bs backoff;
    else, recursively split chunks on OOM. Concats on-device; L2-normalizes rows.
    """
    outs = []
    if start_batch_size is None:
        avgL = max(1, _estimate_tokens(seqs) // max(1, len(seqs)))
        start_batch_size = max(1, max_tokens_per_batch // avgL)

    supports_bs = _supports_kwarg(backend.embed_list, "batch_size")

    for chunk in _split_by_max_tokens(seqs, max_tokens_per_batch):
        if supports_bs:
            X = _embed_with_oom_backoff_using_bs(
                backend.embed_list,
                (chunk,),
                dict(pooling=pooling, rc_invariant=rc_invariant),
                start_bs=start_batch_size,
            )
        else:
            X = _call_no_bs_with_recursive_split(
                lambda c, **k: backend.embed_list(c, pooling=pooling, rc_invariant=rc_invariant),
                chunk,
                {},
            )
        if isinstance(X, tuple):
            X = X[0]
        outs.append(X)
    X = torch.cat(outs, dim=0)
    return _l2_normalize_rows(X)


# ======================================
# RC agreement (uses batched embeddings)
# ======================================
@torch.inference_mode()
def _rc_agreement(
    seqs: List[str],
    pooler,
    *,
    max_tokens_per_batch: int = 200_000,
) -> float:
    """
    Average cosine agreement between forward vs reverse-complement embeddings.
    Uses batched pooler embedding to avoid OOM. We compute with rc_average=False
    regardless of the pooler current setting.
    """
    cfg = pooler.get_config()
    need_reset = cfg.get("rc_average", False)
    if need_reset:
        pooler.set_config(rc_average=False)

    X_f = _embed_with_pooler_batched(pooler, seqs, max_tokens_per_batch=max_tokens_per_batch)
    X_r = _embed_with_pooler_batched(
        pooler, [s.translate(_RC_MAP)[::-1] for s in seqs], max_tokens_per_batch=max_tokens_per_batch
    )

    if need_reset:
        pooler.set_config(rc_average=True)

    rc_cos = (X_f * X_r).sum(dim=1).mean().item()
    return rc_cos


# ---------------------------
# Ranking logic
# ---------------------------
def _rank_tuple(stats: Dict[str, float], rc_mean: float = 0.0) -> Tuple:
    # Emphasize worst-case separation (min_L2, then min_cos), then means, then RC agreement.
    return (
        round(stats["min_L2"], 9),
        round(stats["min_cos"], 9),
        round(stats["mean_L2"], 9),
        round(stats["mean_cos"], 9),
        round(rc_mean, 9),
    )


# ======================================================
# NEW: Two-stage search
#   Stage 1: layer taps only (find best layers)
# Stage 2: for top-K layers → sweep taus & pooling
# ======================================================
@torch.inference_mode()
def layers_first_then_tau_pool_sweep(
    seqs: List[str],
    backend: HyenaBackend,
    *,
    layer_candidates: Optional[List[int]] = None,
    top_k_layers: int = 5,
    tau_candidates: Tuple[float, ...] = (40.0, 56.0, 64.0, 72.0, 96.0),
    pooling_axes: Tuple[str, ...] = ("position", "layers→position", "position→layers"),
    include_baselines: bool = True,
    include_auto: bool = True,
    include_presets: bool = False,  # optional; can enable if you like
    max_tokens_per_batch: int = 200_000,
    print_every: int = 50,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Stage 1: Evaluate single-layer specs with a fixed simple pooling (mean over positions),
             rc_average=False, direction='mean'.
    Stage 2: Take top-K layers and sweep taus + pooling axes under exp_left/exp_right + linear/mean.
    """
    if not seqs:
        raise ValueError("No sequences provided.")
    if layer_candidates is None:
        layer_candidates = [-1, -3, -5, -7, -9, -11]

    entries: List[Dict[str, Any]] = []

    def _record(label: str, cfg: Dict[str, Any], X: torch.Tensor, rc_mean: float = 0.0):
        stats = _pairwise_stats_from_normalized(_l2_normalize_rows(X))
        entries.append({
            "label": label,
            "config": cfg,
            "stats": stats,
            "rc_mean": rc_mean,
            "rank": _rank_tuple(stats, rc_mean),
        })

    # ---------- Optional baselines ----------
    if include_baselines:
        try:
            X = _baseline_embed_list(
                backend, seqs, pooling="mean", rc_invariant=False,
                max_tokens_per_batch=max_tokens_per_batch
            )
            _record("baseline:embed_list_mean", {"baseline": "embed_list_mean"}, X, rc_mean=0.0)
        except Exception as e:
            print("[baseline] embed_list mean failed:", repr(e))

        try:
            X = _baseline_embed_list(
                backend, seqs, pooling="mean", rc_invariant=True,
                max_tokens_per_batch=max_tokens_per_batch
            )
            _record("baseline:embed_list_mean_rcinv", {"baseline": "embed_list_mean_rcinv"}, X, rc_mean=1.0)
        except Exception as e:
            print("[baseline] embed_list mean rc invariant failed:", repr(e))

    # ---------- Optional presets ----------
    if include_presets:
        for preset in ("cluster_max_sep", "balanced_rc", "rc_invariant"):
            try:
                p = HyenaDNAPooler.from_preset(backend, preset=preset)
                X = _embed_with_pooler_batched(p, seqs, max_tokens_per_batch=max_tokens_per_batch)
                rc = _rc_agreement(seqs, p, max_tokens_per_batch=max_tokens_per_batch)
                cfg = {"preset": preset}
                cfg.update(p.get_config())
                _record(f"preset:{preset}", cfg, X, rc)
            except Exception as e:
                print(f"[preset] {preset} failed:", repr(e))

    # ---------- Optional auto ----------
    if include_auto:
        try:
            pooler_auto = backend.build_pooler(
                auto_select=True,
                auto_seqs=seqs,
                auto_max_seqs=len(seqs),
                auto_rc_floor=0.85,
                auto_verbose=True,
            )
            X = _embed_with_pooler_batched(pooler_auto, seqs, max_tokens_per_batch=max_tokens_per_batch)
            rc = _rc_agreement(seqs, pooler_auto, max_tokens_per_batch=max_tokens_per_batch)
            cfg = {"auto": True}
            cfg.update(pooler_auto.get_config())
            _record("auto:pooler", cfg, X, rc)
        except Exception as e:
            print("[auto] pooler auto-select failed:", repr(e))

    # =========================
    # Stage 1: Layers only
    # =========================
    stage1_results = []
    for i, layer in enumerate(layer_candidates, 1):
        cfg = {
            "direction": "mean",           # simple, no tau
            "pooling_axis": "position",    # mean over positions for that layer
            "layer_spec": layer,           # single layer tap
            "rc_average": False,
        }
        try:
            pooler = backend.build_pooler(**cfg)
            X = _embed_with_pooler_batched(pooler, seqs, max_tokens_per_batch=max_tokens_per_batch)
            stats = _pairwise_stats_from_normalized(_l2_normalize_rows(X))
            rank = _rank_tuple(stats, 0.0)
            stage1_results.append({"layer": layer, "cfg": cfg, "stats": stats, "rank": rank})
            _record("stage1:layer", cfg, X, rc_mean=0.0)
        except Exception:
            pass

        if print_every and (i % print_every == 0):
            best_so_far = max(stage1_results, key=lambda r: r["rank"]) if stage1_results else None
            if best_so_far:
                print(f"[stage1] {i}/{len(layer_candidates)} layers; current best layer={best_so_far['layer']} stats={best_so_far['stats']}")

    if not stage1_results:
        # Fallback: nothing worked; bail out with whatever baseline we have
        entries.sort(key=lambda r: r["rank"], reverse=True)
        best = entries[0] if entries else {"label": "none", "config": {}, "stats": {}, "rc_mean": 0.0, "rank": tuple()}
        return best, entries

    stage1_results.sort(key=lambda r: r["rank"], reverse=True)
    top_layers = [r["layer"] for r in stage1_results[:top_k_layers]]

    # =========================
    # Stage 2: Taus & pooling on top-K layers
    # =========================
    directions = [("exp_left", True), ("exp_right", True), ("linear_left", False), ("linear_right", False), ("mean", False)]
    total = len(top_layers) * sum(len(tau_candidates) if needs_tau else 1 for _, needs_tau in directions) * len(pooling_axes)
    tested = 0

    for layer in top_layers:
        for direction, needs_tau in directions:
            taus = tau_candidates if needs_tau else (None,)
            for tau in taus:
                for axis in pooling_axes:
                    cfg = {
                        "direction": direction,
                        "pooling_axis": axis,
                        "layer_spec": layer,  # keep the selected layer
                        "rc_average": False,
                    }
                    if tau is not None:
                        cfg["tau"] = float(tau)
                    try:
                        pooler = backend.build_pooler(**cfg)
                        X = _embed_with_pooler_batched(pooler, seqs, max_tokens_per_batch=max_tokens_per_batch)
                        rc = _rc_agreement(seqs, pooler, max_tokens_per_batch=max_tokens_per_batch)
                        _record("stage2:tau+pool", cfg, X, rc_mean=rc)
                    except Exception:
                        pass

                    tested += 1
                    if print_every and (tested % print_every == 0):
                        best_so_far = max(entries, key=lambda r: r["rank"]) if entries else None
                        if best_so_far:
                            print(f"[stage2] {tested}/{total} tested; current best:", best_so_far["stats"], best_so_far["config"])

    # Leaderboard
    entries.sort(key=lambda r: r["rank"], reverse=True)
    best = entries[0] if entries else {"label": "none", "config": {}, "stats": {}, "rc_mean": 0.0, "rank": tuple()}
    return best, entries


# ---------------------------
# Main API: choose + parity check
# ---------------------------
@torch.inference_mode()
def choose_config_via_pooler_and_check_last(
    seqs: List[str],
    backend: "HyenaBackend",
    *,
    # layer-first strategy params
    layer_candidates: Optional[List[int]] = None,
    top_k_layers: int = 5,
    tau_candidates: Tuple[float, ...] = (40.0, 56.0, 64.0, 72.0, 96.0),
    pooling_axes: Tuple[str, ...] = ("position", "layers→position", "position→layers"),
    include_baselines: bool = True,
    include_auto: bool = True,
    include_presets: bool = False,
    # parity baseline params
    baseline_pooling: str = "mean",
    baseline_rc_invariant: bool = False,
    # allclose thresholds for the “last layer” parity check
    atol: float = 1e-5,
    rtol: float = 1e-4,
    # batching
    max_tokens_per_batch: int = 200_000,
) -> Tuple[
    torch.Tensor,           # X_best: embeddings under best pooler config
    Dict[str, Any],         # best_cfg: chosen pooler config
    Dict[str, float],       # sep_best: separation metrics for X_best
    torch.Tensor,           # X_last_pooler: last-layer mean from a Pooler
    torch.Tensor,           # X_last_embedlist: last-layer mean from embed_list
    Dict[str, Any],         # last_check: equality report for pooler-vs-embed_list last-layer
]:
    """
    New strategy:
      - Stage 1: sweep single-layer taps; rank by separation (no RC).
      - Stage 2: refine top-K layers with taus & pooling variants (with RC agreement).
      - Keep last-layer parity checks as before.
    """
    if not seqs:
        raise ValueError("No sequences provided.")

    # Two-stage search
    best, leaderboard = layers_first_then_tau_pool_sweep(
        seqs,
        backend,
        layer_candidates=layer_candidates,
        top_k_layers=top_k_layers,
        tau_candidates=tau_candidates,
        pooling_axes=pooling_axes,
        include_baselines=include_baselines,
        include_auto=include_auto,
        include_presets=include_presets,
        max_tokens_per_batch=max_tokens_per_batch,
        print_every=50,
    )
    best_cfg = best["config"] if best else {}

    # Realize best pooler and embed
    pooler_best = backend.build_pooler(**best_cfg) if best_cfg else backend.build_pooler(
        direction="mean", pooling_axis="layers→position", layer_spec=("last_k", 1), rc_average=False
    )
    X_best = _embed_with_pooler_batched(pooler_best, seqs, max_tokens_per_batch=max_tokens_per_batch)
    X_best = _l2_normalize_rows(X_best)
    sep_best = _pairwise_stats_from_normalized(X_best)

    # 2) LAST-LAYER MEAN via Pooler (strict last layer; no RC averaging for separation)
    pooler_last = backend.build_pooler(
        direction="mean",
        pooling_axis="layers→position",
        layer_spec=("last_k", 1),
        rc_average=False,
    )
    X_last_pooler = _embed_with_pooler_batched(pooler_last, seqs, max_tokens_per_batch=max_tokens_per_batch)
    X_last_pooler = _l2_normalize_rows(X_last_pooler)

    # 3) LAST-LAYER MEAN via embed_list baseline (no RC invariance for separation)
    X_last_embedlist = _baseline_embed_list(
        backend,
        seqs,
        pooling=baseline_pooling,
        rc_invariant=baseline_rc_invariant,
        max_tokens_per_batch=max_tokens_per_batch,
    )

    # 4) Numerical parity check
    allclose_equal = torch.allclose(X_last_pooler, X_last_embedlist, atol=atol, rtol=rtol)
    last_check = {
        "allclose_equal": bool(allclose_equal),
        "atol": float(atol),
        "rtol": float(rtol),
    }
    if not allclose_equal:
        diff = (X_last_pooler - X_last_embedlist).abs()
        denom = X_last_embedlist.abs().clamp_min(1e-12)
        last_check["max_abs_diff"] = float(diff.max().item())
        last_check["max_rel_diff"] = float((diff / denom).max().item())

    # Expose the full leaderboard for inspection
    choose_config_via_pooler_and_check_last.leaderboard = leaderboard  # type: ignore[attr-defined]
    return X_best, best_cfg, sep_best, X_last_pooler, X_last_embedlist, last_check

@torch.inference_mode()
def compare_embedding_strategies(
    seqs: List[str],
    backend: "HyenaBackend",
    *,
    best_cfg: Optional[Dict[str, Any]] = None,       # <- now optional
    baseline_pooling: str = "mean",
    baseline_rc_invariant: bool = False,
    max_tokens_per_batch: int = 200_000,
    verbose: bool = True,
    return_embeddings: bool = False,
) -> Dict[str, Any]:
    """
    Compare:
      (A) Last-layer Pooler (strict last layer, no RC averaging)
      (B) embed_list baseline (mean pooling, RC-invariant toggle off by default)
      (C) Best config (defaults to the empirically found winner if not provided)

    Returns dict with per-method stats, RC agreement, sorted leaderboard, and optional embeddings.
    """
    # ---------- local RC helpers ----------
    __RC_MAP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    def _revcomp_local(s: str) -> str:
        return s.translate(__RC_MAP)[::-1]

    def _rc_agreement_embedlist() -> float:
        Xf = _baseline_embed_list(
            backend, seqs, pooling=baseline_pooling, rc_invariant=baseline_rc_invariant,
            max_tokens_per_batch=max_tokens_per_batch
        )
        Xr = _baseline_embed_list(
            backend, [_revcomp_local(s) for s in seqs],
            pooling=baseline_pooling, rc_invariant=baseline_rc_invariant,
            max_tokens_per_batch=max_tokens_per_batch
        )
        return float((Xf * Xr).sum(dim=1).mean().item())

    # ---------- (A) Last-layer Pooler ----------
    pooler_last = backend.build_pooler(
        direction="mean",
        pooling_axis="layers→position",
        layer_spec=("last_k", 1),
        rc_average=False,
    )
    X_last = _embed_with_pooler_batched(pooler_last, seqs, max_tokens_per_batch=max_tokens_per_batch)
    X_last = _l2_normalize_rows(X_last)
    stats_last = _pairwise_stats_from_normalized(X_last)
    rc_last = _rc_agreement(seqs, pooler_last, max_tokens_per_batch=max_tokens_per_batch)

    # ---------- (B) embed_list baseline ----------
    X_base = _baseline_embed_list(
        backend, seqs,
        pooling=baseline_pooling,
        rc_invariant=baseline_rc_invariant,
        max_tokens_per_batch=max_tokens_per_batch,
    )
    stats_base = _pairwise_stats_from_normalized(X_base)
    rc_base = _rc_agreement_embedlist()

    # ---------- (C) Best config (defaults to DEFAULT_BEST_CFG) ----------
    if best_cfg is None:
        best_cfg = dict(DEFAULT_BEST_CFG)  # copy to avoid accidental mutation
    # Be lenient about axis spelling
    if "pooling_axis" in best_cfg and isinstance(best_cfg["pooling_axis"], str):
        best_cfg["pooling_axis"] = _normalize_axis(best_cfg["pooling_axis"])

    pooler_best = backend.build_pooler(**best_cfg)
    X_best = _embed_with_pooler_batched(pooler_best, seqs, max_tokens_per_batch=max_tokens_per_batch)
    X_best = _l2_normalize_rows(X_best)
    stats_best = _pairwise_stats_from_normalized(X_best)
    rc_best = _rc_agreement(seqs, pooler_best, max_tokens_per_batch=max_tokens_per_batch)

    # ---------- Leaderboard ----------
    def _rank_tuple(stats: Dict[str, float], rc_mean: float) -> Tuple[float, float, float, float, float]:
        return (
            round(stats["min_L2"], 9),
            round(stats["min_cos"], 9),
            round(stats["mean_L2"], 9),
            round(stats["mean_cos"], 9),
            round(rc_mean, 9),
        )

    rows = [
        {"name": "best_config", "config": best_cfg, "stats": stats_best, "rc_mean": rc_best,
         "rank": _rank_tuple(stats_best, rc_best)},
        {"name": "last_layer_pooler",
         "config": {"direction": "mean", "pooling_axis": "layers→position", "layer_spec": ("last_k", 1), "rc_average": False},
         "stats": stats_last, "rc_mean": rc_last, "rank": _rank_tuple(stats_last, rc_last)},
        {"name": "embed_list_mean",
         "config": {"baseline": "embed_list", "pooling": baseline_pooling, "rc_invariant": baseline_rc_invariant},
         "stats": stats_base, "rc_mean": rc_base, "rank": _rank_tuple(stats_base, rc_base)},
    ]
    rows.sort(key=lambda r: r["rank"], reverse=True)
    winner = rows[0]

    if verbose:
        def _fmt(s: Dict[str, float]) -> str:
            return (f"min_L2={s['min_L2']:.4f}  mean_L2={s['mean_L2']:.4f}  "
                    f"min_cos={s['min_cos']:.4f}  mean_cos={s['mean_cos']:.4f}")
        print("\n[Comparison — higher is better for all metrics]")
        for i, r in enumerate(rows, 1):
            print(f"#{i} {r['name']:>18}  rc_mean={r['rc_mean']:.4f}  { _fmt(r['stats']) }")
        print(f"\nWinner: {winner['name']}")

    out = {
        "winner": winner,
        "leaderboard": rows,
        "configs": {
            "best_config": best_cfg,
            "last_layer_pooler": {"direction": "mean", "pooling_axis": "layers→position", "layer_spec": ("last_k", 1), "rc_average": False},
            "embed_list_mean": {"pooling": baseline_pooling, "rc_invariant": baseline_rc_invariant},
        },
        "stats": {
            "best_config": stats_best,
            "last_layer_pooler": stats_last,
            "embed_list_mean": stats_base,
        },
        "rc_mean": {
            "best_config": rc_best,
            "last_layer_pooler": rc_last,
            "embed_list_mean": rc_base,
        },
    }
    if return_embeddings:
        out["embeddings"] = {"best_config": X_best, "last_layer_pooler": X_last, "embed_list_mean": X_base}
    return out

# ---------------------------
# Example
# ---------------------------
if __name__ == "__main__":
    # Replace with your real sequences (all of them!)
    work = Path("/g/data/te53/en9803/sandpit/graph_genomics/chr22")
    work.mkdir(parents=True, exist_ok=True)
    ref_fa = work / "chm13v2_chr22.fa.gz"  # this should already exist

    fasta = pysam.FastaFile(str(ref_fa))
    if "chr22" in fasta.references:
        seq = fasta.fetch("chr22")
        fasta.close()
        chrom22_seq_clean = seq.upper().replace("N", "")
        if not chrom22_seq_clean:
            raise RuntimeError("chr22 sequence is empty after N removal. Check input FASTA?")
    # Short demo: two 10kb tiles (expand to your dataset)
    seqs = [chrom22_seq_clean[i:i + WINDOW] for i in range(0, WINDOW * 50, WINDOW)]

    backend = HyenaBackend(
        model_name=getattr(cfg_index, "model_name", None),
        model_dir=getattr(cfg_index, "model_dir", None),
        pooling="none",            # IMPORTANT: let Pooler control pooling
        normalize=False,           # we normalize explicitly
        offline=True,
        prefer_cuda=True,
    )

    X_auto, auto_cfg, sep_auto, Xp_last, Xb_last, check = choose_config_via_pooler_and_check_last(
        seqs,
        backend,
        layer_candidates=[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12],  # broader layer probe
        top_k_layers=5,
        tau_candidates=(40.0, 56.0, 64.0, 72.0, 96.0),
        pooling_axes=("position", "layers→position", "position→layers"),
        include_baselines=True,
        include_auto=True,
        include_presets=False,
        baseline_pooling="mean",
        baseline_rc_invariant=False,
        max_tokens_per_batch=200_000,  # drop if you still see OOM (e.g., 100_000 or 50_000)
    )

    print("\n[Best config after layers→taus/pooling sweep]")
    print(json.dumps(auto_cfg, indent=2))
    print("[Separation metrics]", json.dumps(sep_auto, indent=2))
    print("\n[Last-layer mean parity check: Pooler vs embed_list]")
    print(json.dumps(check, indent=2))

    # Optional: print top-10 leaderboard
    lb = getattr(choose_config_via_pooler_and_check_last, "leaderboard", [])
    print("\n[Top 10 leaderboard]")
    for i, row in enumerate(lb[:10], 1):
        print(f"#{i} rank={row['rank']}, stats={row['stats']}, rc_mean={row['rc_mean']:.4f}")
        cfg = {k: v for k, v in row["config"].items()
               if k in {"direction", "tau", "pooling_axis", "layer_spec", "rc_average", "channel_groups", "gem_p",
                        "head_k", "tail_k"}}
        print("   cfg:", cfg)

    res = compare_embedding_strategies(seqs, backend, verbose=True)

