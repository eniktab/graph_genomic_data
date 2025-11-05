# ===== EMBEDDING QC REPORT (one block) =======================================
# Requirements:
#   - torch
#   - embs: torch.Tensor of shape (N, H) on CUDA (or CPU), rows ordered as
#           (forward, reverse-complement) pairs for each chunk.
# Optional:
#   - global `cfg` with attributes `window` and `embed_max_len` (info note only)

from __future__ import annotations
import torch
import matplotlib.pyplot as plt
import umap  # pip install umap-learn
from typing import List, Dict, Any, Tuple

def _device_of(x: torch.Tensor) -> torch.device:
    return x.device if isinstance(x, torch.Tensor) else torch.device('cpu')

def _quantiles(x: torch.Tensor, qs=(0.05, 0.5, 0.95)):
    x = x.float()
    return torch.quantile(x, torch.tensor(qs, device=x.device)).tolist()

def _stats(name: str, v: torch.Tensor):
    v = v.float()
    m = v.mean().item()
    s = v.std(unbiased=False).item()
    mn = v.min().item()
    mx = v.max().item()
    p5, p50, p95 = _quantiles(v, (0.05, 0.5, 0.95))
    print(f"{name}: mean={m:.4f}, std={s:.4f}, p5={p5:.4f}, p50={p50:.4f}, p95={p95:.4f}, min={mn:.4f}, max={mx:.4f}")
    return dict(mean=m, std=s, p5=p5, p50=p50, p95=p95, min=mn, max=mx)

@torch.no_grad()
def embedding_qc_report(
    embs: torch.Tensor,
    far_pairs: int = 20_000,
    far_min_gap: int = 200,
    batch: int = 2048,
    rc_topk: int = 2,
    seed: int | None = 42,
):
    """
    Print diagnostics and return a dict of metrics.
    Assumes embs are appended as (forward, RC) pairs, i.e., N is even.
    """

    # --- Config sanity (optional) ---
    if 'cfg' in globals():
        try:
            if getattr(cfg, 'window', None) and getattr(cfg, 'embed_max_len', None):
                if cfg.window > cfg.embed_max_len:
                    print(f"[Note] window ({cfg.window}) > embed_max_len ({cfg.embed_max_len}). "
                          f"If you did not tile+pool internally, each vector may represent only the first {cfg.embed_max_len} bp.")
        except Exception:
            pass

    # --- Basic checks ---
    if embs.ndim != 2:
        raise ValueError("embs must be 2D (N, H).")
    N, H = embs.shape
    if N < 4:
        raise ValueError("Need at least 4 rows (2 forward+RC pairs).")
    if (N % 2) != 0:
        raise ValueError("N must be even: expected (forward, RC) pairs in order.")

    device = _device_of(embs)
    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    # --- Normalize once for cosine ---
    E = torch.nn.functional.normalize(embs, dim=1)

    # Split into forward (Ef) and RC (Erc)
    f_idx = torch.arange(0, N, 2, device=device)
    rc_idx = f_idx + 1
    Ef, Erc = E[f_idx], E[rc_idx]
    F = Ef.shape[0]

    metrics = {}

    # --- RC-invariance: cosine between each forward and its RC partner ---
    sim_rc = (Ef * Erc).sum(dim=1)  # cosine (L2-normalized rows)
    metrics['rc_similarity'] = _stats("RC(sim(f, rc(f)))", sim_rc)

    # --- Locality: forward neighbors (±1) ---
    if F >= 2:
        sim_adj = (Ef[:-1] * Ef[1:]).sum(dim=1)
        metrics['adjacency_similarity_forward'] = _stats("Adjacency(sim(f[i], f[i+1]))", sim_adj)
    else:
        sim_adj = torch.tensor([], device=device)

    # --- Far pairs: random non-neighbors with minimum index gap ---
    def random_far_pairs(M=far_pairs, min_gap=far_min_gap):
        # Try to enforce |i-j| >= min_gap; fall back to i!=j if masked out.
        i = torch.randint(0, F, (M,), device=device)
        j = torch.randint(0, F, (M,), device=device)
        mask = (i - j).abs() >= min_gap
        if mask.sum() == 0:
            mask = (i != j)
        i = i[mask]
        j = j[mask]
        if i.numel() == 0:
            # If still empty (degenerate tiny F), compare the first two (if possible)
            if F >= 2:
                return (Ef[0:1] * Ef[1:2]).sum(dim=1)
            # Else return a zero-vector to keep stats defined
            return torch.zeros(1, device=device)
        return (Ef[i] * Ef[j]).sum(dim=1)

    sim_far = random_far_pairs()
    metrics['far_similarity'] = _stats(f"Far(sim(f[i], f[j]), gap≥{far_min_gap})", sim_far)

    # --- Margin: mean(adj) - mean(far) ---
    if sim_adj.numel() > 0 and sim_far.numel() > 0:
        margin = sim_adj.mean().item() - sim_far.mean().item()
        print(f"Adjacency-Far margin: {margin:.4f}")
        metrics['adjacency_far_margin'] = margin

    # --- Retrieval diagnostics (within forward set) ---
    # A) Self should be rank-1
    # B) Nearest non-self should be positional neighbor (±1)
    self_rank1 = 0
    neighbor_ok = 0
    for a in range(0, F, batch):
        b = min(a + batch, F)
        Q = Ef[a:b]               # (B,H)
        S = Q @ Ef.T              # (B,F), cosine sims

        # Self @ rank-1?
        idxs = torch.argmax(S, dim=1)
        qs = torch.arange(a, b, device=device)
        self_rank1 += (idxs == qs).sum().item()

        # Mask self to find nearest non-self; check if that NN is ±1
        S[torch.arange(0, b - a, device=device), qs] = float("-inf")
        idxs2 = torch.argmax(S, dim=1)
        neighbor_ok += ((idxs2 - qs).abs() <= 1).sum().item()

    pct_self_rank1 = 100.0 * self_rank1 / F
    pct_neighbor_ok = 100.0 * neighbor_ok / F
    print(f"Forward self at rank-1: {pct_self_rank1:.2f}%")
    print(f"Nearest non-self is positional neighbor (±1): {pct_neighbor_ok:.2f}%")
    metrics['forward_self_rank1_pct'] = pct_self_rank1
    metrics['nearest_neighbor_is_positional_pct'] = pct_neighbor_ok

    # --- Cross-strand retrieval: does forward find its RC in top-k? ---
    rc_topk = max(1, min(int(rc_topk), F))
    rc_hits = 0
    for a in range(0, F, batch):
        b = min(a + batch, F)
        Q = Ef[a:b]               # (B,H)
        S = Q @ Erc.T             # (B,F)
        _, idxs = torch.topk(S, k=rc_topk, dim=1)
        truth = torch.arange(a, b, device=device)  # RC index equals forward index
        rc_hits += (idxs == truth.unsqueeze(1)).any(dim=1).sum().item()

    pct_rc_topk = 100.0 * rc_hits / F
    print(f"RC match found in top-{rc_topk}: {pct_rc_topk:.2f}%")
    metrics[f'rc_in_top_{rc_topk}_pct'] = pct_rc_topk

    # --- Suggested gates (tune for your data) ---
    gates = {
        "rc_mean_ge_0.85": metrics['rc_similarity']['mean'] >= 0.85,
        "adj_minus_far_margin_ge_0.35": metrics.get('adjacency_far_margin', 0.0) >= 0.35,
        "rc_topk_ge_95pct": pct_rc_topk >= 95.0,
        "neighbor_ok_ge_95pct": pct_neighbor_ok >= 95.0,
    }
    print("Gates:", {k: ("PASS" if v else "FAIL") for k, v in gates.items()})
    metrics['gates'] = gates

    # --- Final compact line ---
    compact = (
        f"RCmean={metrics['rc_similarity']['mean']:.3f}  "
        f"ADJmean={metrics.get('adjacency_similarity_forward', {}).get('mean', float('nan')):.3f}  "
        f"FARmean={metrics['far_similarity']['mean']:.3f}  "
        f"MARGIN={metrics.get('adjacency_far_margin', float('nan')):.3f}  "
        f"SELF@1={pct_self_rank1:.1f}%  "
        f"NEIGHBOR±1={pct_neighbor_ok:.1f}%  "
        f"RC@{rc_topk}={pct_rc_topk:.1f}%"
    )
    print("COMPACT:", compact)

    return metrics


# ===== UMAP helper ============================================================
@torch.no_grad()
def visualize_sequence_level_umap(
    emb: torch.Tensor,                   # [B,H] or [B,T,H]
    mask_bt: torch.Tensor | None = None, # optional [B,T] mask for token-level inputs
    pooling: str = "mean",               # "mean" | "gem"
    gem_p: float = 3.0,
    l2_normalize: bool = True,           # normalize per vector before UMAP
    n_neighbors: int = 10,
    min_dist: float = 0.1,
    metric: str = "cosine",
    labels: list[str] | None = None,     # optional labels for each sequence
    save_path: str | None = None,
    random_state: int = 42,
):
    """
    If emb is [B,T,H], we pool across T. If [B,H], we use directly.
    """

    def _masked_mean(x, m):
        m = m.to(dtype=x.dtype).unsqueeze(-1)     # [B,T,1]
        s = (x * m).sum(dim=1)                    # [B,H]
        z = m.sum(dim=1).clamp_min(1)             # [B,1]
        return s / z

    def _masked_gem(x, m, p: float):
        eps = 1e-6
        m = m.to(dtype=x.dtype).unsqueeze(-1)
        xc = x.clamp_min(eps)
        s = ((xc ** p) * m).sum(dim=1)
        z = m.sum(dim=1).clamp_min(1)
        return (s / z).clamp_min(eps) ** (1.0 / p)

    if emb.ndim == 3:
        B, T, H = emb.shape
        if mask_bt is None:
            # Unmasked mean across tokens
            if pooling == "mean":
                X = emb.mean(dim=1)  # [B,H]
            elif pooling == "gem":
                X = _masked_gem(emb, torch.ones(B, T, device=emb.device, dtype=torch.bool), gem_p)
            else:
                raise ValueError("pooling must be 'mean' or 'gem'")
        else:
            if pooling == "mean":
                X = _masked_mean(emb, mask_bt)  # [B,H]
            elif pooling == "gem":
                X = _masked_gem(emb, mask_bt, gem_p)
            else:
                raise ValueError("pooling must be 'mean' or 'gem'")
    elif emb.ndim == 2:
        B, H = emb.shape
        X = emb
    else:
        raise ValueError(f"emb must be [B,H] or [B,T,H], got {tuple(emb.shape)}")

    if l2_normalize:
        X = torch.nn.functional.normalize(X, p=2, dim=1)

    X_np = X.detach().float().cpu().numpy()
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        random_state=random_state,
    )
    Y = reducer.fit_transform(X_np)  # [B,2]

    plt.figure(figsize=(6, 5))
    plt.scatter(Y[:, 0], Y[:, 1], s=80)
    for i in range(B):
        txt = labels[i] if labels and i < len(labels) else str(i)
        plt.text(Y[i, 0], Y[i, 1], txt, fontsize=9, ha="center", va="center")
    plt.title(f"UMAP (sequence-level)  B={B}, H={H}, metric={metric}")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2"); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    return Y, reducer


@torch.no_grad()
def _pairwise_separation(X: torch.Tensor) -> Dict[str, float]:
    """
    X: [N, D], L2-normalized rows preferred but not required.
    Returns mean/min cosine distance and L2 (i != j), plus average nearest-nonself stats.
    """
    N = X.shape[0]
    if N < 2:
        return dict(mean_cos_dist=float("nan"), min_cos_dist=float("nan"),
                    mean_L2=float("nan"), min_L2=float("nan"),
                    nn_cos_sim=float("nan"), nn_L2=float("nan"))

    Xn = torch.nn.functional.normalize(X, p=2, dim=1)
    cos_sim = Xn @ Xn.T                              # [N,N]
    cos_dist = 1.0 - cos_sim
    eye = torch.eye(N, dtype=torch.bool, device=X.device)
    cos_dist_ns = cos_dist.masked_fill(eye, float("inf"))
    nn_cos_sim = (1.0 - cos_dist_ns.min(dim=1).values).mean().item()

    xx = (X**2).sum(dim=1, keepdim=True)
    L2sq = (xx + xx.T - 2.0 * (X @ X.T)).clamp_min(0.0)
    L2 = torch.sqrt(L2sq + 1e-12)
    L2_ns = L2.masked_fill(eye, float("inf"))
    nn_L2 = L2_ns.min(dim=1).values.mean().item()

    return dict(
        mean_cos_dist=float(cos_dist[~eye].mean().item()),
        min_cos_dist=float(cos_dist_ns.min().item()),
        mean_L2=float(L2[~eye].mean().item()),
        min_L2=float(L2_ns.min().item()),
        nn_cos_sim=float(nn_cos_sim),
        nn_L2=float(nn_L2),
    )

def _qc_one(pooler, seqs: List[str], *, label: str,
            far_pairs=20000, far_min_gap=2500, batch=2048, rc_topk=2) -> Dict[str, Any]:
    """
    Runs your embedding_qc_report() + extra separation metrics.
    Returns a compact dict; also prints the QC summary lines you’re used to.
    """
    E = pooler.embed(seqs)                 # [N, D]
    print(f"\n=== {label} ===")
    # Your QC (prints the big block of stats)
    qc = embedding_qc_report(E, far_pairs=far_pairs, far_min_gap=far_min_gap,
                             batch=batch, rc_topk=rc_topk)
    # Our extra separation metrics (numerical, easy to sort)
    sep = _pairwise_separation(E)
    print(f"Separation: minL2={sep['min_L2']:.4f}  minCos={sep['min_cos_dist']:.4f}  "
          f"meanL2={sep['mean_L2']:.4f}  meanCos={sep['mean_cos_dist']:.4f}  "
          f"NN_cos_sim={sep['nn_cos_sim']:.4f}  NN_L2={sep['nn_L2']:.4f}")
    return dict(label=label, **pooler.get_config(), **sep)

def compare_configs_with_qc(
    backend,
    seqs: List[str],
    *,
    taus: Tuple[float, ...] = (48.0, 56.0, 64.0, 72.0, 80.0),
    test_layer_mixes: bool = True,
    test_rc_average: bool = True,
    far_pairs=20000, far_min_gap=2500, batch=2048, rc_topk=2
) -> List[Dict[str, Any]]:
    """
    Evaluates: tau sweep around 64, 2–3 last-k layer mixes, and RC-average toggle.
    Returns a list of result dicts sorted by (min_L2, min_cos_dist) descending.
    """
    results = []

    # --- τ sweep (winner path: exp_left, position, layer=-7, RC off) ---
    for tau in taus:
        pooler = backend.build_pooler(direction="exp_left", tau=tau,
                                      pooling_axis="position", layer_spec=-7, rc_average=False)
        results.append(_qc_one(pooler, seqs, label=f"exp_left tau={tau:.1f}",
                               far_pairs=far_pairs, far_min_gap=far_min_gap,
                               batch=batch, rc_topk=rc_topk))

    # --- light layer mixes (keeps pooling path; can lift RC slightly) ---
    if test_layer_mixes:
        for k in (2, 3):
            pooler = backend.build_pooler(direction="exp_left", tau=64.0,
                                          pooling_axis="layers→position", layer_spec=("last_k", k),
                                          rc_average=False)
            results.append(_qc_one(pooler, seqs, label=f"layers→position last_k={k}",
                                   far_pairs=far_pairs, far_min_gap=far_min_gap,
                                   batch=batch, rc_topk=rc_topk))

    # --- optional RC averaging on the winner path (expect lower separation) ---
    if test_rc_average:
        pooler = backend.build_pooler(direction="exp_left", tau=64.0,
                                      pooling_axis="position", layer_spec=-7, rc_average=True)
        results.append(_qc_one(pooler, seqs, label="exp_left tau=64 RC-avg",
                               far_pairs=far_pairs, far_min_gap=far_min_gap,
                               batch=batch, rc_topk=rc_topk))

    # sort for clustering use-case: maximize worst-case separation
    results.sort(key=lambda r: (r["min_L2"], r["min_cos_dist"], r["mean_L2"], r["mean_cos_dist"]), reverse=True)

    # neat printout
    print("\n=== RANKED (by min_L2, then min_cos_dist) ===")
    for r in results:
        cfg = f"{r['direction']} τ={r['tau']:.1f}, axis={r['pooling_axis']}, layer={r['layer_spec']}, RCavg={r['rc_average']}"
        print(f"{cfg:70s}  minL2={r['min_L2']:.4f}  minCos={r['min_cos_dist']:.4f}  "
              f"meanL2={r['mean_L2']:.4f}  meanCos={r['mean_cos_dist']:.4f}  "
              f"NNcos={r['nn_cos_sim']:.4f}  NN_L2={r['nn_L2']:.4f}")

    return results



# Run the report
#metrics = embedding_qc_report(embs_best, far_pairs=20000, far_min_gap=2500, batch=2048, rc_topk=2)

# Optional: UMAP on the forward set only
#_ = visualize_sequence_level_umap(emb, l2_normalize=True, n_neighbors=15, min_dist=0.05)

_ = compare_configs_with_qc(
    backend_hy,
    seqs,
    taus=(56.0, 64.0, 72.0),      # quick sweep
    test_layer_mixes=True,        # try last_k=2,3
    test_rc_average=True,         # see RC-avg tradeoff
    far_pairs=20000,
    far_min_gap=2500,
    batch=2048,
    rc_topk=2
)