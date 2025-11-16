# evaluate_index_correctness.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Literal
import glob
import json
import math
import random

import numpy as np
import torch
import pandas as pd

# Your ANN engine
from src.RaftGPU import RaftGPU  # type: ignore
# Your Hyena backend
from src.HyenaBackend import HyenaBackend
try:
    from src.HyenaBackend import HyenaDNAPooler
except Exception:
    HyenaDNAPooler = None


# ----------------------------- IO helpers -----------------------------

def _raft_attr(x, *names, default=None):
    for n in names:
        if hasattr(x, n):
            return getattr(x, n)
    return default

def _discover_shards(aligned_dir: Path) -> List[Path]:
    shard_dir = aligned_dir / "shards"
    paths = sorted(Path(p) for p in glob.glob(str(shard_dir / "aligned-*.pt")))
    if not paths:
        raise FileNotFoundError(f"No aligned shards found under {shard_dir}")
    return paths

def _load_manifest(dataset_dir: Path) -> Dict[str, Any]:
    mf = dataset_dir / "manifest.json"
    if mf.is_file():
        return json.loads(mf.read_text())
    return {}

def _read_meta_csv(aligned_dir: Path) -> pd.DataFrame:
    meta = aligned_dir / "meta.csv"
    if not meta.is_file():
        raise FileNotFoundError(f"Missing {meta}")
    return pd.read_csv(meta)

def _ensure_numpy(x):
    try:
        import cupy as cp  # type: ignore
        if isinstance(x, cp.ndarray):
            return x.get()
    except Exception:
        pass
    return x

# ----------------------------- reducers -------------------------------

def _reduce_token_tensor(
    H_t: torch.Tensor,            # [T_pad, H]
    mask_t: torch.Tensor,         # [T_pad] bool
    how: Literal["center", "mean_masked"] = "center"
) -> torch.Tensor:                # [H], fp32
    T = int(mask_t.sum().item())
    if T <= 0:
        return torch.zeros(H_t.shape[-1], dtype=torch.float32)
    if how == "center":
        idx = T // 2
        return H_t[idx].to(torch.float32)
    return H_t[:T].mean(dim=0).to(torch.float32)

# ----------------------------- vectorizers ----------------------------

@torch.inference_mode()
def _embed_to_vectors(
    backend: HyenaBackend,
    sequences: List[str],
    which: Literal["best", "last"],
    reducer: Literal["center", "mean_masked"],
    *,
    batch_size: int,
    best_layer_spec: int,
) -> np.ndarray:
    """
    Embed sequences with Hyena (no pooling), pick layer ('last' or 'best'),
    reduce per-token states to one vector per sample, and return float32 [N, H].
    """
    tok = backend.tokenizer
    dev = backend.device
    rows: List[torch.Tensor] = []

    for i in range(0, len(sequences), batch_size):
        chunk = sequences[i:i+batch_size]
        enc = tok(chunk, padding="longest", truncation=True,
                  max_length=getattr(backend, "max_length", None), return_tensors="pt")
        ids = enc["input_ids"].to(dev, non_blocking=True)  # [B,T]
        mask = (ids != int(tok.pad_token_id)).to(dev)
        out = backend.model(ids)
        hs_list = out.hidden_states if hasattr(out, "hidden_states") else [out.last_hidden_state]

        if which == "last":
            L = hs_list[-1]                     # [B,T,H]
        else:
            try:
                L = hs_list[best_layer_spec]
            except Exception:
                idx = (len(hs_list) + best_layer_spec) if best_layer_spec < 0 else best_layer_spec
                idx = max(0, min(len(hs_list)-1, idx))
                L = hs_list[idx]

        # reduce each sample
        B = L.shape[0]
        vecs = []
        for b in range(B):
            vecs.append(_reduce_token_tensor(L[b], mask[b], reducer))
        V = torch.stack(vecs, dim=0).contiguous()   # [B,H] fp32
        rows.append(V.to("cpu", non_blocking=True))

        del out, hs_list, L, ids, mask, V

    X = torch.cat(rows, dim=0).numpy().astype(np.float32, copy=False)
    return X


# ----------------------------- dataset readers ------------------------

def _rebuild_aligned_text_windows(dataset_dir: Path, window: int) -> Tuple[List[str], List[int]]:
    """
    Reconstruct the aligned windows (strings) and their starts from FASTA is not required;
    we rely on shard order + meta.csv for starts, and we read sequence content from shards’ input_ids
    if needed. Here we just return starts and let the caller slice from the original FASTA if provided.
    For query generation, most users prefer to take the exact sequences again from FASTA.
    """
    df = _read_meta_csv(dataset_dir / "aligned")
    starts = df["start"].astype(int).tolist()
    return [], starts

def _gather_index_metas_in_build_order(dataset_dir: Path) -> List[Dict[str, Any]]:
    """
    Recreate metas in the exact order the index builder used:
    - iterate shards sorted by name
    - append per-sample metas in shard order
    """
    aligned_dir = dataset_dir / "aligned"
    shard_paths = _discover_shards(aligned_dir)
    metas: List[Dict[str, Any]] = []
    for sp in shard_paths:
        obj = torch.load(sp, map_location="cpu")
        n = obj["input_ids"].shape[0]
        starts = obj["starts"].tolist()
        ends = obj["ends"].tolist()
        labels = obj["labels"].tolist()
        for i in range(n):
            metas.append({
                "start": int(starts[i]),
                "end": int(ends[i]),
                "label": int(labels[i]),
                "source_shard": sp.name,
                "row_in_shard": i,
            })
        del obj
    return metas


# ----------------------------- build or load index --------------------

def build_or_load_index(
    index_outdir: str | Path,
    dataset_dir: str | Path,
    *,
    which: Literal["best", "last"] = "best",
    reducer: Literal["center", "mean_masked"] = "center",
    metric: Literal["cosine", "l2"] = "cosine",
    build_algo: Literal["nn_descent", "ivf_pq"] = "nn_descent",
    graph_degree: int = 64,
    inter_graph_degree: int = 128,
    nn_descent_niter: int = 20,
    ivf_pq_dim: int = 64,
    ivf_pq_bits: int = 8,
    ivf_n_lists: Optional[int] = None,
    ivf_n_probes: Optional[int] = None,
    limit: Optional[int] = None,
) -> Tuple[RaftGPU, List[Dict[str, Any]]]:
    """
    If an index exists in index_outdir, load it and its metas; else vectorize aligned split and build it.
    Returns (raft, metas).
    """
    index_outdir = Path(index_outdir)
    dataset_dir = Path(dataset_dir)
    index_outdir.mkdir(parents=True, exist_ok=True)

    # Try to load
    try:
        raft = RaftGPU.load(index_outdir)  # type: ignore[attr-defined]
        # try to get metas from the raft object
        if hasattr(raft, "metas") and isinstance(raft.metas, list):
            metas = raft.metas
        else:
            # fallback: rebuild metas from shards
            metas = _gather_index_metas_in_build_order(dataset_dir)
        return raft, metas
    except Exception:
        pass  # build below

    # Build: vectorize aligned split exactly like in your builder
    from src._drafts.current.make_index import _vectorize_aligned_split  # reuse your helper
    aligned_dir = dataset_dir / "aligned"
    X, metas = _vectorize_aligned_split(
        aligned_dir, which=which, reducer=reducer, dtype="fp16", limit=limit
    )  # [N,D] fp32

    raft = RaftGPU(
        dim=X.shape[1], metric=metric, build_algo=build_algo,
        graph_degree=graph_degree, intermediate_graph_degree=inter_graph_degree,
        nn_descent_niter=nn_descent_niter,
        ivf_pq_dim=ivf_pq_dim, ivf_pq_bits=ivf_pq_bits,
        ivf_n_lists=ivf_n_lists, ivf_n_probes=ivf_n_probes,
    )
    raft.add(X, metas)
    raft.save(index_outdir, include_dataset=True)
    return raft, metas


# ----------------------------- query generators -----------------------

def make_aligned_queries_from_fasta(
    fasta: str | Path,
    contig: str,
    starts: List[int],
    window: int,
    n: int,
    seed: int = 13
) -> List[str]:
    """Take a random subset of aligned windows from FASTA (exact sequences)."""
    import pysam
    rng = random.Random(seed)
    with pysam.FastaFile(str(fasta)) as fa:
        L = len(starts)
        choose = starts if n >= L else rng.sample(starts, n)
        seqs = [fa.fetch(contig, s, s + window).upper() for s in choose]
    return seqs

def make_misaligned_queries_from_fasta(
    fasta: str | Path,
    contig: str,
    window: int,
    stride: int,
    offset: int,
    n: int,
    seed: int = 17
) -> Tuple[List[str], List[int]]:
    """Make n misaligned windows (start≡offset mod stride). Returns (seqs, true_starts)."""
    import pysam
    rng = random.Random(seed)
    with pysam.FastaFile(str(fasta)) as fa:
        L = fa.get_reference_length(contig)
        max_start = L - window
        positions = list(range(offset, max_start + 1, stride))
        choose = positions if n >= len(positions) else rng.sample(positions, n)
        seqs = [fa.fetch(contig, s, s + window).upper() for s in choose]
    return seqs, choose


# ----------------------------- evaluation -----------------------------

def _infer_best_layer_spec(dataset_dir: Path, default: int = -7) -> int:
    mani = _load_manifest(dataset_dir)
    hl = mani.get("hidden_states_saved")
    if isinstance(hl, list):
        for e in hl:
            if isinstance(e, str) and e.startswith("best:"):
                try:
                    return int(e.split("best:")[1])
                except Exception:
                    pass
    return default

def evaluate_correctness(
    *,
    dataset_dir: str | Path,
    index_outdir: str | Path,
    fasta: str | Path,
    contig: str,
    window: int,
    stride: int,
    which: Literal["best", "last"] = "best",
    reducer: Literal["center", "mean_masked"] = "center",
    n_id_queries: int = 128,
    n_mis_queries: int = 128,
    mis_offset: Optional[int] = None,         # None → from dataset manifest
    # Hyena config for queries:
    model_name: str = "hyenadna-large-1m-seqlen-hf",
    model_dir: str = "/g/data/te53/en9803/data/scratch/hf-cache/models/",
    batch: int = 128,
) -> Dict[str, Any]:
    """
    Runs:
      (A) Identity check on aligned windows
      (B) Misaligned check with fixed offset
    Prints a small report and returns metrics.
    """
    dataset_dir = Path(dataset_dir)
    aligned_dir = dataset_dir / "aligned"

    # Load or build index and its metas
    raft, metas = build_or_load_index(index_outdir, dataset_dir,
                                      which=which, reducer=reducer)

    # Map: index row -> start (from metas order)
    idx2start = np.array([m["start"] for m in metas], dtype=np.int64)

    # Starts for aligned windows (from CSV)
    df_meta = _read_meta_csv(aligned_dir)
    aligned_starts = df_meta["start"].astype(int).tolist()

    # Best-layer spec inference
    best_spec = _infer_best_layer_spec(dataset_dir, default=-7)

    # Hyena backend for queries (no pooling, raw states)
    backend = HyenaBackend(model_name=model_name, model_dir=model_dir,
                           pooling="none", normalize=False, offline=True, prefer_cuda=True)

    # (A) Identity check
    id_q = make_aligned_queries_from_fasta(fasta, contig, aligned_starts, window, n_id_queries)
    X_id = _embed_to_vectors(backend, id_q, which=which, reducer=reducer,
                             batch_size=batch, best_layer_spec=best_spec)
    D_id, I_id = raft.search(X_id, k=10)  # device arrays or numpy
    D_id, I_id = _ensure_numpy(D_id), _ensure_numpy(I_id)

    # Ground truth indices for identity queries:
    # We sampled sequences at specific aligned starts; rebuild starts chosen by re-fetching:
    # Easiest: take the first n_id_queries starts used by make_aligned_queries_from_fasta via sampling again.
    # To ensure determinism across calls, we re-run the selection here:
    chosen_starts = make_aligned_queries_from_fasta(fasta, contig, aligned_starts, window, n_id_queries)
    # But the function returns seqs, not starts; keep it simple: align by content is expensive.
    # Instead, draw deterministically ourselves:
    rng = random.Random(13)
    choose = aligned_starts if n_id_queries >= len(aligned_starts) else rng.sample(aligned_starts, n_id_queries)
    gt_aligned = np.array(choose, dtype=np.int64)

    # Convert ground truth starts to ground-truth index rows
    # (metas were appended in shard order; we build a map start→list(rows))
    start2rows: Dict[int, List[int]] = {}
    for i, s in enumerate(idx2start.tolist()):
        start2rows.setdefault(int(s), []).append(i)
    gt_rows = np.array([start2rows[st][0] for st in gt_aligned], dtype=np.int64)

    # Compute recall@k and MRR for identity
    def _recall_at(I: np.ndarray, gt: np.ndarray, k: int) -> float:
        hits = 0
        for i in range(I.shape[0]):
            if int(gt[i]) in I[i, :k]:
                hits += 1
        return hits / float(I.shape[0])

    def _mrr(I: np.ndarray, gt: np.ndarray) -> float:
        rr = 0.0
        for i in range(I.shape[0]):
            row = I[i]
            try:
                pos = np.where(row == gt[i])[0]
                rr += 1.0 / (1 + int(pos[0])) if pos.size else 0.0
            except Exception:
                pass
        return rr / float(I.shape[0])

    id_r1  = _recall_at(I_id, gt_rows, 1)
    id_r5  = _recall_at(I_id, gt_rows, 5)
    id_r10 = _recall_at(I_id, gt_rows, 10)
    id_mrr = _mrr(I_id, gt_rows)

    # (B) Misaligned check
    mani = _load_manifest(dataset_dir)
    off = int(mis_offset if mis_offset is not None else mani.get("misalign_offset", stride // 2))
    mis_q, mis_starts = make_misaligned_queries_from_fasta(fasta, contig, window, stride, off, n_mis_queries)
    X_mis = _embed_to_vectors(backend, mis_q, which=which, reducer=reducer,
                              batch_size=batch, best_layer_spec=best_spec)
    D_mis, I_mis = raft.search(X_mis, k=10)
    D_mis, I_mis = _ensure_numpy(D_mis), _ensure_numpy(I_mis)

    # Expected aligned window for a misaligned start s is floor(s/stride)*stride
    exp_aligned = np.array([ (s // stride) * stride for s in mis_starts ], dtype=np.int64)
    gt_rows_mis = np.array([start2rows.get(int(st), [None])[0] for st in exp_aligned], dtype=object)
    # Some exp_aligned near tail might be missing if that window was filtered; drop Nones.
    mask_valid = np.array([r is not None for r in gt_rows_mis], dtype=bool)
    gt_rows_mis_valid = np.array([r for r in gt_rows_mis if r is not None], dtype=np.int64)
    I_mis_valid = I_mis[mask_valid]

    mis_r1  = _recall_at(I_mis_valid, gt_rows_mis_valid, 1) if len(I_mis_valid) else float('nan')
    mis_r5  = _recall_at(I_mis_valid, gt_rows_mis_valid, 5) if len(I_mis_valid) else float('nan')
    mis_r10 = _recall_at(I_mis_valid, gt_rows_mis_valid, 10) if len(I_mis_valid) else float('nan')
    mis_mrr = _mrr(I_mis_valid, gt_rows_mis_valid) if len(I_mis_valid) else float('nan')

    # Pretty report (similar spirit to your printouts)
    print("\n======================================================================")
    print("CORRECTNESS VALIDATION")
    print("======================================================================")
    print(f"  Config: which={which}, reducer={reducer}, best_layer_spec={best_spec}, metric={raft.metric}")
    _algo = getattr(raft, "build_algo", getattr(raft, "_build_algo", "unknown"))
    print(f"  Index:  N={len(metas):,}, D={getattr(raft, 'dim', '?')}, algo={_algo}")
    print("\n  Identity (aligned windows as queries)")
    print(f"    Recall@1:  {id_r1:8.3%}")
    print(f"    Recall@5:  {id_r5:8.3%}")
    print(f"    Recall@10: {id_r10:8.3%}")
    print(f"    MRR:       {id_mrr:8.4f}")

    print("\n  Misaligned (offset windows as queries)")
    print(f"    offset:    {off}")
    print(f"    Valid:     {len(I_mis_valid)}/{len(I_mis)}")
    if len(I_mis_valid):
        print(f"    Recall@1:  {mis_r1:8.3%}")
        print(f"    Recall@5:  {mis_r5:8.3%}")
        print(f"    Recall@10: {mis_r10:8.3%}")
        print(f"    MRR:       {mis_mrr:8.4f}")

    return {
        "identity": {"r1": id_r1, "r5": id_r5, "r10": id_r10, "mrr": id_mrr},
        "misaligned": {"offset": off, "valid": int(len(I_mis_valid)),
                       "total": int(len(I_mis)), "r1": mis_r1, "r5": mis_r5, "r10": mis_r10, "mrr": mis_mrr},
        "index": {"N": len(metas), "D": raft.dim, "metric": raft.metric, "algo": raft.build_algo},
        "config": {"which": which, "reducer": reducer, "best_layer_spec": best_spec}
    }

if __name__ == "__main__":
    metrics = evaluate_correctness(
        dataset_dir="/g/data/te53/en9803/sandpit/graph_genomics/chr22/windows_raw",
        index_outdir="/g/data/te53/en9803/sandpit/graph_genomics/chr22/index_center_best_cosine",
        fasta="/g/data/te53/en9803/sandpit/graph_genomics/chr22/chm13v2_chr22.fa.gz",
        contig="chr22",
        window=10_000,
        stride=5_000,
        which="best",                # layer used by index ('best' or 'last')
        reducer="center",            # reduction used by index ('center' or 'mean_masked')
        n_id_queries=128,
        n_mis_queries=128,
        # mis_offset=None           # default reads it from dataset manifest
    )
    print(metrics)