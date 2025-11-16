# hel_index_two_runs.py
from __future__ import annotations
from pathlib import Path
import json
from typing import Any, List, Dict, Tuple, Optional

import torch

# Try both import styles depending on your layout
try:
    from src.HELIndexer import HELIndexer  # project-style
    from src.configs import IndexConfig
except Exception:
    from HELIndexer import HELIndexer      # flat file next to this script
    from configs import IndexConfig

try:
    import pysam
except Exception as e:
    raise SystemExit("pysam is required. Try: pip install pysam") from e


# ------------------------------
# FASTA helpers
# ------------------------------
def _ensure_faidx(fa_path: Path) -> None:
    """Ensure .fai exists for FASTA."""
    fai = fa_path.with_suffix(fa_path.suffix + ".fai")
    if not fai.exists():
        pysam.faidx(str(fa_path))


def _pick_sequences(
    ref_fa: Path,
    window: int,
    stride: int,
    n: int = 32,
) -> List[Dict[str, Any]]:
    """
    Grab up to n windows from the first contig for reproducible test queries.
    Steps through the contig with the provided stride; stops before overflow.
    """
    ff = pysam.FastaFile(str(ref_fa))
    try:
        if not ff.references:
            raise RuntimeError("No contigs found in FASTA.")
        chrom = ff.references[0]
        clen = ff.get_reference_length(chrom)
        if clen < window:
            raise RuntimeError(f"First contig shorter than window (len={clen}, window={window}).")

        seqs: List[Dict[str, Any]] = []
        pos = 0
        while len(seqs) < n and pos + window <= clen:
            s = ff.fetch(chrom, pos, pos + window).upper()
            seqs.append({"chrom": chrom, "start": int(pos), "seq": s})
            pos += stride if stride > 0 else window
        return seqs
    finally:
        ff.close()


# ------------------------------
# Small utils
# ------------------------------
def _has_cupy(x: Any) -> bool:
    return x.__class__.__module__.split(".", 1)[0] == "cupy" if hasattr(x, "__class__") else False


def _to_torch(x: Any, *, dtype: torch.dtype, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convert x -> torch.Tensor without touching NumPy.
    Supports: torch.Tensor, CuPy ndarray (via DLPack), Python lists/tuples.
    If device is given, ensures the tensor is on that device.
    """
    if isinstance(x, torch.Tensor):
        t = x.to(dtype=dtype)
        return t.to(device) if device is not None else t

    if _has_cupy(x):
        # CuPy -> Torch via DLPack (keeps it on GPU)
        import cupy as cp  # local import to avoid hard dep if unused
        t = torch.utils.dlpack.from_dlpack(x.toDlpack()).to(dtype)
        return t.to(device) if device is not None else t

    # Fallback: construct from Python sequence
    t = torch.tensor(x, dtype=dtype, device=device if device is not None else None)
    return t


# ------------------------------
# Raft search wrapper (Torch I/O)
# ------------------------------
def _raft_search_torch(raft, Q: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run top-k search and return (idx [B,k] long, dist [B,k] float) as Torch tensors.
    - Accepts Torch Q directly (prefers CUDA if available).
    - Converts any outputs (Torch/CuPy/Python list) back to Torch via DLPack or tensor().
    """
    # Prefer to pass Torch directly; RaftGPU should handle dlpack internally
    if hasattr(raft, "search"):
        out = raft.search(Q, k=k)  # expected (idx, dist) in various array types
        idx_raw, dist_raw = out
    elif hasattr(raft, "kneighbors"):
        # sklearn-like API sometimes returns (dist, idx)
        out = raft.kneighbors(Q, n_neighbors=k, return_distance=True)
        dist_raw, idx_raw = out
    else:
        raise RuntimeError("RaftGPU object has neither .search nor .kneighbors.")

    # Normalize to Torch tensors (no NumPy)
    dev = Q.device
    idx_t = _to_torch(idx_raw, dtype=torch.long, device=dev)
    dist_t = _to_torch(dist_raw, dtype=torch.float32, device=dev)
    return idx_t, dist_t


# ------------------------------
# Drift metrics (Torch-only)
# ------------------------------
def _cosine_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    # a, b: [B, D]
    a_n = torch.linalg.norm(a, dim=1).clamp_min(eps)
    b_n = torch.linalg.norm(b, dim=1).clamp_min(eps)
    cos_sim = (a * b).sum(dim=1) / (a_n * b_n)
    return 1.0 - cos_sim  # 0 == identical direction


def _embedding_drift(e1: torch.Tensor, e2: torch.Tensor) -> Dict[str, Any]:
    """
    Return per-query drift + summary (Torch-only):
      - l2: ||e2 - e1||2
      - rel_l2: ||e2 - e1||2 / ||e1||2
      - cos: cosine distance (1 - cosine similarity)
    """
    assert e1.shape == e2.shape, f"shape mismatch: {tuple(e1.shape)} vs {tuple(e2.shape)}"
    with torch.no_grad():
        diff = e2 - e1
        l2 = torch.linalg.norm(diff, dim=1)
        l2_ref = torch.linalg.norm(e1, dim=1).clamp_min(1e-9)
        rel_l2 = l2 / l2_ref
        cos = _cosine_distance(e1, e2)

    def _summ(x: torch.Tensor) -> Dict[str, float]:
        return {"mean": float(x.mean().item()),
                "median": float(x.median().item()),
                "max": float(x.max().item())}

    return {
        "per_query": {
            "l2": l2.detach().cpu().tolist(),
            "rel_l2": rel_l2.detach().cpu().tolist(),
            "cosine_distance": cos.detach().cpu().tolist(),
        },
        "summary": {
            "l2": _summ(l2),
            "rel_l2": _summ(rel_l2),
            "cosine_distance": _summ(cos),
        },
    }


def _neighbors_drift_torch(
    idx1: torch.Tensor,
    dist1: torch.Tensor,
    idx2: torch.Tensor,
    dist2: torch.Tensor,
    topk: int,
) -> Dict[str, Any]:
    """
    Compare neighbor lists from two runs (Torch-only):
    - top1_agreement: fraction of queries where top-1 index matches
    - jaccard@k: mean Jaccard between sets of k neighbors (computed via Python sets from Torch lists)
    - top1_dist_delta: per-query |d2_0 - d1_0|
    """
    assert idx1.shape == idx2.shape and dist1.shape == dist2.shape
    assert idx1.dim() == 2 and idx2.dim() == 2 and dist1.dim() == 2 and dist2.dim() == 2
    assert idx1.size(1) >= 1 and idx1.size(1) >= topk
    B = idx1.size(0)

    # top-1 equality (Torch)
    top1_eq = (idx1[:, 0] == idx2[:, 0]).float()
    top1_agreement = float(top1_eq.mean().item())

    # jaccard@k (convert each row to Python sets from Torch without NumPy)
    jaccs: List[float] = []
    idx1_cpu = idx1[:, :topk].detach().cpu()
    idx2_cpu = idx2[:, :topk].detach().cpu()
    for i in range(B):
        s1 = set(int(x) for x in idx1_cpu[i].tolist())
        s2 = set(int(x) for x in idx2_cpu[i].tolist())
        inter = len(s1 & s2)
        union = len(s1 | s2)
        jaccs.append(0.0 if union == 0 else inter / union)
    jaccard_k = sum(jaccs) / len(jaccs) if jaccs else 0.0

    # top-1 distance deltas (Torch)
    top1_dd = (dist2[:, 0] - dist1[:, 0]).abs()
    return {
        "top1_agreement": top1_agreement,
        "jaccard_at_k": float(jaccard_k),
        "top1_distance_delta": {
            "mean": float(top1_dd.mean().item()),
            "median": float(top1_dd.median().item()),
            "max": float(top1_dd.max().item()) if top1_dd.numel() else 0.0,
        },
        "per_query_top1_index_equal": top1_eq.detach().cpu().tolist(),
        "per_query_top1_abs_dist_delta": top1_dd.detach().cpu().tolist(),
    }


# ------------------------------
# Build once, save artifacts
# ------------------------------
def build_once_and_save(
    cfg: IndexConfig,          # e.g., IndexConfig(window=WINDOW, stride=5000, rc_index=True)
    tiny_fa: Path,             # e.g., work / "tiny.fa"
    outdir: Path | None = None,
    *,
    embedder: str = "hyena",   # "det" | "hyena" | "nt"
    emb_batch: int = 512,
    n_queries: int = 32,
    top_k: int = 5,
    save_prefix: str = "run1",
) -> Path:
    """
    Instantiate HELIndexer once, build & save index, embed a query set, and save:
      - queries JSON
      - q_emb Torch (.pt)
      - neighbors JSON (indices/distances)
      - hel/meta JSON
    Returns the outdir used.
    """
    outdir = Path(outdir) if outdir is not None else (Path(tiny_fa).parent / "hel_index_out")
    outdir.mkdir(parents=True, exist_ok=True)
    _ensure_faidx(Path(tiny_fa))

    indexer = HELIndexer(
        ref_fasta=Path(tiny_fa),
        cfg=cfg,
        embedder=embedder,
        emb_batch=emb_batch,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    indexer.build_or_load(outdir, reuse_existing=False, include_dataset=True, verbose=True)

    meta = indexer.info()
    print("\n[BUILD] HEL meta:")
    print(json.dumps({k: meta[k] for k in ["embedding_dim", "n_vectors", "metric", "window", "stride"]}, indent=2))
    print(f"[BUILD] RaftGPU: dim={indexer.index.dim}, metric={indexer.index.metric}, py_obj_id={id(indexer.index)}")

    # Queries
    queries = _pick_sequences(Path(tiny_fa), window=cfg.window, stride=cfg.stride, n=n_queries)
    qseqs = [q["seq"] for q in queries]

    # Embeddings
    with torch.inference_mode():
        q_emb = indexer.embedder.embed_best(qseqs)  # [B, D] (GPU if available)
        if torch.cuda.is_available() and q_emb.device.type != "cuda":
            q_emb = q_emb.to("cuda")
        q_emb = q_emb.float().contiguous()

    # Show a little head for sanity
    print("\n[BUILD] Sample query embeddings (first 2, head8):")
    for i, e in enumerate(q_emb[:2]):
        e_head = e[:8].detach().float().cpu().tolist()
        q = queries[i]
        print(f"  q{i}: {q['chrom']}:{q['start']}-{q['start']+cfg.window}  head8={e_head}")

    # Neighbors (Torch-only)
    idx_t, dist_t = _raft_search_torch(indexer.index, q_emb, k=top_k)
    print("\n[BUILD] Top-1 neighbors in RaftGPU index (first 5):")
    for i in range(min(5, len(queries))):
        print(f"  q{i}: nn_index={int(idx_t[i,0].item())}, nn_dist={float(dist_t[i,0].item())}")

    # Save artifacts (avoid NumPy; use Torch + JSON)
    (outdir / f"{save_prefix}_queries.json").write_text(json.dumps(queries, indent=2))
    torch.save(q_emb.detach().cpu(), outdir / f"{save_prefix}_q_emb.pt")
    neighbors_payload = {
        "idx": idx_t.detach().cpu().tolist(),
        "dist": dist_t.detach().cpu().tolist(),
        "top_k": top_k,
    }
    (outdir / f"{save_prefix}_neighbors.json").write_text(json.dumps(neighbors_payload, indent=2))
    (outdir / f"{save_prefix}_hel_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n[BUILD] Saved queries -> {outdir / (save_prefix + '_queries.json')}")
    print(f"[BUILD] Saved embeddings -> {outdir / (save_prefix + '_q_emb.pt')}")
    print(f"[BUILD] Saved neighbors -> {outdir / (save_prefix + '_neighbors.json')}")
    print(f"[BUILD] Index saved in: {outdir} (manifest.json, hel_meta.json as managed by HELIndexer)")
    return outdir


# ------------------------------
# Load again, compare & report drift
# ------------------------------
def load_again_compare_and_print(
    cfg: IndexConfig,          # same IndexConfig as build
    tiny_fa: Path,             # same FASTA path as build
    outdir: Path | None = None,
    *,
    embedder: str = "hyena",
    emb_batch: int = 512,
    load_prefix: str = "run1",
    n_queries_override: int | None = None,  # if provided, truncate queries to this count for comparison
) -> None:
    """
    Instantiate a second HELIndexer, LOAD the saved index, re-embed the same queries,
    and compute/report embedding + retrieval drift against saved artifacts from first run.
    """
    outdir = Path(outdir) if outdir is not None else (Path(tiny_fa).parent / "hel_index_out")
    _ensure_faidx(Path(tiny_fa))

    # Load saved artifacts from run1
    qfile = outdir / f"{load_prefix}_queries.json"
    emb_file = outdir / f"{load_prefix}_q_emb.pt"
    neigh_file = outdir / f"{load_prefix}_neighbors.json"
    meta1_file = outdir / f"{load_prefix}_hel_meta.json"

    if not qfile.exists() or not emb_file.exists() or not neigh_file.exists():
        raise FileNotFoundError("Missing run1 artifacts. Run build_once_and_save(...) first.")

    queries = json.loads(qfile.read_text())
    q_emb1_cpu: torch.Tensor = torch.load(emb_file, map_location="cpu")  # [B, D] cpu
    neigh1 = json.loads(neigh_file.read_text())
    meta1 = json.loads(meta1_file.read_text())
    top_k = int(neigh1.get("top_k", 5))

    if n_queries_override is not None:
        queries = queries[:n_queries_override]
        q_emb1_cpu = q_emb1_cpu[: len(queries)]
        neigh1["idx"] = neigh1["idx"][: len(queries)]
        neigh1["dist"] = neigh1["dist"][: len(queries)]

    # 1) Instantiate & LOAD
    indexer2 = HELIndexer(
        ref_fasta=Path(tiny_fa),
        cfg=cfg,
        embedder=embedder,
        emb_batch=emb_batch,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    indexer2.build_or_load(outdir, reuse_existing=True, include_dataset=True, verbose=True)

    meta2 = indexer2.info()
    print("\n[LOAD] HEL meta:")
    print(json.dumps({k: meta2[k] for k in ["embedding_dim", "n_vectors", "metric", "window", "stride"]}, indent=2))
    print(f"[LOAD] RaftGPU: dim={indexer2.index.dim}, metric={indexer2.index.metric}, py_obj_id={id(indexer2.index)}")

    # Basic meta comparability
    meta_comp = {
        "embedding_dim_equal": (meta1.get("embedding_dim") == meta2.get("embedding_dim")),
        "n_vectors_equal": (meta1.get("n_vectors") == meta2.get("n_vectors")),
        "metric_equal": (meta1.get("metric") == meta2.get("metric")),
        "window_equal": (meta1.get("window") == meta2.get("window")),
        "stride_equal": (meta1.get("stride") == meta2.get("stride")),
    }
    print("\n[COMPARE] Index meta equality (run1 vs run2):")
    print(json.dumps(meta_comp, indent=2))

    # 2) Re-embed the exact same queries
    qseqs = [q["seq"] for q in queries]
    with torch.inference_mode():
        q_emb2 = indexer2.embedder.embed_best(qseqs)
        if torch.cuda.is_available() and q_emb2.device.type != "cuda":
            q_emb2 = q_emb2.to("cuda")
        q_emb2 = q_emb2.float().contiguous()

    # 3) Embedding drift (against saved run1 embeddings)
    q_emb1 = q_emb1_cpu.to(q_emb2.device).float().contiguous()
    drift = _embedding_drift(q_emb1, q_emb2)

    print("\n[EMBEDDING DRIFT] summary (run1 -> run2):")
    print(json.dumps(drift["summary"], indent=2))

    # 4) Retrieval drift — run2 neighbors (Torch) vs run1 neighbors (JSON->Torch)
    idx1_t = torch.tensor(neigh1["idx"], dtype=torch.long, device=q_emb2.device)
    dist1_t = torch.tensor(neigh1["dist"], dtype=torch.float32, device=q_emb2.device)
    idx2_t, dist2_t = _raft_search_torch(indexer2.index, q_emb2, k=top_k)

    rdrift = _neighbors_drift_torch(idx1_t, dist1_t, idx2_t, dist2_t, topk=top_k)

    print("\n[RETRIEVAL DRIFT] (run1 vs run2):")
    print(json.dumps(
        {
            "top1_agreement": rdrift["top1_agreement"],
            "jaccard_at_k": rdrift["jaccard_at_k"],
            "top1_distance_delta": rdrift["top1_distance_delta"],
        },
        indent=2,
    ))

    # 5) A small sample printout for transparency
    print("\n[SAMPLE] First 3 queries — top-1 run1 vs run2:")
    for i in range(min(3, len(queries))):
        print(
            f"  q{i}: run1(nn_idx={int(idx1_t[i,0].item())}, dist={float(dist1_t[i,0].item())}) | "
            f"run2(nn_idx={int(idx2_t[i,0].item())}, dist={float(dist2_t[i,0].item())})"
        )

    print("\n[LOAD] Done.")


# ------------------------------
# Example usage (no argparse)
# ------------------------------
if __name__ == "__main__":
    # Adjust these as you like:
    work = Path("/g/data/te53/en9803/sandpit/graph_genomics/chr22")
    tiny_fa = work / "tiny.fa"

    cfg_index = IndexConfig(window=10_000, stride=5_000, rc_index=True)

    # Larger test size and richer comparison
    N_QUERIES = 32
    TOP_K = 5

    # First run: build and save artifacts
    outdir = build_once_and_save(
        cfg_index,
        tiny_fa,
        n_queries=N_QUERIES,
        top_k=TOP_K,
        embedder="hyena",
        emb_batch=512,
        save_prefix="run1",
    )

    # Second run: load, re-embed, and compare against run1
    load_again_compare_and_print(
        cfg_index,
        tiny_fa,
        outdir,
        embedder="hyena",
        emb_batch=512,
        load_prefix="run1",
        n_queries_override=None,  # or set to an int to truncate for quick checks
    )
