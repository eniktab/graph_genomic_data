# build_index_from_aligned.py
from __future__ import annotations
from pathlib import Path
from typing import List, Literal, Tuple, Dict, Any, Optional

import glob
import json
import numpy as np
import torch

# Your ANN engine (RAPIDS cuVS CAGRA/IVF-PQ)
from src.RaftGPU import RaftGPU  # type: ignore


def _discover_shards(aligned_dir: Path) -> List[Path]:
    shard_dir = aligned_dir / "shards"
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"Missing shards directory: {shard_dir}")
    paths = sorted(Path(p) for p in glob.glob(str(shard_dir / "aligned-*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shard files found under {shard_dir}")
    return paths


def _reduce_token_tensor(
    H_t: torch.Tensor,            # [T_pad, H] raw
    mask_t: torch.Tensor,         # [T_pad] bool
    how: Literal["center", "mean_masked"] = "center"
) -> torch.Tensor:                # [H]
    T = int(mask_t.sum().item())
    if T <= 0:
        # degenerate; fall back to zeros
        return torch.zeros(H_t.shape[-1], dtype=H_t.dtype)
    if how == "center":
        idx = T // 2
        return H_t[idx].contiguous()              # [H]
    # mean over valid tokens only
    return (H_t[:T].mean(dim=0)).contiguous()     # [H]


def _vectorize_aligned_split(
    aligned_dir: Path,
    *,
    which: Literal["best", "last"] = "best",
    reducer: Literal["center", "mean_masked"] = "center",
    dtype: Literal["fp16", "fp32"] = "fp16",
    limit: Optional[int] = None
) -> Tuple[np.ndarray, list]:
    """
    Loads aligned shards and returns (X, metas):
      X: float32 [N, D]
      metas: list of metadata rows (dicts) aligned to X's rows
    """
    shards = _discover_shards(aligned_dir)
    want_key = "hidden_best" if which == "best" else "hidden_last"

    rows: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []
    dtype_t = torch.float16 if dtype == "fp16" else torch.float32

    total = 0
    for sp in shards:
        obj = torch.load(sp, map_location="cpu")
        if want_key not in obj:
            raise KeyError(f"{want_key} not in shard: {sp.name} (keys: {list(obj.keys())})")

        H_raw: torch.Tensor = obj[want_key]          # [n, T_pad, H] in f16/f32
        ids: torch.Tensor = obj["input_ids"]         # [n, T_pad] long
        mask: torch.Tensor = obj["attention_mask"]   # [n, T_pad] bool
        starts: torch.Tensor = obj["starts"]         # [n]
        ends: torch.Tensor = obj["ends"]             # [n]
        labels: torch.Tensor = obj["labels"]         # [n]  (aligned=1)

        n, T_pad, H = H_raw.shape

        # per-sample reduce to [H]
        vs = []
        for i in range(n):
            v = _reduce_token_tensor(H_raw[i], mask[i], reducer)
            vs.append(v.to(dtype=torch.float32, copy=False))   # final index in fp32
        V = torch.stack(vs, dim=0)                             # [n, H] fp32

        rows.append(V.numpy())
        for i in range(n):
            metas.append({
                "contig": None,    # optional: can read from CSV if you prefer
                "start": int(starts[i].item()),
                "end": int(ends[i].item()),
                "label": int(labels[i].item()),
                "source_shard": sp.name,
                "row_in_shard": i,
                "which": which,
                "reducer": reducer,
            })

        total += n
        if limit is not None and total >= limit:
            # Trim last block to exact limit
            over = total - limit
            if over > 0:
                rows[-1] = rows[-1][:-over]
                metas = metas[:-over]
            break

        # free promptly
        del obj, H_raw, ids, mask, starts, ends, labels, V, vs

    X = np.concatenate(rows, axis=0).astype(np.float32, copy=False)  # [N, H]
    return X, metas


def build_aligned_index(
    outdir: str | Path,
    *,
    dataset_dir: str | Path,                # parent that contains aligned/ and misaligned/
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
    limit: Optional[int] = None,            # for quick smoke tests
) -> Tuple[RaftGPU, Dict[str, Any]]:
    """
    Vectorizes aligned shards -> builds RaftGPU ANN index -> saves it to outdir.
    Returns (raft, manifest_dict).
    """
    dataset_dir = Path(dataset_dir)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) Vectorize (aligned only)
    aligned_dir = dataset_dir / "aligned"
    X, metas = _vectorize_aligned_split(
        aligned_dir, which=which, reducer=reducer, dtype="fp16", limit=limit
    )  # X: [N, H] float32

    if X.ndim != 2 or X.shape[0] == 0:
        raise RuntimeError("No vectors produced from aligned shards.")
    N, D = X.shape

    # 2) Build ANN (defaults: CAGRA NN-Descent + cosine)
    raft = RaftGPU(
        dim=D, metric=metric, build_algo=build_algo,
        graph_degree=graph_degree, intermediate_graph_degree=inter_graph_degree,
        nn_descent_niter=nn_descent_niter,
        ivf_pq_dim=ivf_pq_dim, ivf_pq_bits=ivf_pq_bits,
        ivf_n_lists=ivf_n_lists, ivf_n_probes=ivf_n_probes,
    )
    raft.add(X, metas)  # accepts NumPy; moves/normalizes on device for cosine. :contentReference[oaicite:1]{index=1}

    # 3) Save index (includes metas; embeddings embedded if supported by cuVS)
    mani = raft.save(outdir, include_dataset=True)              # robust save/load behavior. :contentReference[oaicite:2]{index=2}

    # Also drop a small human-readable manifest for provenance
    manifest = {
        "source": str(dataset_dir),
        "split": "aligned",
        "N": int(N), "D": int(D),
        "which": which, "reducer": reducer,
        "metric": metric, "build_algo": build_algo,
        "raft_manifest": mani,
    }
    (outdir / "reader_manifest.json").write_text(json.dumps(manifest, indent=2))
    return raft, manifest

if __name__ == "__main__":
    raft, manifest = build_aligned_index(
        outdir="/g/data/te53/en9803/sandpit/graph_genomics/chr22/index_center_best_cosine",
        dataset_dir="/g/data/te53/en9803/sandpit/graph_genomics/chr22/windows_raw",
        which="best",                 # use 'hidden_best' from shards
        reducer="center",             # vector = center token; use 'mean_masked' for masked mean
        metric="cosine",              # RaftGPU will L2-normalize rows for cosine internally
        build_algo="nn_descent",      # CAGRA NN-Descent
    )
    print(raft)