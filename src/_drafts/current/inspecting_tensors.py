# spaghetti_inspect_embeddings.py
# -----------------------------------------------------------------------------
# Step-by-step, print-heavy "spaghetti" script for:
#   - Loading aligned shards (hidden_best / hidden_last; raw token states)
#   - Reducing token embeddings → window vectors via multiple reducers
#   - Trying dimensionality reductions (identity / PCA / random orthonormal)
#   - Building a RaftGPU index and querying with aligned & misaligned windows
#   - Inspecting recall@k, MRR, and nearest neighbor metas
#
# Notes:
#   * No argparse; edit the CONFIG block below.
#   * HyenaBackend + RaftGPU are expected to be importable from src/.
#   * We never pool at model level; we reduce token states ourselves.
#   * Nothing is forced to "mean" — multiple reducers are available.
# -----------------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
import json, glob, random
from typing import List, Dict, Any, Literal, Optional

import numpy as np
import pandas as pd
import torch

# Environment-provided backends
from src.HyenaBackend import HyenaBackend
from src.HyenaBackend import HyenaDNAPooler  # noqa: F401
from src.RaftGPU import RaftGPU

# ==============================
# CONFIG (edit and re-run)
# ==============================
dataset_dir = Path("/g/data/te53/en9803/sandpit/graph_genomics/chr22/windows_raw")
fasta_path  = Path("/g/data/te53/en9803/sandpit/graph_genomics/chr22/chm13v2_chr22.fa.gz")
contig_name = "chr22"
out_root    = Path("/g/data/te53/en9803/sandpit/graph_genomics/chr22/spaghetti_out")

# What hidden set to use from shards and queries
which_hidden: Literal["best", "last"] = "best"    # "best" or "last"
best_layer_spec: int = -7                         # only used if which_hidden == "best"

# Try multiple reducers without forcing any single one:
# options = "center", "mean_masked", "first", "last", "max", "median"
REDUCER_MODES = ["center", "mean_masked"]        # edit list as you like

# Dimensionality transform choices to try (in order):
# ("identity", None), ("pca", 128), ("random_ortho", 128), ...
TRANSFORMS = [("identity", None), ("pca", 128)]

# PCA fit speed-up: sample at most this many rows (None = use all)
PCA_SAMPLE_ROWS: Optional[int] = 20_000

# ANN settings
metric: Literal["cosine", "l2"] = "cosine"
build_algo: Literal["nn_descent", "ivf_pq"] = "nn_descent"
k_neighbors = 10

# Query generation
n_aligned_queries = 64
n_misaligned_queries = 64
stride = 5_000
misalign_offset: Optional[int] = None  # None → take from manifest if present
rng = random.Random(13)

# ========================================
# Setup & manifest / metadata inspection
# ========================================
out_root.mkdir(parents=True, exist_ok=True)

print("\n=== MANIFEST / META ===")
mani_path = dataset_dir / "manifest.json"
if mani_path.is_file():
    mani = json.loads(mani_path.read_text())
else:
    mani = {}

print("manifest keys:", list(mani.keys()))
if mani:
    print(json.dumps(mani, indent=2))

model_name = mani.get("model_name", "hyenadna-large-1m-seqlen-hf")
model_dir  = mani.get("model_dir", "hyenadna-large-1m-seqlen-hf")
window_len = int(mani.get("window", 10_000))
if which_hidden == "best":
    try:
        if isinstance(mani.get("hidden_states_saved"), list):
            for e in mani["hidden_states_saved"]:
                if isinstance(e, str) and e.startswith("best:"):
                    best_layer_spec = int(e.split("best:")[1])
    except Exception:
        pass

print(f"[model] {model_name} @ {model_dir}")
print(f"[dataset] window={window_len}  which_hidden={which_hidden}  best_layer_spec={best_layer_spec}")

meta_csv = dataset_dir / "aligned" / "meta.csv"
if not meta_csv.is_file():
    raise FileNotFoundError(f"Missing {meta_csv}")
df_aligned = pd.read_csv(meta_csv)
aligned_starts_all = df_aligned["start"].astype(int).tolist()
print(f"[aligned] total windows: {len(aligned_starts_all):,}")

# Discover shards
print("\n=== SHARDS ===")
shard_dir = dataset_dir / "aligned" / "shards"
shard_paths = sorted(Path(p) for p in glob.glob(str(shard_dir / "aligned-*.pt")))
if not shard_paths:
    raise FileNotFoundError(f"No shards found under {shard_dir}")
for p in shard_paths[:3]:
    print("  shard:", p.name)

# Probe first shard contents
probe = torch.load(shard_paths[0], map_location="cpu")
print("first shard keys:", sorted(probe.keys()))
for key in ["hidden_best", "hidden_last", "attention_mask", "input_ids", "starts", "ends", "labels"]:
    if key in probe:
        t = probe[key]
        if torch.is_tensor(t):
            print(f"  {key}: shape={tuple(t.shape)} dtype={t.dtype}")
        else:
            print(f"  {key}: type={type(t)}")
del probe

# ==========================
# Helper: cupy -> numpy
# ==========================
# (kept tiny and inline; avoids accidental device arrays in prints)
def _ensure_numpy(x):
    try:
        import cupy as cp  # type: ignore
        if isinstance(x, cp.ndarray):
            return x.get()
    except Exception:
        pass
    return x

# ========================================================
# Loop over reducers, vectorize, then try transforms each
# ========================================================
for reducer_mode in REDUCER_MODES:
    print(f"\n\n###############################")
    print(f"### REDUCER: {reducer_mode}")
    print(f"###############################")

    # -------------------------------------------
    # Vectorize all shards → X_base [N, H], metas
    # -------------------------------------------
    which_key = "hidden_best" if which_hidden == "best" else "hidden_last"
    rows = []
    metas: List[Dict[str, Any]] = []

    for sp in shard_paths:
        obj = torch.load(sp, map_location="cpu")

        # Fallback if the desired key is missing
        use_key = which_key
        if use_key not in obj:
            alt = "hidden_last" if which_key == "hidden_best" else "hidden_best"
            if alt in obj:
                print(f"[warn] {which_key} missing in {sp.name}; falling back to {alt}")
                use_key = alt
            else:
                raise KeyError(f"{which_key} not present in {sp.name}. Keys={list(obj.keys())}")

        H_raw: torch.Tensor = obj[use_key]           # [n, T_pad, H]
        mask: torch.Tensor = obj["attention_mask"]   # [n, T_pad]
        starts: torch.Tensor = obj["starts"]         # [n]
        ends: torch.Tensor = obj["ends"]             # [n]
        labels: torch.Tensor = obj["labels"]         # [n]

        n, T_pad, H = H_raw.shape
        # Reduce [T_pad, H] → [H] per row, without a separate function
        vecs = []
        for i in range(n):
            T = int(mask[i].sum().item())
            if T <= 0:
                v = torch.zeros(H, dtype=torch.float32)
            else:
                X = H_raw[i, :T, :].to(torch.float32)
                if reducer_mode == "center":
                    v = X[T // 2]
                elif reducer_mode == "mean_masked":
                    v = X.mean(dim=0)
                elif reducer_mode == "first":
                    v = X[0]
                elif reducer_mode == "last":
                    v = X[-1]
                elif reducer_mode == "max":
                    v = X.max(dim=0).values
                elif reducer_mode == "median":
                    v = X.median(dim=0).values
                else:
                    raise ValueError(f"unknown reducer_mode={reducer_mode}")
            vecs.append(v)
        V = torch.stack(vecs, dim=0).to(torch.float32).contiguous()  # [n, H]
        rows.append(V.numpy())

        for i in range(n):
            metas.append({
                "start": int(starts[i]),
                "end": int(ends[i]),
                "label": int(labels[i]),
                "source_shard": sp.name,
                "row_in_shard": i,
                "which": which_hidden,
                "reducer": reducer_mode,
            })

        del obj, H_raw, mask, starts, ends, labels, V, vecs

    X_base = np.concatenate(rows, axis=0).astype(np.float32, copy=False) if rows else np.zeros((0, H), np.float32)
    print(f"[vectorize] X_base: shape={X_base.shape} dtype={X_base.dtype}")
    if X_base.size:
        means = X_base.mean(axis=0); stds = X_base.std(axis=0)
        norms = np.linalg.norm(X_base, axis=1)
        print(f"[vectorize] feature mean(mean)={means.mean():.6f}  mean(std)={stds.mean():.6f}")
        print(f"[vectorize] row ||x||₂ min/mean/max = {norms.min():.4f}/{norms.mean():.4f}/{norms.max():.4f}")
        print(f"[vectorize] sample[0][:5] = {np.array2string(X_base[0,:5], precision=4)}")

    # -------------------------------------------------------
    # Try each transform: identity / PCA(d) / random_ortho(d)
    # -------------------------------------------------------
    for tname, param in TRANSFORMS:
        if tname == "identity":
            X_work = X_base
            d_eff = X_work.shape[1]
            transform_dir = out_root / f"transform_identity_d{d_eff}"
            transform_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[transform] identity → d={d_eff}")
        elif tname == "pca":
            d_eff = int(param)
            transform_dir = out_root / f"transform_pca_d{d_eff}"
            transform_dir.mkdir(parents=True, exist_ok=True)
            # Fit PCA via SVD (subsample if configured)
            if PCA_SAMPLE_ROWS is None or X_base.shape[0] <= PCA_SAMPLE_ROWS:
                fit_rows = X_base
            else:
                idx = np.random.default_rng(0).choice(X_base.shape[0], size=PCA_SAMPLE_ROWS, replace=False)
                fit_rows = X_base[idx]
            mu = fit_rows.mean(axis=0).astype(np.float32, copy=False)
            Xc = (fit_rows - mu).astype(np.float32, copy=False)
            U, S, VT = np.linalg.svd(Xc, full_matrices=False)
            comps = VT[:d_eff, :].astype(np.float32, copy=False)  # [d, D]
            # Save transform
            np.savez(transform_dir / "pca.npz", mean=mu, components=comps)
            # Apply
            Xc_full = (X_base - mu).astype(np.float32, copy=False)
            X_work = (comps @ Xc_full.T).T.astype(np.float32, copy=False)
            exp_var = (S[:d_eff]**2).sum() / (S**2).sum()
            print(f"\n[transform] PCA(d={d_eff})  approx explained variance (fit subset): {exp_var:.4f}")
        elif tname == "random_ortho":
            d_eff = int(param)
            transform_dir = out_root / f"transform_random_ortho_d{d_eff}"
            transform_dir.mkdir(parents=True, exist_ok=True)
            # Random Gaussian → QR for orthonormal columns
            D = X_base.shape[1]
            G = np.random.default_rng(0).standard_normal(size=(D, d_eff)).astype(np.float32, copy=False)
            Q, _ = np.linalg.qr(G, mode="reduced")  # [D, d], columns orthonormal
            mu = np.zeros((D,), np.float32)
            np.savez(transform_dir / "random_ortho.npz", W=Q, mu=mu)
            X_work = ((X_base - mu) @ Q).astype(np.float32, copy=False)
            print(f"\n[transform] random_ortho(d={d_eff})")
        else:
            raise ValueError(f"Unknown transform: {tname}")

        if X_work.size:
            means2 = X_work.mean(axis=0); stds2 = X_work.std(axis=0)
            norms2 = np.linalg.norm(X_work, axis=1)
            print(f"[transform] X_work: shape={X_work.shape} dtype={X_work.dtype}")
            print(f"[transform] feature mean(mean)={means2.mean():.6f}  mean(std)={stds2.mean():.6f}")
            print(f"[transform] row ||x||₂ min/mean/max = {norms2.min():.4f}/{norms2.mean():.4f}/{norms2.max():.4f}")

        # -----------------------
        # Build / save the index
        # -----------------------
        index_dir = out_root / f"index_{which_hidden}_{reducer_mode}_{tname}_d{X_work.shape[1]}_{metric}_{build_algo}"
        raft = RaftGPU(
            dim=int(X_work.shape[1]), metric=metric, build_algo=build_algo,
            graph_degree=64, intermediate_graph_degree=128, nn_descent_niter=20,
            ivf_pq_dim=64, ivf_pq_bits=8, ivf_n_lists=None, ivf_n_probes=None,
        )
        raft.add(X_work, metas)
        raft_manifest = raft.save(index_dir, include_dataset=True)

        algo_name = getattr(raft, "build_algo", getattr(raft, "_build_algo", "unknown"))
        dim_name  = getattr(raft, "dim", X_work.shape[1])
        print(f"\n[index] saved → {index_dir}")
        print(f"[index] N={len(metas):,}  D={dim_name}  algo={algo_name}  metric={metric}")
        print(f"[index] manifest keys: {list(raft_manifest.keys())}")

        # =========================================
        # Build queries (aligned + misaligned)
        # =========================================
        print("\n=== QUERIES / SEARCH ===")
        # aligned sample
        if n_aligned_queries >= len(aligned_starts_all):
            aligned_starts = aligned_starts_all
        else:
            aligned_starts = rng.sample(aligned_starts_all, n_aligned_queries)

        # misaligned sample: s ≡ offset (mod stride)
        off = int(misalign_offset if misalign_offset is not None else mani.get("misalign_offset", stride // 2))
        try:
            import pysam
            with pysam.FastaFile(str(fasta_path)) as fa:
                L = fa.get_reference_length(contig_name)
            mis_positions = list(range(off, max(0, L - window_len) + 1, stride))
        except Exception as e:
            raise RuntimeError(f"Failed to open FASTA or contig not found: {e}")
        if n_misaligned_queries >= len(mis_positions):
            mis_starts = mis_positions
        else:
            mis_starts = rng.sample(mis_positions, n_misaligned_queries)

        print(f"[queries] aligned={len(aligned_starts)}  misaligned_offset={off}  misaligned={len(mis_starts)}")

        # fetch sequences
        aligned_seqs, mis_seqs = [], []
        with pysam.FastaFile(str(fasta_path)) as fa:
            for s in aligned_starts:
                aligned_seqs.append(fa.fetch(contig_name, s, s + window_len).upper())
            for s in mis_starts:
                mis_seqs.append(fa.fetch(contig_name, s, s + window_len).upper())
        if aligned_seqs:
            print("[queries] example aligned seq length:", len(aligned_seqs[0]))

        # embed queries with Hyena (no pooling), then reduce with SAME reducer_mode
        backend = HyenaBackend(model_name=model_name, model_dir=model_dir,
                               pooling="none", normalize=False, offline=True, prefer_cuda=True)
        tok = backend.tokenizer
        dev = backend.device

        def _embed_reduce(seqs: List[str]) -> np.ndarray:
            batch = 128
            rows_q = []
            for i in range(0, len(seqs), batch):
                chunk = seqs[i:i+batch]
                enc = tok(chunk, padding="longest", truncation=True,
                          max_length=getattr(backend, "max_length", None), return_tensors="pt")
                ids = enc["input_ids"].to(dev, non_blocking=True)
                mask = (ids != int(tok.pad_token_id)).to(dev)
                out = backend.model(ids)
                hs_list = out.hidden_states if hasattr(out, "hidden_states") else [out.last_hidden_state]
                if which_hidden == "last":
                    Lm = hs_list[-1]
                else:
                    idx = best_layer_spec if best_layer_spec >= 0 else len(hs_list) + best_layer_spec
                    idx = max(0, min(len(hs_list) - 1, idx))
                    Lm = hs_list[idx]
                B = Lm.shape[0]
                vecs_q = []
                for b in range(B):
                    T = int(mask[b].sum().item())
                    if T <= 0:
                        v = torch.zeros(Lm.shape[-1], dtype=torch.float32, device=Lm.device)
                    else:
                        X = Lm[b, :T, :].to(torch.float32)
                        if reducer_mode == "center":
                            v = X[T // 2]
                        elif reducer_mode == "mean_masked":
                            v = X.mean(dim=0)
                        elif reducer_mode == "first":
                            v = X[0]
                        elif reducer_mode == "last":
                            v = X[-1]
                        elif reducer_mode == "max":
                            v = X.max(dim=0).values
                        elif reducer_mode == "median":
                            v = X.median(dim=0).values
                        else:
                            raise ValueError(f"unknown reducer_mode={reducer_mode}")
                    vecs_q.append(v)
                Vq = torch.stack(vecs_q, dim=0).to(torch.float32).contiguous()
                rows_q.append(Vq.to("cpu", non_blocking=True))
                del out, hs_list, Lm, ids, mask, Vq, vecs_q
            if rows_q:
                return torch.cat(rows_q, dim=0).numpy().astype(np.float32, copy=False)
            return np.zeros((0, X_base.shape[1]), np.float32)

        Xq_aligned_base = _embed_reduce(aligned_seqs)
        Xq_misaln_base  = _embed_reduce(mis_seqs)

        # apply saved transform (match the index)
        if tname == "identity":
            Xq_aligned = Xq_aligned_base
            Xq_misaln  = Xq_misaln_base
        elif tname == "pca":
            z = np.load(transform_dir / "pca.npz")
            mean = z["mean"].astype(np.float32, copy=False)
            comps = z["components"].astype(np.float32, copy=False)  # [d, D]
            Xq_aligned = ((Xq_aligned_base - mean) @ comps.T).astype(np.float32, copy=False)
            Xq_misaln  = ((Xq_misaln_base  - mean) @ comps.T).astype(np.float32, copy=False)
        elif tname == "random_ortho":
            z = np.load(transform_dir / "random_ortho.npz")
            W = z["W"].astype(np.float32, copy=False)   # [D, d]
            mu = z["mu"].astype(np.float32, copy=False)
            Xq_aligned = ((Xq_aligned_base - mu) @ W).astype(np.float32, copy=False)
            Xq_misaln  = ((Xq_misaln_base  - mu) @ W).astype(np.float32, copy=False)
        else:
            raise ValueError

        print(f"[queries] Xq_aligned: {Xq_aligned.shape}   Xq_misaln: {Xq_misaln.shape}")

        # search
        D_a, I_a = raft.search(Xq_aligned, k=k_neighbors)
        D_m, I_m = raft.search(Xq_misaln,  k=k_neighbors)
        D_a, I_a = _ensure_numpy(D_a), _ensure_numpy(I_a)
        D_m, I_m = _ensure_numpy(D_m), _ensure_numpy(I_m)
        print("[search] got distances/indices for aligned+misaligned")

        # build start→row map for GT
        idx2start = np.array([m["start"] for m in metas], dtype=np.int64)
        start2rows: Dict[int, List[int]] = {}
        for i, s in enumerate(idx2start.tolist()):
            start2rows.setdefault(int(s), []).append(i)

        # aligned ground-truth rows
        gt_rows_aligned = []
        for s in aligned_starts:
            rows_here = start2rows.get(int(s), [])
            if rows_here:
                gt_rows_aligned.append(rows_here[0])
        gt_rows_aligned = np.array(gt_rows_aligned, dtype=np.int64)
        if I_a.shape[0] > len(gt_rows_aligned):
            I_a = I_a[:len(gt_rows_aligned)]

        # misaligned expected aligned window row = floor(s/stride)*stride
        exp_starts = np.array([(s // stride) * stride for s in mis_starts], dtype=np.int64)
        gt_rows_mis, mask_valid = [], []
        for s in exp_starts:
            rows_here = start2rows.get(int(s), [])
            if rows_here:
                gt_rows_mis.append(rows_here[0])
                mask_valid.append(True)
            else:
                mask_valid.append(False)
        gt_rows_mis = np.array(gt_rows_mis, dtype=np.int64)
        mask_valid = np.array(mask_valid, dtype=bool)
        I_m_valid = I_m[mask_valid]

        # metrics (inline loops)
        def _recall_at(I, gt, k):
            if len(gt) == 0:
                return float('nan')
            hits = 0
            for i in range(len(gt)):
                if int(gt[i]) in I[i, :k]:
                    hits += 1
            return hits / float(len(gt))

        def _mrr(I, gt):
            if len(gt) == 0:
                return float('nan')
            rr = 0.0
            for i in range(len(gt)):
                row = I[i]
                pos = np.where(row == gt[i])[0]
                rr += 1.0 / (1 + int(pos[0])) if pos.size else 0.0
            return rr / float(len(gt))

        print("\n[metrics] Identity (aligned):")
        print("  R@1 :", f"{_recall_at(I_a, gt_rows_aligned, 1):.3%}")
        print("  R@5 :", f"{_recall_at(I_a, gt_rows_aligned, 5):.3%}")
        print("  R@10:", f"{_recall_at(I_a, gt_rows_aligned, 10):.3%}")
        print("  MRR :", f"{_mrr(I_a, gt_rows_aligned):.4f}")

        print("\n[metrics] Misaligned:")
        print("  Valid:", f"{len(I_m_valid)}/{len(I_m)}")
        if len(I_m_valid):
            print("  R@1 :", f"{_recall_at(I_m_valid, gt_rows_mis, 1):.3%}")
            print("  R@5 :", f"{_recall_at(I_m_valid, gt_rows_mis, 5):.3%}")
            print("  R@10:", f"{_recall_at(I_m_valid, gt_rows_mis, 10):.3%}")
            print("  MRR :", f"{_mrr(I_m_valid, gt_rows_mis):.4f}")

        # show a few neighbors
        print("\n[neighbors] First 3 aligned queries → neighbor starts:")
        try:
            decoded = raft.decode(I_a)  # preferred if available
        except Exception:
            decoded = [[metas[int(j)] for j in I_a[i]] for i in range(min(3, I_a.shape[0]))]
        for qi in range(min(3, I_a.shape[0])):
            nns = [d.get("start") for d in decoded[qi]]
            print(f"  q{qi}: gt_start={aligned_starts[qi]} → nn_starts={nns}")

print("\nAll reducer/transform combinations completed.")
