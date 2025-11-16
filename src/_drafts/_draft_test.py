# tests/test_save_load_equivalence.py
import os
from pathlib import Path

import numpy as np
import cupy as cp
import pytest

# if RaftGPU lives in the same file, adjust this import accordingly:
# from your_module import RaftGPU
from src.RaftGPU import RaftGPU


def _pick_large_N(D: int, *, min_n: int = 200_000, max_n: int = 2_000_000) -> int:
    """Pick a large N that fits current free VRAM (conservative cushion)."""
    free, _ = cp.cuda.runtime.memGetInfo()
    bytes_per_vec = D * 4
    budget = int(free * 0.35)                # ~35% of free VRAM
    est = max(min_n, budget // (12 * bytes_per_vec))  # ~12x cushion for graph/temps
    return int(max(min_n, min(est, max_n)))


def _build_index(D=96, seed=123):
    cp.random.seed(seed); np.random.seed(seed)
    N = int(os.environ.get("RAFTGPU_TEST_N", 0)) or _pick_large_N(D)
    X = cp.random.standard_normal((N, D), dtype=cp.float32)
    # light structure to reduce accidental ties:
    X += (cp.arange(N, dtype=cp.float32)[:, None] % 7) * 1e-3

    metas = [(int(i), f"row-{int(i)}") for i in range(N)]

    ann = RaftGPU(D, metric="cosine", build_algo="nn_descent")
    ann.add(X, metas)

    # queries: 100 exact rows + 100 perturbed rows
    idx_sample = cp.asarray(cp.random.choice(N, size=100, replace=False))
    Q_eq = X.take(idx_sample, axis=0).copy()
    Q_pt = Q_eq + cp.float32(3e-4)
    Q = cp.concatenate([Q_eq, Q_pt], axis=0)
    return ann, X, metas, Q, idx_sample


def _metrics(Ia_cp, Da_cp, Ib_cp, Db_cp):
    """Overlap/score drift between two result sets (A=baseline, B=post-load)."""
    Ia = cp.asnumpy(Ia_cp); Da = cp.asnumpy(Da_cp)
    Ib = cp.asnumpy(Ib_cp); Db = cp.asnumpy(Db_cp)
    nq, k = Ia.shape

    recalls, jaccs, rank_disp = [], [], []
    diffs = []

    for q in range(nq):
        A = Ia[q].tolist(); B = Ib[q].tolist()
        setA, setB = set(A), set(B)
        inter = setA & setB
        union = setA | setB

        recalls.append(len(inter) / k)
        jaccs.append(len(inter) / len(union) if union else 1.0)

        # mean absolute rank displacement on intersection
        if inter:
            posA = {id_: i for i, id_ in enumerate(A)}
            posB = {id_: i for i, id_ in enumerate(B)}
            disp = [abs(posA[id_] - posB[id_]) for id_ in inter]
            rank_disp.append(np.mean(disp))
            for id_ in inter:
                diffs.append(float(Da[q, posA[id_]] - Db[q, posB[id_]]))
        else:
            rank_disp.append(float(k))  # worst case

    recalls = np.asarray(recalls, dtype=np.float64)
    jaccs   = np.asarray(jaccs,   dtype=np.float64)
    rdisp   = np.asarray(rank_disp, dtype=np.float64)
    diffs   = np.asarray(diffs,   dtype=np.float64) if diffs else np.array([0.0])

    mae = float(np.mean(np.abs(diffs)))
    rmse = float(np.sqrt(np.mean(diffs**2)))

    return dict(
        recall_mean=float(recalls.mean()),
        recall_p05=float(np.percentile(recalls, 5.0)),
        jaccard_mean=float(jaccs.mean()),
        rank_disp_mean=float(rdisp.mean()),
        rank_disp_p95=float(np.percentile(rdisp, 95.0)),
        mae_mean=mae,
        rmse_mean=rmse,
        n_pairs=int(diffs.size),
    )


def _top1_self_hit_rate(I_cp, idx_sample):
    """For the first 100 'exact row' queries, how often is top-1 the same row?"""
    I = cp.asnumpy(I_cp)
    target = cp.asnumpy(idx_sample)
    return float((I[:100, 0] == target).mean())


def _no_worse_than(post, base, *, abs_margin: float, rel_margin: float) -> bool:
    """Allow post to be lower than base by at most max(abs_margin, rel_margin * base)."""
    allowed_drop = max(abs_margin, rel_margin * base)
    return post + allowed_drop >= base


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_save_load_similarity(tmp_path: Path):
    K = int(os.environ.get("RAFTGPU_TEST_K", 20))

    # ---------- Baseline: freshly built index ----------
    ann, X, metas, Q, idx_sample = _build_index()
    D0, I0 = ann.search(Q, k=K)
    base_selfhit = _top1_self_hit_rate(I0, idx_sample)

    # ---------- Save + Load ----------
    # Prefer embedding the dataset; if the environment falls back to .npy, we still compare.
    ann.save(tmp_path, include_dataset=True)
    ann2 = RaftGPU.load(tmp_path)
    D1, I1 = ann2.search(Q, k=K)

    # ---------- Similarity (post-load vs baseline) ----------
    sim = _metrics(I0, D0, I1, D1)
    post_selfhit = _top1_self_hit_rate(I1, idx_sample)

    # ---- PASS CRITERIA (relative to the initial build) ----
    # 1) Top-1 exact-row self-hit must not degrade meaningfully vs initial build.
    assert _no_worse_than(post_selfhit, base_selfhit, abs_margin=0.01, rel_margin=0.10), (
        f"Top-1 self-hit dropped too much after load: "
        f"baseline={base_selfhit:.4f}, post={post_selfhit:.4f}"
    )

    # 2) Distance drift on overlapping ids stays tiny (absolute scale)
    #    (Cosine-as-inner-product distances are well-behaved at these tolerances.)
    assert sim["mae_mean"] <= 1e-4, f"MAE drift too high: {sim['mae_mean']:.2e}"
    assert sim["rmse_mean"] <= 3e-4, f"RMSE drift too high: {sim['rmse_mean']:.2e}"

    # ---------- Informative prints (don’t gate the test) ----------
    print("\n[Similarity vs initial build]")
    for k, v in sim.items():
        if k == "n_pairs":
            continue
        print(f"  {k:>16}: {v:.6f}")
    print(f"  {'top1_self_hit':>16}: baseline={base_selfhit:.6f}  post-load={post_selfhit:.6f}")

    # Optional: ensure we stayed on device
    V = ann2.dataset_cupy()
    assert isinstance(V, cp.ndarray) and V.dtype == cp.float32 and V.flags.c_contiguous