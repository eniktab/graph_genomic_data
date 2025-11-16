# test_raftgpu.py
"""
CUDA-safe unit tests for RaftGPU.

Design goals
------------
- Never compare host and device arrays directly.
- Do device math with CuPy, and only convert tiny scalars/JSON to host.
- Treat search results as metadata-first (current RaftGPU.search API).
- Keep coverage: init, add/search (np/cp/torch), k-clamp, cosine vs L2,
  save/load (with/without dataset), manifest, thread-safety, alignment stress,
  and integration consistency.

Notes:
- Requires GPU with CuPy + cuVS installed.
- Some features (e.g., dataset_cupy()) are optional; tests skip gracefully.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
import math
import os

import numpy as np
import pytest

# Hard skip if GPU deps aren’t present
pytest.importorskip("cupy")
pytest.importorskip("cuvs")

import cupy as cp
import torch

# Import RaftGPU (repo layout first; fallback to /mnt/data)
try:
    from src.RaftGPU import RaftGPU
except Exception:  # pragma: no cover
    import sys
    sys.path.append("/mnt/data")
    from RaftGPU import RaftGPU  # type: ignore

# -------------------------
# Helpers (CUDA-safe)
# -------------------------
def _pure_metas_from_decoded(raft: RaftGPU, decoded_rows):
    """
    Given output of raft.decode(I, D) (a list of rows with per-item tuples),
    strip off any decode-added fields (e.g., row_idx, distance), returning ONLY
    the original meta tuples.
    """
    if not raft.metas:
        return [[] for _ in decoded_rows]
    meta_arity = len(raft.metas[0])
    out = []
    for row in decoded_rows:
        out.append([tuple(item[:meta_arity]) for item in row])
    return out


def _rows_from_search(raft: RaftGPU, search_out):
    """
    Normalize RaftGPU.search() output into a list-of-rows of *metas*.

    API: search_out is a tuple (D_cp, I_cp).
    We must convert to NumPy before passing to raft.decode(), which expects host arrays.
    """
    assert isinstance(search_out, tuple) and len(search_out) in (2, 3), (
        "Expected RaftGPU.search() to return (D_cp, I_cp[, aux])"
    )
    D_cp, I_cp = search_out[0], search_out[1]
    # Convert device -> host for decode()
    I_np = np.asarray(cp.asnumpy(I_cp))
    D_np = np.asarray(cp.asnumpy(D_cp), dtype=np.float32)
    decoded = raft.decode(I_np, D_np)
    return _pure_metas_from_decoded(raft, decoded)


def _first_row_metas(raft: RaftGPU, q, k: int):
    """Run search(k) and return the first row of metas (host Python objects)."""
    D_cp, I_cp = raft.search(q, k=k)
    I_np = np.asarray(cp.asnumpy(I_cp))
    D_np = np.asarray(cp.asnumpy(D_cp), dtype=np.float32)
    decoded = raft.decode(I_np, D_np)
    rows = _pure_metas_from_decoded(raft, decoded)
    assert isinstance(rows, list) and len(rows) >= 1
    return rows[0]


def _first_row_metas_dists(raft: RaftGPU, q, k: int):
    """
    Run search(k) and return (metas, dists_np) for first row.
    Distances are explicitly copied to host as a NumPy array for reliable comparison.
    """
    D_cp, I_cp = raft.search(q, k=k)
    I_np = np.asarray(cp.asnumpy(I_cp))
    D_np = np.asarray(cp.asnumpy(D_cp), dtype=np.float32)
    decoded = raft.decode(I_np, D_np)
    rows = _pure_metas_from_decoded(raft, decoded)
    assert isinstance(rows, list) and len(rows) >= 1
    metas = rows[0]
    dists_np = D_np[0].astype(np.float32, copy=False)
    return metas, dists_np


def _get_dataset_cp_or_skip(raft: RaftGPU):
    """
    Return the dataset as a CuPy array without host mixing, or skip if unavailable.
    """
    if not hasattr(raft, "dataset_cupy"):
        pytest.skip("RaftGPU.dataset_cupy() unavailable")
    V = raft.dataset_cupy()
    if V is None:
        pytest.skip("Dataset not attached to RaftGPU")
    assert isinstance(V, cp.ndarray)
    return V


def _make_small_dataset(n_vecs=500, dim=64, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_vecs, dim), dtype=np.float32)
    metas = [(i,) for i in range(n_vecs)]
    return X, metas


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _topk_row_idxs(raft: RaftGPU, Q, k: int):
    """
    Run search() and return neighbor indices as Python lists.
    - single query -> [i0, i1, ...]
    - multi  query -> [[...], [...], ...]
    """
    D_cp, I_cp = raft.search(Q, k=k)
    I_np = np.asarray(cp.asnumpy(I_cp), dtype=np.int64)
    if I_np.ndim == 1 or I_np.shape[0] == 1:
        return I_np.reshape(1, -1)[0].tolist()
    return [row.tolist() for row in I_np]


# A shared dataset fixture for the Indexing tests block.
@pytest.fixture
def sample_data():
    """Common dataset for indexing tests: returns (X, metas)."""
    return _make_small_dataset(n_vecs=500, dim=64, seed=7)


# ----------------------------
# Basics and construction
# ----------------------------
class TestRaftGPUBasics:
    def test_init_nn_descent(self):
        raft = RaftGPU(dim=128, metric="cosine", build_algo="nn_descent")
        assert raft.dim == 128
        assert raft.metric.lower() == "cosine"
        # Cosine emulated via inner_product + L2 normalization
        assert raft._metric_name == "inner_product"
        assert raft._build_algo == "nn_descent"
        assert len(raft) == 0

    def test_init_ivf_pq(self):
        raft = RaftGPU(
            dim=256,
            metric="l2",
            build_algo="ivf_pq",
            ivf_n_lists=256,
            ivf_n_probes=32,
            ivf_pq_dim=64,
            ivf_pq_bits=8,
            refinement_rate=2.0,
        )
        assert raft.dim == 256
        assert raft._metric_name == "sqeuclidean"
        assert raft._build_algo == "ivf_pq"
        assert len(raft) == 0

    def test_metric_mapping(self):
        for m in ["cosine", "cos", "ip", "inner_product"]:
            assert RaftGPU(dim=64, metric=m)._metric_name == "inner_product"
        for m in ["l2", "euclidean", "sqeuclidean"]:
            assert RaftGPU(dim=64, metric=m)._metric_name == "sqeuclidean"

    def test_add_numpy_and_len(self):
        X, metas = _make_small_dataset(100, 64)
        raft = RaftGPU(dim=64, metric="cosine")
        raft.add(X, metas)
        assert len(raft) == 100

    def test_add_cupy(self):
        X, metas = _make_small_dataset(100, 64)
        raft = RaftGPU(dim=64, metric="cosine")
        raft.add(cp.asarray(X), metas)
        # simple retrieval check
        row = _first_row_metas(raft, X[0:1], k=5)
        assert row[0] == metas[0]

    @pytest.mark.skipif(torch is None, reason="PyTorch not installed")
    def test_add_torch(self):
        X, metas = _make_small_dataset(200, 32)
        raft = RaftGPU(dim=32, metric="cosine")
        raft.add(torch.from_numpy(X), metas)
        row = _first_row_metas(raft, X[5:6], k=5)
        assert row[0] == metas[5]

    def test_k_clamp(self):
        X, metas = _make_small_dataset(20, 16)
        raft = RaftGPU(dim=16, metric="cosine")
        raft.add(X, metas)
        # k > N -> clamps at N
        metas_row = _first_row_metas(raft, X[0:1], k=999)
        assert len(metas_row) == len(X)
        # k == 0 -> at least 1
        top = _first_row_metas(raft, X[0:1], k=0)
        assert len(top) == 1

    def test_batch_query(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 64), dtype=np.float32)
        metas = [(i,) for i in range(200)]
        raft = RaftGPU(dim=64, metric="cosine")
        raft.add(X, metas)

        Q = X[:10]
        rows = _rows_from_search(raft, raft.search(Q, k=5))
        assert len(rows) == 10
        for i in range(10):
            assert rows[i][0] == metas[i]



# -------------------------
# Metric behavior (device-only math)
# -------------------------
class TestRaftGPUMetrics:
    def test_cosine_normalization(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 64), dtype=np.float32)
        metas = [(i,) for i in range(100)]
        raft = RaftGPU(dim=64, metric="cosine")
        raft.add(X, metas)

        V = _get_dataset_cp_or_skip(raft)
        norms = cp.linalg.norm(V, axis=1)
        assert cp.allclose(norms, 1.0, rtol=1e-5)

    def test_l2_no_normalization(self):
        rng = np.random.default_rng(42)
        X = (rng.standard_normal((100, 64)).astype(np.float32) * 2.5)
        metas = [(i,) for i in range(100)]
        raft = RaftGPU(dim=64, metric="l2")
        raft.add(X, metas)

        V = _get_dataset_cp_or_skip(raft)
        norms = cp.linalg.norm(V, axis=1)
        # At least one row should differ substantially from unit norm
        assert bool(cp.any(cp.abs(norms - 1.0) > 1e-3))

    def test_cosine_semantic_correctness(self):
        rng = np.random.default_rng(42)
        d = 64
        base = rng.standard_normal((1, d), dtype=np.float32)
        similar = base + 0.1 * rng.standard_normal((1, d), dtype=np.float32)
        different = 5.0 * rng.standard_normal((1, d), dtype=np.float32)
        X = np.vstack([base, similar, different, rng.standard_normal((97, d), dtype=np.float32)])
        metas = [(i,) for i in range(100)]
        raft = RaftGPU(dim=d, metric="cosine")
        raft.add(X, metas)

        row = _first_row_metas(raft, base, k=3)
        # Expect base and similar in the top-2 (order may vary)
        assert metas[0] in row[:2] and metas[1] in row[:2]

# -------------------------
# Save / load (manifest + consistency)
# -------------------------
class TestRaftGPUSaveLoad:
    @pytest.fixture
    def indexed_raft(self):
        X, metas = _make_small_dataset(500, 64)
        raft = RaftGPU(dim=64, metric="cosine")
        raft.add(X, metas)
        return raft, X, metas

    def test_save_and_load_with_dataset(self, temp_dir, indexed_raft):
        raft, X, metas = indexed_raft
        manifest = raft.save(temp_dir, include_dataset=True)
        assert isinstance(manifest, dict)
        assert "schema" in manifest
        assert "files" in manifest and "index" in manifest["files"]
        assert "metric" in manifest and manifest["metric"] in ("cosine", "l2", "ip", "inner_product")
        raft2 = RaftGPU.load(temp_dir, metric="cosine")
        # Quick retrieval parity
        r1 = _first_row_metas(raft, X[42:43], k=5)
        r2 = _first_row_metas(raft2, X[42:43], k=5)
        assert r1 == r2

    def test_save_and_load_without_dataset_with_fallback(self, temp_dir):
        # fallback path: save w/o dataset but provide embeddings as fallback
        X, metas = _make_small_dataset(300, 48)
        raft = RaftGPU(dim=48, metric="cosine")
        raft.add(X, metas)
        raft.save(temp_dir, include_dataset=False, fallback_embeddings=X)
        raft2 = RaftGPU.load(temp_dir, metric="cosine")
        r1 = _first_row_metas(raft, X[123:124], k=5)
        r2 = _first_row_metas(raft2, X[123:124], k=5)
        assert r1 == r2

    def test_manifest_fields(self, temp_dir, indexed_raft):
        raft, X, metas = indexed_raft
        m = raft.save(temp_dir, include_dataset=True)
        # required surface
        for key in ["schema", "created_by", "metric", "dim", "include_dataset", "build_algo", "params", "files"]:
            assert key in m
        # params sanity
        assert "graph_degree" in m["params"]
        # files surface
        assert "index" in m["files"]

        # JSON round trip
        path = temp_dir / "manifest.check.json"
        path.write_text(json.dumps(m, indent=2))
        m2 = json.loads(path.read_text())
        assert m2["files"]["index"] == m["files"]["index"]

    def test_metric_mismatch_on_load(self, temp_dir, indexed_raft):
        raft, X, metas = indexed_raft
        raft.save(temp_dir, include_dataset=True)
        with pytest.raises(RuntimeError, match="Metric mismatch"):
            RaftGPU.load(temp_dir, metric="l2")

    def test_fallback_normalization_detection(self, temp_dir):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 64), dtype=np.float32) * 2.0
        metas = [(i,) for i in range(100)]

        raft1 = RaftGPU(dim=64, metric="cosine")
        raft1.add(X, metas)
        m1 = raft1.save(temp_dir / "unnorm", include_dataset=False, fallback_embeddings=X)
        assert m1.get("fallback_normalized", False) is False

        Xn = X / np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
        raft2 = RaftGPU(dim=64, metric="cosine")
        raft2.add(Xn, metas)
        m2 = raft2.save(temp_dir / "norm", include_dataset=False, fallback_embeddings=Xn)
        assert m2.get("fallback_normalized", False) is True


# -------------------------
# Thread-safety of search params
# -------------------------
class TestRaftGPUTuningThreadSafety:
    def test_search_params_immutability(self):
        X, metas = _make_small_dataset(300, 64)
        raft = RaftGPU(dim=64, metric="cosine")
        raft.add(X, metas)

        original_itopk = raft.search_params.itopk_size
        q = X[0:1]

        # Larger k should allocate a fresh params object; original remains
        _ = _first_row_metas(raft, q, k=100)
        assert raft.search_params.itopk_size == original_itopk

        # Smaller k shouldn't mutate original either
        _ = _first_row_metas(raft, q, k=10)
        assert raft.search_params.itopk_size == original_itopk


# -------------------------
# Dataset attachment
# -------------------------
class TestRaftGPUDatasetAttachment:
    def test_dataset_cupy_with_dtype_and_layout(self):
        X, metas = _make_small_dataset(100, 64)
        raft = RaftGPU(dim=64, metric="cosine")
        raft.add(X, metas)

        V = _get_dataset_cp_or_skip(raft)
        # Ensure it's a CuPy fp32 C-contiguous matrix
        assert V.dtype == cp.float32 and V.flags.c_contiguous
        # retrieval still works
        row = _first_row_metas(raft, X[0:1], k=5)
        assert row[0] == metas[0]

    def test_fp64_to_fp32_conversion(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 64)).astype(np.float64)
        metas = [(i,) for i in range(100)]
        raft = RaftGPU(dim=64, metric="cosine")
        raft.add(X, metas)
        row = _first_row_metas(raft, X[0:1], k=5)
        assert row[0] == metas[0]


# -------------------------
# Build algorithms
# -------------------------
class TestRaftGPUAlgorithms:
    def test_nn_descent_recall_smoke(self):
        X, metas = _make_small_dataset(600, 64)
        raft = RaftGPU(dim=64, metric="cosine", build_algo="nn_descent", graph_degree=32, nn_descent_niter=10)
        raft.add(X, metas)

        # self-match sanity
        for i in [0, 123, 599]:
            row = _first_row_metas(raft, X[i : i + 1], k=5)
            assert row[0] == metas[i]

    def test_ivf_pq_build_and_query(self):
        X, metas = _make_small_dataset(800, 64)
        raft = RaftGPU(
            dim=64,
            metric="l2",
            build_algo="ivf_pq",
            ivf_n_lists=256,
            ivf_n_probes=24,
            ivf_pq_dim=64,
            ivf_pq_bits=8,
            refinement_rate=2.0,
        )
        raft.add(X, metas)
        # simple retrieval check (L2)
        row = _first_row_metas(raft, X[42:43], k=10)
        assert row[0] == metas[42]

    def test_ivf_pq_save_load_consistency(self, temp_dir):
        rng = np.random.default_rng(0)
        n, d = 1000, 64
        X = rng.standard_normal((n, d), dtype=np.float32)
        metas = [(i,) for i in range(n)]

        raft1 = RaftGPU(dim=d, metric="l2", build_algo="ivf_pq", ivf_n_lists=256, ivf_n_probes=24)
        raft1.add(X, metas)

        q = X[42:43]
        r1 = _first_row_metas(raft1, q, k=10)
        assert r1[0] == metas[42]

        raft1.save(temp_dir, include_dataset=False, fallback_embeddings=X)
        raft2 = RaftGPU.load(temp_dir, metric="l2")
        r2 = _first_row_metas(raft2, q, k=10)
        # IVF-PQ is approximate; rebuild may cause stable-tie permutations.
        # Enforce identical top-1 and same membership of the top-k.
        ids1 = _topk_row_idxs(raft1, q, k=10)  # -> [i0, i1, ...] for single query
        ids2 = _topk_row_idxs(raft2, q, k=10)
        assert ids1[0] == ids2[0] == 42
        assert set(ids1) == set(ids2)



# -------------------------
# Integration: consistency across save/load (cosine)
# -------------------------
class TestRaftGPUIntegration:
    def test_retrieval_consistency_across_save_load(self, temp_dir):
        X, metas = _make_small_dataset(1500, 96, seed=3)
        raft1 = RaftGPU(dim=96, metric="cosine")
        raft1.add(X, metas)

        queries = X[[0, 149, 777, 1000]]
        before = [_first_row_metas(raft1, q[None, :], k=10) for q in queries]

        raft1.save(temp_dir, include_dataset=True)
        raft2 = RaftGPU.load(temp_dir, metric="cosine")

        after = [_first_row_metas(raft2, q[None, :], k=10) for q in queries]
        assert before == after

    @pytest.mark.slow
    def test_retrieval_with_fallback_streaming(self, temp_dir):
        rng = np.random.default_rng(123)
        n_vecs, d = 20_000, 64
        X = rng.standard_normal((n_vecs, d), dtype=np.float32)
        metas = [(i,) for i in range(n_vecs)]

        raft1 = RaftGPU(dim=d, metric="cosine")
        raft1.add(X, metas)
        raft1.save(temp_dir, include_dataset=False, fallback_embeddings=X)

        raft2 = RaftGPU.load(temp_dir, metric="cosine")

        q = X[1234:1235]
        row = _first_row_metas(raft2, q, k=5)
        assert row[0] == metas[1234]


# -------------------------
# Indexing smoke tests block
# -------------------------
class TestRaftGPUIndexing:
    def test_add_and_search_numpy(self, sample_data):
        X, metas = sample_data
        raft = RaftGPU(dim=X.shape[1], metric="cosine")
        raft.add(X, metas)
        row = _first_row_metas(raft, X[0:1], k=5)
        assert row[0] == metas[0]

    def test_add_and_search_cupy(self, sample_data):
        X, metas = sample_data
        raft = RaftGPU(dim=X.shape[1], metric="cosine")
        raft.add(cp.asarray(X), metas)
        row = _first_row_metas(raft, X[1:2], k=5)
        assert row[0] == metas[1]

    @pytest.mark.skipif(torch is None, reason="PyTorch not installed")
    def test_add_and_search_torch(self, sample_data):
        X, metas = sample_data
        raft = RaftGPU(dim=X.shape[1], metric="cosine")
        raft.add(torch.from_numpy(X), metas)
        row = _first_row_metas(raft, X[2:3], k=5)
        assert row[0] == metas[2]

    def test_k_clamping_and_shapes(self, sample_data):
        X, metas = sample_data
        raft = RaftGPU(dim=X.shape[1], metric="cosine")
        raft.add(X, metas)
        Q = X[:4]
        k = 7
        D_cp, I_cp = raft.search(Q, k=k)
        assert isinstance(D_cp, cp.ndarray) and isinstance(I_cp, cp.ndarray)
        assert D_cp.shape == (Q.shape[0], k)
        assert I_cp.shape == (Q.shape[0], k)
        # k> N should clamp internally; here k << N so we only check non-empty
        rows = _rows_from_search(raft, (D_cp, I_cp))
        assert len(rows) == Q.shape[0]
        # each row decodes to metas tuples
        assert isinstance(rows[0][0], tuple) and len(rows[0][0]) == len(metas[0])

    def test_batch_query(self, sample_data):
        X, metas = sample_data
        raft = RaftGPU(dim=X.shape[1], metric="cosine")
        raft.add(X, metas)
        Q = X[:10]
        rows = _rows_from_search(raft, raft.search(Q, k=5))
        assert len(rows) == 10
        for i in range(10):
            assert rows[i][0] == metas[i]

# -------------------------
# Retrieval accuracy / consistency
# -------------------------
class TestRaftGPURetrievalAccuracy:
    def test_exact_match_multiple(self):
        rng = np.random.default_rng(42)
        n, d = 1000, 128
        X = rng.standard_normal((n, d), dtype=np.float32)
        metas = [(i,) for i in range(n)]

        raft = RaftGPU(dim=d, metric="cosine")  # default: nn_descent (CAGRA)
        raft.add(X, metas)

        # Query 10 random rows and ensure exact self-match at rank 1
        for _ in range(10):
            idx = int(rng.integers(0, n))
            q = X[idx:idx + 1]  # shape (1, d) -> single-query path
            top = _topk_row_idxs(raft, q, k=1)[0]
            assert top == idx, f"Failed to find exact match for index {idx}"

    def test_retrieval_consistency_across_save_load(self, temp_dir):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((1000, 128), dtype=np.float32)
        metas = [(i,) for i in range(1000)]

        raft1 = RaftGPU(dim=128, metric="cosine")
        raft1.add(X, metas)

        # Take a handful of queries (multi-query path)
        queries = X[::100]  # shape (10, 128)
        before = _topk_row_idxs(raft1, queries, k=10)

        # Save with dataset included to guarantee identical rebuild if needed
        raft1.save(str(temp_dir), include_dataset=True)

        # Load back; may deserialize index or rebuild from persisted embeddings
        raft2 = RaftGPU.load(str(temp_dir), metric="cosine")
        after = _topk_row_idxs(raft2, queries, k=10)

        # Exact equality of the ordered top-k lists
        assert before == after, "Mismatch (with dataset) after save/load"

    def test_retrieval_with_fallback_streaming(self, temp_dir):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5000, 128), dtype=np.float32)
        metas = [(i,) for i in range(5000)]

        raft1 = RaftGPU(dim=128, metric="cosine")
        raft1.add(X, metas)

        q = X[42:43]
        r1 = _topk_row_idxs(raft1, q, k=20)

        # Save WITHOUT dataset but WITH fallback embeddings, so load() can rebuild
        raft1.save(str(temp_dir), include_dataset=False, fallback_embeddings=X)

        raft2 = RaftGPU.load(str(temp_dir), metric="cosine")
        r2 = _topk_row_idxs(raft2, q, k=20)

        assert r1 == r2, "Mismatch after rebuild from fallback embeddings"

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