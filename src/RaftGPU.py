from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, List, Sequence


@dataclass
class _Manifest:
    schema: str
    created_by: str
    dim: int
    metric: str
    metric_internal: str
    build_algo: str
    include_dataset: bool
    params: dict
    files: dict
    fallback_normalized: bool | None
    index_serialized: bool
    vectors_normalized: bool


class RaftGPU:
    """
    GPU ANN using RAPIDS cuVS (CAGRA graph index or IVF-PQ).
    - Default build: CAGRA NN-Descent (high recall; minimal tuning).
    - Optional build: IVF-PQ with optional exact post-search refinement.
    - Cosine similarity handled by L2 on L2-normalized vectors (internal metric='inner_product').
    - Accepts NumPy, CuPy, or Torch tensors; data is moved to GPU (CuPy) and made row-major fp32.
    - Robust save/load with fallback rebuild if this cuVS build can't (de)serialize indexes.

    Public API
    ----------
    __init__(...)
    add(X, metas)
    search(Q, k=10) -> (D_cp, I_cp)
    save(dir, include_dataset=True, fallback_embeddings=None) -> dict
    load(dir, metric=None) -> RaftGPU
    dataset_cupy(dtype=None, ensure_row_major=True) -> cupy.ndarray

    Allocator notes
    ---------------
    To avoid CUDA misaligned frees when mixing CuPy and cuVS allocations, we by default make CuPy
    use RMM's allocator if available. This can be disabled with env var `RAFTGPU_DISABLE_AUTO_RMM=1`.
    You can also call `RaftGPU.configure_memory_pool()` explicitly at process start.
    """

    # ---------------- init ----------------
    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        *,
        build_algo: str = "nn_descent",  # 'nn_descent' (CAGRA NN-Descent) or 'ivf_pq'
        # NN-Descent knobs
        graph_degree: int = 64,
        intermediate_graph_degree: int = 128,
        nn_descent_niter: int = 20,
        # IVF-PQ knobs (used only if build_algo='ivf_pq')
        ivf_n_lists: int | None = None,
        ivf_n_probes: int | None = None,
        ivf_pq_dim: int = 64,
        ivf_pq_bits: int = 8,
        refinement_rate: float = 2.0,  # candidate multiplier when refine=True
        refine: bool = False,  # enable exact re-ranking after IVF-PQ search
        # Search knob for CAGRA
        search_itopk_size: int = 128,
    ) -> None:
        import importlib

        # Lazy imports so the module can import on CPU-only hosts
        self._cp = importlib.import_module("cupy")
        self._np = importlib.import_module("numpy")
        self.cagra = importlib.import_module("cuvs.neighbors.cagra")
        self.ivf_pq = importlib.import_module("cuvs.neighbors.ivf_pq")

        # --- Allocator harmonization (avoid CuPy/cuvs mismatch) ---
        # By default, switch CuPy to use RMM's allocator if available.
        self._using_rmm_allocator = False
        if os.environ.get("RAFTGPU_DISABLE_AUTO_RMM", "").strip() not in ("1", "true", "True"):
            try:
                import rmm
                # We don't reinitialize here (respect app-level config).
                # We only point CuPy at RMM's allocator so both share the same resource.
                self._cp.cuda.set_allocator(rmm.rmm_cupy_allocator)
                self._using_rmm_allocator = True
            except Exception:
                # RMM not present or failed; carry on with CuPy's default allocator.
                self._using_rmm_allocator = False

        # Basic attrs
        self.dim = int(dim)
        self.metric = metric.lower()
        self._metric_name = self._map_metric(self.metric)  # 'inner_product' or 'sqeuclidean'
        if build_algo not in ("nn_descent", "ivf_pq"):
            raise ValueError("build_algo must be 'nn_descent' or 'ivf_pq'")
        self._build_algo = build_algo

        # Params
        self._graph_degree = int(graph_degree)
        self._intermediate_graph_degree = int(intermediate_graph_degree)
        self._nn_descent_niter = int(nn_descent_niter)

        self._ivf_n_lists = ivf_n_lists
        self._ivf_n_probes = ivf_n_probes
        self._ivf_pq_dim = int(ivf_pq_dim)
        self._ivf_pq_bits = int(ivf_pq_bits)
        self._refinement_rate = float(refinement_rate)
        self._refine = bool(refine)

        # Index+data
        self.index = None  # cuVS index handle
        self._dataset_cp = None  # CuPy fp32 row-major dataset we built with (or wrapped lazily from index.dataset)
        self._row_norm2 = None  # cached row norms for L2 refinement
        self.metas: List[Any] = []  # Python-side metadata, one per row

        # Track whether vectors in _dataset_cp are already normalized
        self._vectors_normalized = False
        # Default CAGRA search params (kept immutable for thread-safety)
        self.search_params = self.cagra.SearchParams(itopk_size=int(search_itopk_size))

    # ---------------- properties & utils ----------------
    def __len__(self) -> int:
        return len(self.metas)

    def __repr__(self) -> str:
        return (
            f"RaftGPU(dim={self.dim}, metric='{self.metric}' -> '{self._metric_name}', "
            f"algo='{self._build_algo}', size={len(self)})"
        )

    @staticmethod
    def _map_metric(metric: str) -> str:
        """
        Map user metric -> cuVS internal metric.
        Best practice: emulate cosine via inner_product on L2-normalized vectors.
        """
        m = metric.lower()
        if m in ("cosine", "cos"):
            return "inner_product"  # normalize rows; rank by dot
        if m in ("ip", "inner_product"):
            return "inner_product"
        if m in ("l2", "euclidean", "sqeuclidean"):
            return "sqeuclidean"
        raise ValueError(f"Unsupported metric '{metric}' (use 'cosine' or 'l2')")

    # ---- Device conversion helpers ----
    @staticmethod
    def _to_cupy_array(x):
        """Convert RAFT/pylibraft device_ndarray or anything with __cuda_array_interface__ to cupy.ndarray."""
        import cupy as cp
        return cp.asarray(x)

    def _as_device_row_major_f32(self, X: Any):
        """
        Ensure CuPy fp32 C-contiguous (row-major) array.
        Accepts NumPy, CuPy, or Torch.
        """
        cp = self._cp
        np = self._np

        # Torch
        if "torch" in str(type(X)):
            import torch
            from torch.utils.dlpack import to_dlpack
            if isinstance(X, torch.Tensor):
                if X.device.type == "cuda":
                    t = X.detach().contiguous()
                    X_cp = cp.from_dlpack(to_dlpack(t))
                else:
                    X_cp = cp.asarray(X.detach().cpu().numpy())
            else:
                X_cp = cp.asarray(np.asarray(X))
        # CuPy or CUDA array interface
        elif hasattr(X, "__cuda_array_interface__"):
            X_cp = cp.asarray(X)
        # NumPy or array-like
        else:
            X_cp = cp.asarray(np.asarray(X))

        if X_cp.dtype != cp.float32:
            X_cp = X_cp.astype(cp.float32, copy=False)
        if X_cp.ndim != 2:
            raise ValueError(f"expected 2D array, got shape={X_cp.shape}")
        if not X_cp.flags.c_contiguous:
            X_cp = cp.ascontiguousarray(X_cp)
        return X_cp

    def _normalize_inplace_if_cosine(self, A_cp, force: bool = False):
        """
        If metric is cosine, L2-normalize rows (in-place if possible).

        Args:
            A_cp: CuPy array to normalize
            force: If True, normalize regardless of _vectors_normalized flag
                   (used when normalizing query vectors)
        """
        if self._metric_name != "inner_product":
            return A_cp

        # Don't double-normalize dataset vectors
        if not force and self._vectors_normalized and A_cp is self._dataset_cp:
            return A_cp

        cp = self._cp
        norms = cp.linalg.norm(A_cp, axis=1, keepdims=True)
        eps = cp.array(1e-12, dtype=A_cp.dtype)
        A_cp /= cp.maximum(norms, eps)
        return A_cp

    # ---------------- build ----------------
    def add(
        self,
        X: Any,
        metas: Sequence[Any],
        *,
        _skip_normalization: bool = False
    ) -> None:
        """
        Build the index from embeddings X (N x D) and attach metas.
        Accepts NumPy, CuPy, or Torch. Device math only; ensures row-major, fp32.

        Args:
            X: Embeddings (NumPy, CuPy, Torch)
            metas: Sequence of metadata, one per row
            _skip_normalization: (Internal) If True, skip normalization even
                if metric is cosine. Used by load() when rebuilding from
                already-normalized vectors.
        """
        if self.index is not None and len(self) > 0:
            raise RuntimeError("Index already built; create a new RaftGPU for another dataset.")

        Xg = self._as_device_row_major_f32(X)
        if Xg.shape[1] != self.dim:
            raise ValueError(f"X has dim={Xg.shape[1]}, expected {self.dim}")

        # Normalize for cosine, unless explicitly skipped (e.g., by load())
        if not _skip_normalization:
            Xg = self._normalize_inplace_if_cosine(Xg, force=True)
            self._vectors_normalized = (self._metric_name == "inner_product")
        else:
            # trusting caller (load()) that vectors are already normalized if cosine
            self._vectors_normalized = (self._metric_name == "inner_product")

        if self._build_algo == "nn_descent":
            # Index & Build params (CAGRA)
            ip = self.cagra.IndexParams(
                metric=self._metric_name,
                build_algo="nn_descent",
                graph_degree=int(self._graph_degree),
                intermediate_graph_degree=int(self._intermediate_graph_degree),
                nn_descent_niter=int(self._nn_descent_niter),
            )
            self.index = self.cagra.build(ip, Xg)
        else:
            # IVF-PQ
            n_rows = int(Xg.shape[0])
            n_lists = int(self._ivf_n_lists or max(1, int(round(math.sqrt(n_rows)))))
            ip = self.ivf_pq.IndexParams(
                metric=self._metric_name,
                n_lists=n_lists,
                pq_bits=int(self._ivf_pq_bits),
                pq_dim=int(self._ivf_pq_dim),
            )
            self.index = self.ivf_pq.build(ip, Xg)

        self._dataset_cp = Xg
        self._precompute_row_norms()  # compute and cache norms after build
        self.metas = list(metas)

    def _precompute_row_norms(self) -> None:
        """Precompute and cache row norms for L2 distance refinement."""
        if self._dataset_cp is None:
            self._row_norm2 = None
            return

        cp = self._cp
        # Only compute for L2 metric (for cosine, vectors are unit norm)
        if self._metric_name != "inner_product":
            self._row_norm2 = cp.sum(self._dataset_cp * self._dataset_cp, axis=1).astype(cp.float32, copy=False)
        else:
            self._row_norm2 = None

    # ---------------- dataset exposure ----------------
    def dataset_cupy(self, dtype=None, ensure_row_major=True):
        """
        Return a CuPy view of the dataset attached to the index.
        Prefers the array we built with (self._dataset_cp); otherwise tries index.dataset.
        """
        cp = self._cp
        if self.index is None:
            raise RuntimeError("Index not built")

        V = self._dataset_cp
        if V is None:
            ds = getattr(self.index, "dataset", None)
            if ds is None:
                raise RuntimeError("Index has no attached dataset")
            V = self._to_cupy_array(ds)

        # normalize view guarantees
        if ensure_row_major and not V.flags.c_contiguous:
            V = cp.ascontiguousarray(V)
        if dtype is not None and V.dtype != dtype:
            V = V.astype(dtype, copy=False)

        # cache the normalized/correct view to avoid repeating work
        self._dataset_cp = V
        return V

    # ---------------- search ----------------
    def _clone_search_params_if_needed(self, k: int):
        """CAGRA: make a local SearchParams if k > itopk_size (thread-safe)."""
        sp = self.search_params
        if k <= int(sp.itopk_size):
            return sp
        return self.cagra.SearchParams(itopk_size=max(int(sp.itopk_size), int(k)))

    def _ivf_search_params(self):
        """Build a fresh IVF-PQ SearchParams with sensible defaults."""
        n_probes = int(self._ivf_n_probes or 64)
        return self.ivf_pq.SearchParams(n_probes=n_probes)

    def _refine_exact(self, Qg, I_cand_cp, k_final: int, *, chunk_queries: int | None = None):
        """
        Vectorized re-ranking on GPU.
        - For cosine: vectors are L2-normalized; rank by descending dot, output distances as -dot.
        - For L2: use exact squared L2 via norm trick: ||v||^2 + ||q||^2 - 2*q·v.
        """
        cp = self._cp
        V = self.dataset_cupy(dtype=cp.float32, ensure_row_major=True)  # (N, D)
        nq, kcand = int(I_cand_cp.shape[0]), int(I_cand_cp.shape[1])
        is_ip = (self._metric_name == "inner_product")

        if chunk_queries is None:
            D = int(V.shape[1])
            bytes_per_elem = 4  # float32
            est = kcand * D * bytes_per_elem
            target = 512 * 1024 * 1024
            chunk_queries = max(1, min(nq, target // max(est, 1)))

        qnorm2 = None
        vnorm2 = None
        if not is_ip:
            if self._row_norm2 is None:
                self._precompute_row_norms()
            vnorm2 = self._row_norm2
            qnorm2 = cp.sum(Qg * Qg, axis=1).astype(cp.float32, copy=False)  # (nq,)

        D_out = cp.empty((nq, k_final), dtype=cp.float32)
        I_out = cp.empty((nq, k_final), dtype=I_cand_cp.dtype)

        for start in range(0, nq, chunk_queries):
            end = min(start + chunk_queries, nq)
            Ic = I_cand_cp[start:end]  # (b, kcand)
            Qc = Qg[start:end]  # (b, D)

            # Gather candidate vectors -> (b, kcand, D)
            C = V.take(Ic, axis=0)

            if is_ip:
                # sims = einsum('bkd,bd->bk') via batched matmul
                sims = (Qc[:, None, :] @ C.transpose(0, 2, 1)).squeeze(1)
                d = -sims
            else:
                sims = cp.einsum('bkd,bd->bk', C, Qc, optimize=True)  # q·v
                d = (vnorm2.take(Ic, axis=0) + qnorm2[start:end, None] - 2.0 * sims)  # (b,k)

            # Row-wise top-k
            part = cp.argpartition(d, kth=k_final - 1, axis=1)[:, :k_final]  # (b,k)
            d_top = cp.take_along_axis(d, part, axis=1)  # (b,k)
            order = cp.argsort(d_top, axis=1)  # (b,k)
            idx_top = cp.take_along_axis(part, order, axis=1)  # (b,k)

            I_out[start:end] = cp.take_along_axis(Ic, idx_top, axis=1)
            D_out[start:end] = cp.take_along_axis(d, idx_top, axis=1).astype(cp.float32, copy=False)

            del C, sims, d, part, d_top, order, idx_top

        return D_out, I_out

    def search(self, Q: Any, k: int = 10):
        """
        Search top-k.
        Returns: (D_cp, I_cp) -> both **CuPy** device arrays
        - D_cp: (nq, k) distances (ascending)
        - I_cp: (nq, k) neighbor row indices
        """
        if self.index is None:
            raise RuntimeError("Index not built")

        cp = self._cp
        Qg = self._as_device_row_major_f32(Q)
        if Qg.shape[1] != self.dim:
            raise ValueError(f"Q has dim={Qg.shape[1]}, expected {self.dim}")

        # Normalize queries for cosine (always force normalization for queries)
        Qg = self._normalize_inplace_if_cosine(Qg, force=True)

        n = len(self.metas)
        if n == 0:
            raise RuntimeError("Empty index")
        k_clamped = max(1, min(int(k), n))

        if self._build_algo == "nn_descent":
            sp = self._clone_search_params_if_needed(k_clamped)
            d_dev, i_dev = self.cagra.search(sp, self.index, Qg, k_clamped)
        else:
            sp = self._ivf_search_params()
            # Expand candidate set if refinement is enabled
            if self._refine and self._refinement_rate > 1.0:
                k_cand = max(k_clamped, int(math.ceil(k_clamped * self._refinement_rate)))
            else:
                k_cand = k_clamped
            d_dev, i_dev = self.ivf_pq.search(sp, self.index, Qg, k_cand)

        # RAFT device_ndarray -> CuPy, contiguous on device
        I_cp = self._to_cupy_array(i_dev)
        if not I_cp.flags.c_contiguous:
            I_cp = cp.ascontiguousarray(I_cp)

        # Optional exact re-ranking for IVF-PQ
        if self._build_algo == "ivf_pq" and self._refine and I_cp.shape[1] > k_clamped:
            D_cp, I_cp = self._refine_exact(Qg, I_cp, k_clamped)
        else:
            D_cp = self._to_cupy_array(d_dev)
            if not D_cp.flags.c_contiguous:
                D_cp = cp.ascontiguousarray(D_cp)

            # Ensure D/I match k_clamped if we fetched k_cand but didn't refine
            if I_cp.shape[1] != k_clamped:
                I_cp = I_cp[:, :k_clamped]
                D_cp = D_cp[:, :k_clamped]

        return D_cp, I_cp

    def decode(self, neighbors, distances=None):
        """
        Map neighbor ids (+ optional distances) into metadata rows.

        Returns:
            list over queries -> list over k -> (meta_obj_or_fields..., score?, row_idx)
        """
        np = self._np
        I = np.asarray(neighbors)
        D = np.asarray(distances) if distances is not None else None

        out = []
        for q in range(I.shape[0]):
            row = []
            for j in range(I.shape[1]):
                idx = int(I[q, j])
                if idx < 0:
                    continue
                meta = self.metas[idx]
                # Preserve old behavior for list/tuple metas; otherwise keep object as a single field
                if isinstance(meta, (list, tuple)):
                    meta_fields = tuple(meta)
                else:
                    meta_fields = (meta,)
                if D is not None:
                    row.append((*meta_fields, float(D[q, j]), idx))
                else:
                    row.append((*meta_fields, idx))
            out.append(row)
        return out

    # ---------------- save / load ----------------
    def _maybe_serialize_index(self) -> tuple[bytes | None, bool]:
        """
        Try to serialize the index using whatever API exists in this cuVS build.
        Returns (bytes_or_None, serialized_flag).
        """
        try:
            ser = getattr(self.cagra, "serialize", None) if self._build_algo == "nn_descent" else getattr(
                self.ivf_pq, "serialize", None
            )
            if callable(ser):
                return ser(self.index), True
        except Exception:
            pass
        return None, False

    def _to_host_pinned(self, X_dev):
        """
        Copy a device array to a pinned host ndarray (may be slightly faster for large transfers).
        """
        import numpy as np
        from cupy.cuda import alloc_pinned_memory
        nbytes = int(X_dev.nbytes)
        buf = alloc_pinned_memory(nbytes)
        host = np.ndarray(X_dev.shape, dtype=X_dev.dtype, buffer=buf)
        X_dev.get(out=host)  # async copy to pinned; syncs on following CPU use
        return np.asarray(host)

    @staticmethod
    def configure_memory_pool(*, initial_pool_size: int | None = None) -> None:
        """
        Make CuPy use RMM so cuVS and CuPy share the same cudaMallocAsync pool.
        Call once per process, before any large allocations.
        """
        import rmm, cupy as cp
        rmm.reinitialize(pool_allocator=True, initial_pool_size=initial_pool_size)
        cp.cuda.set_allocator(rmm.rmm_cupy_allocator)

    def enable_memory_pool(self, *, use_rmm: bool = True, initial_pool_size: int | None = None):
        """
        Enable a GPU memory pool to reduce cudaMalloc/cudaFree overhead.
        Call once after constructing the class.

        (Kept for backward compatibility; prefer `configure_memory_pool` at process start.)
        """
        if use_rmm:
            import rmm, cupy as cp
            rmm.reinitialize(
                pool_allocator=True,
                initial_pool_size=initial_pool_size,  # e.g., 8<<30 = 8GB
            )
            cp.cuda.set_allocator(rmm.rmm_cupy_allocator)
            self._using_rmm_allocator = True
        else:
            import cupy as cp
            _ = cp.get_default_memory_pool()
            self._using_rmm_allocator = False

    def save(
        self,
        out_dir: str | os.PathLike,
        *,
        include_dataset: bool = True,
        fallback_embeddings: Any | None = None,
    ) -> dict:
        """
        Save index + metas + manifest.

        Strategy:
        1) Prefer new path-based `cuvs.neighbors.*.save()` (can embed dataset on device).
        2) Fallback to bytes-based serialize() -> write as a single index file.
        3) Persist embeddings **only if needed** for rebuilds (avoid device→host copies otherwise).
        """
        cp = self._cp
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        files_dict: dict = {}
        serialized_ok = False
        used_path_based = False

        # Try to save index to file using available API
        index_file = "index.cagra" if self._build_algo == "nn_descent" else "index.ivfpq"
        idx_path = out / index_file

        try:
            # Attempt 1: New path-based save (preferred)
            saver = getattr(self.cagra, "save", None) if self._build_algo == "nn_descent" else getattr(
                self.ivf_pq, "save", None
            )
            if callable(saver):
                saver(str(idx_path), self.index, include_dataset=include_dataset)
                serialized_ok = True
                used_path_based = True
        except Exception:
            serialized_ok = False
            used_path_based = False
            if idx_path.exists():
                idx_path.unlink()

        # Attempt 2: Older bytes-based serialize()
        if not serialized_ok:
            try:
                ser_bytes, ok = self._maybe_serialize_index()
                if ok and ser_bytes is not None:
                    with open(idx_path, "wb") as f:
                        f.write(ser_bytes)
                    serialized_ok = True
            except Exception:
                serialized_ok = False
                if idx_path.exists():
                    idx_path.unlink()

        if serialized_ok:
            files_dict["index"] = index_file

        # Metas (tiny; host write is fine)
        metas_path = out / "metas.json"
        with open(metas_path, "w") as f:
            json.dump(self.metas, f)
        files_dict["metas"] = "metas.json"

        # Persist embeddings only if they are needed for a future rebuild:
        # - If we used path-based save and include_dataset=True, the dataset is embedded: no .npy dump.
        # - Else, we write embeddings so load() can rebuild.
        fallback_normalized = None
        need_embeddings_dump = True
        if used_path_based and include_dataset:
            need_embeddings_dump = False  # dataset is inside the saved index

        if need_embeddings_dump:
            if include_dataset:
                if self._dataset_cp is None:
                    raise RuntimeError("No in-memory dataset to persist (include_dataset=True).")
                try:
                    X_host = self._to_host_pinned(self._dataset_cp)
                except Exception:
                    X_host = cp.asnumpy(self._dataset_cp)
                ds_path = out / "dataset_f32.npy"
                with open(ds_path, "wb") as f:
                    self._np.save(f, X_host)
                files_dict["dataset_embeddings"] = "dataset_f32.npy"
            else:
                if fallback_embeddings is None:
                    raise ValueError("include_dataset=False requires fallback_embeddings")
                Xg = self._as_device_row_major_f32(fallback_embeddings)
                if self._metric_name == "inner_product":
                    norms = cp.linalg.norm(Xg, axis=1)
                    fallback_normalized = bool(cp.all(cp.abs(norms - 1.0) < 1e-3).item())
                try:
                    X_host = self._to_host_pinned(Xg)
                except Exception:
                    X_host = cp.asnumpy(Xg)
                fb_path = out / "embeddings_f32.npy"
                with open(fb_path, "wb") as f:
                    self._np.save(f, X_host)
                files_dict["fallback_embeddings"] = "embeddings_f32.npy"

            # Save row norms if they exist (for L2 metric) only when we're already dumping arrays.
            if self._row_norm2 is not None:
                try:
                    norms_host = self._to_host_pinned(self._row_norm2)
                except Exception:
                    norms_host = cp.asnumpy(self._row_norm2)
                norms_path = out / "row_norms.npy"
                with open(norms_path, "wb") as f:
                    self._np.save(f, norms_host)
                files_dict["row_norms"] = "row_norms.npy"

        # manifest
        manifest = _Manifest(
            schema="raftgpu-manifest-v1",
            created_by="RaftGPU",
            dim=self.dim,
            metric=self.metric,
            metric_internal=self._metric_name,
            build_algo=self._build_algo,
            include_dataset=bool(include_dataset),
            params=dict(
                graph_degree=self._graph_degree,
                intermediate_graph_degree=self._intermediate_graph_degree,
                nn_descent_niter=self._nn_descent_niter,
                ivf_n_lists=self._ivf_n_lists,
                ivf_n_probes=self._ivf_n_probes,
                ivf_pq_dim=self._ivf_pq_dim,
                ivf_pq_bits=self._ivf_pq_bits,
                refinement_rate=self._refinement_rate,
                refine=self._refine,
                search_itopk_size=int(self.search_params.itopk_size),
            ),
            files=files_dict,
            fallback_normalized=fallback_normalized,
            index_serialized=serialized_ok,
            vectors_normalized=self._vectors_normalized,
        )
        (out / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2))

        return asdict(manifest)

    # ------------- load (classmethod) -------------
    @classmethod
    def load(cls, in_dir: str | os.PathLike, metric: str | None = None) -> "RaftGPU":
        import numpy as np
        """
        Load an index from a directory created by save().

        Strategy:
        1) Try path-based `cuvs.neighbors.*.load()` (preferred; can restore dataset inside index).
        2) Fallback to bytes-based `deserialize()`.
        3) If neither yields a ready-to-search index+dataset, rebuild from persisted embeddings.
        """
        in_dir = Path(in_dir)
        manifest_path = in_dir / "manifest.json"
        m = json.loads(manifest_path.read_text())
        files = m.get("files", {})

        dim = int(m["dim"])
        saved_metric_raw = m["metric"]
        user_metric = metric or saved_metric_raw
        # Enforce metric consistency (compare cuVS-internal mapped metrics)
        saved_internal = cls._map_metric(saved_metric_raw)
        requested_internal = cls._map_metric(user_metric)
        if saved_internal != requested_internal:
            raise RuntimeError(
                f"Metric mismatch: saved='{saved_metric_raw}'→'{saved_internal}', "
                f"requested='{user_metric}'→'{requested_internal}'"
            )
        include_dataset_saved = bool(m.get("include_dataset", False))
        vectors_normalized_manifest = bool(m.get("vectors_normalized", False))
        fallback_normalized_manifest = m.get("fallback_normalized", None)

        params = m.get("params", {})
        raft = cls(
            dim=dim,
            metric=user_metric,
            build_algo=m.get("build_algo", "nn_descent"),
            graph_degree=params.get("graph_degree", 64),
            intermediate_graph_degree=params.get("intermediate_graph_degree", 128),
            nn_descent_niter=params.get("nn_descent_niter", 20),
            ivf_n_lists=params.get("ivf_n_lists"),
            ivf_n_probes=params.get("ivf_n_probes"),
            ivf_pq_dim=params.get("ivf_pq_dim", 64),
            ivf_pq_bits=params.get("ivf_pq_bits", 8),
            refinement_rate=params.get("refinement_rate", 2.0),
            refine=params.get("refine", False),
            search_itopk_size=params.get("search_itopk_size", 128),
        )

        # Load metas (preserve original types; don't coerce to tuples)
        metas_path = in_dir / files["metas"]
        raft.metas = json.loads(metas_path.read_text())

        # Try to deserialize the graph index (only if present)
        deserialized = False
        idx_rel = files.get("index")
        if idx_rel:
            idx_path = in_dir / idx_rel

            # Attempt 1: New path-based load (preferred)
            try:
                loader = getattr(raft.cagra, "load", None) if raft._build_algo == "nn_descent" else getattr(
                    raft.ivf_pq, "load", None
                )
                if callable(loader):
                    raft.index = loader(str(idx_path))
                    deserialized = True
            except Exception:
                raft.index = None
                deserialized = False

            # Attempt 2: Older bytes-based deserialize
            if not deserialized:
                try:
                    with open(idx_path, "rb") as f:
                        blob = f.read()
                    deser = getattr(raft.cagra, "deserialize", None) if raft._build_algo == "nn_descent" else getattr(
                        raft.ivf_pq, "deserialize", None
                    )
                    if callable(deser):
                        raft.index = deser(blob)
                        deserialized = True
                except Exception:
                    raft.index = None
                    deserialized = False

        # Identify any persisted embeddings on disk (for rebuilds or attach-only scenarios)
        emb_rel = files.get("dataset_embeddings") or files.get("fallback_embeddings")
        using_fallback = bool(files.get("fallback_embeddings"))
        dataset_normalized = (
            bool(fallback_normalized_manifest)
            if using_fallback and fallback_normalized_manifest is not None
            else vectors_normalized_manifest
        )

        # ---- Case A: index was saved with dataset and path-deserialized successfully ----
        # Trust the dataset embedded in the index; do NOT force-load a .npy; avoid eager CuPy wrapping.
        if deserialized and include_dataset_saved:
            # Leave dataset under cuVS/RMM control; wrap lazily in dataset_cupy() if needed.
            raft._dataset_cp = None
            raft._vectors_normalized = vectors_normalized_manifest
            raft._row_norm2 = None
            return raft

        # ---- Case B: need to attach embeddings and possibly rebuild ----
        if not emb_rel:
            # No disk embeddings to rebuild from
            if deserialized:
                # Index might still be usable (e.g., CAGRA) without dataset;
                # we set _dataset_cp=None and allow search (refine disabled)
                raft._dataset_cp = None
                raft._row_norm2 = None
                raft._vectors_normalized = vectors_normalized_manifest
                return raft
            raise RuntimeError(
                "Index was saved without dataset and no embeddings are present to rebuild."
            )

        # Load embeddings from disk -> device (only when necessary)
        X_host = np.load(in_dir / emb_rel).astype(np.float32, copy=False)
        X_dev = raft._as_device_row_major_f32(X_host)

        if deserialized:
            # We have an index (likely bytes-deserialized) but it doesn't carry dataset.
            # Attach dataset to this wrapper (for refine/dataset_cupy); no rebuild.
            raft._dataset_cp = X_dev
            raft._vectors_normalized = vectors_normalized_manifest if not using_fallback else bool(dataset_normalized)
            raft._row_norm2 = None  # computed lazily if needed
            return raft

        # No index -> rebuild from embeddings
        raft.index = None
        raft._dataset_cp = None
        raft._vectors_normalized = False  # will be set by add()
        raft.add(X_dev, raft.metas, _skip_normalization=bool(dataset_normalized))
        return raft
