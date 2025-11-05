from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple


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
    search(Q, k=10, return_distances=False)
    save(dir, include_dataset=True, fallback_embeddings=None) -> dict
    load(dir, metric=None) -> RaftGPU
    dataset_cupy(dtype=None, ensure_row_major=True) -> cupy.ndarray
    """

    # ---------------- init ----------------
    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        *,
        # choose build algorithm
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
        refinement_rate: float = 2.0,     # candidate multiplier when refine=True
        refine: bool = False,             # enable exact re-ranking after IVF-PQ search
        # Search knob for CAGRA
        search_itopk_size: int = 128,
    ) -> None:
        import importlib

        # Lazy imports so the module can import on CPU-only hosts
        self._cp = importlib.import_module("cupy")
        self._np = importlib.import_module("numpy")
        self.cagra = importlib.import_module("cuvs.neighbors.cagra")
        self.ivf_pq = importlib.import_module("cuvs.neighbors.ivf_pq")

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
        self._dataset_cp = None  # CuPy fp32 row-major dataset we built with
        self._row_norm2 = None   # cached row norms for L2 refinement
        self.metas: List[Any] = []  # Python-side metadata, one per row

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
        try:
            return cp.asarray(x)
        except Exception:
            # Fall back: try .__cuda_array_interface__ via memoryview
            try:
                return cp.asarray(memoryview(x))
            except Exception as e:
                raise TypeError(f"Cannot convert object of type {type(x)} to CuPy array") from e

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
                    # Move via DLPack to CuPy
                    t = X.detach().contiguous()
                    X_cp = cp.from_dlpack(to_dlpack(t))
                else:
                    X_cp = cp.asarray(X.detach().cpu().numpy())
            else:
                X_cp = cp.asarray(np.asarray(X))
        # CuPy
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

    def _normalize_inplace_if_cosine(self, A_cp):
        """If metric is cosine, L2-normalize rows (in-place if possible)."""
        if self._metric_name != "inner_product":
            return A_cp
        cp = self._cp
        norms = cp.linalg.norm(A_cp, axis=1, keepdims=True)
        eps = cp.array(1e-12, dtype=A_cp.dtype)
        A_cp /= cp.maximum(norms, eps)
        return A_cp

    # ---------------- build ----------------
    def add(self, X: Any, metas: Sequence[Any]) -> None:
        """
        Build the index from embeddings X (N x D) and attach metas.
        Accepts NumPy, CuPy, or Torch. Device math only; ensures row-major, fp32.
        """
        if self.index is not None and len(self) > 0:
            raise RuntimeError("Index already built; create a new RaftGPU for another dataset.")

        Xg = self._as_device_row_major_f32(X)
        if Xg.shape[1] != self.dim:
            raise ValueError(f"X has dim={Xg.shape[1]}, expected {self.dim}")

        # Normalize for cosine
        Xg = self._normalize_inplace_if_cosine(Xg)

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
        self._row_norm2 = None  # invalidate cached norms (dataset changed)
        self.metas = list(metas)

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

        if ensure_row_major and not V.flags.c_contiguous:
            V = cp.ascontiguousarray(V)
        if dtype is not None and V.dtype != dtype:
            V = V.astype(dtype, copy=False)
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
        # cuVS Python API: no refinement control in SearchParams
        return self.ivf_pq.SearchParams(n_probes=n_probes)

    def _refine_exact(self, Qg, I_cand_cp, k_final: int, *, chunk_queries: int | None = None):
        """
        Vectorized re-ranking on GPU.
        - For cosine: vectors are L2-normalized; rank by descending dot, output distances as -dot.
        - For L2: use exact squared L2 via norm trick: ||v||^2 + ||q||^2 - 2*q·v.
        Processes queries in chunks to keep memory bounded.
        Returns (D_cp, I_cp) for the top-k_final results.
        """
        cp = self._cp
        V = self.dataset_cupy(dtype=cp.float32, ensure_row_major=True)  # (N, D)
        nq, kcand = int(I_cand_cp.shape[0]), int(I_cand_cp.shape[1])
        is_ip = (self._metric_name == "inner_product")

        # Heuristic chunk size to cap working set (~512MB)
        if chunk_queries is None:
            D = int(V.shape[1])
            bytes_per_elem = 4  # float32
            est = kcand * D * bytes_per_elem
            target = 512 * 1024 * 1024
            chunk_queries = max(1, min(nq, target // max(est, 1)))

        # Precompute norms if L2
        qnorm2 = None
        vnorm2 = None
        if not is_ip:
            if self._row_norm2 is None:
                self._row_norm2 = cp.sum(V * V, axis=1).astype(cp.float32, copy=False)  # (N,)
            vnorm2 = self._row_norm2
            qnorm2 = cp.sum(Qg * Qg, axis=1).astype(cp.float32, copy=False)             # (nq,)

        D_out = cp.empty((nq, k_final), dtype=cp.float32)
        I_out = cp.empty((nq, k_final), dtype=I_cand_cp.dtype)

        for start in range(0, nq, chunk_queries):
            end = min(start + chunk_queries, nq)
            Ic = I_cand_cp[start:end]                          # (b, kcand)
            Qc = Qg[start:end]                                 # (b, D)

            # Gather candidate vectors -> (b, kcand, D)
            C = V.take(Ic, axis=0)

            if is_ip:
                # sims = einsum('bkd,bd->bk')
                #sims = cp.einsum('bkd,bd->bk', C, Qc, optimize=True)
                sims = (Qc[:, None, :] @ C.transpose(0, 2, 1)).squeeze(1)
                d = -sims
            else:
                sims = cp.einsum('bkd,bd->bk', C, Qc, optimize=True)  # q·v
                d = (vnorm2.take(Ic, axis=0) + qnorm2[start:end, None] - 2.0 * sims)  # (b,k)

            # Row-wise top-k
            part = cp.argpartition(d, kth=k_final - 1, axis=1)[:, :k_final]            # (b,k)
            d_top = cp.take_along_axis(d, part, axis=1)                                # (b,k)
            order = cp.argsort(d_top, axis=1)                                          # (b,k)
            idx_top = cp.take_along_axis(part, order, axis=1)                          # (b,k)

            I_out[start:end] = cp.take_along_axis(Ic, idx_top, axis=1)
            D_out[start:end] = cp.take_along_axis(d, idx_top, axis=1).astype(cp.float32, copy=False)

            # Free big temporaries early
            del C, sims, d, part, d_top, order, idx_top

        return D_out, I_out

    def search(
        self,
        Q: Any,
        k: int = 10,
        *,
        return_distances: bool = False,  # kept for API compatibility; method always returns (D, I)
    ):
        """
        Search top-k.
        Returns: (D_cp, I_cp) -> both **CuPy** device arrays
        - D_cp: (nq, k) distances (ascending)
        - I_cp: (nq, k) neighbor row indices
        Use `decode(I_cp, D_cp)` to map indices back to your `metas`.
        """

        if self.index is None:
            raise RuntimeError("Index not built")

        cp = self._cp
        Qg = self._as_device_row_major_f32(Q)
        if Qg.shape[1] != self.dim:
            raise ValueError(f"Q has dim={Qg.shape[1]}, expected {self.dim}")

        # Normalize queries for cosine
        Qg = self._normalize_inplace_if_cosine(Qg)

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
        if self._build_algo == "ivf_pq" and I_cp.shape[1] > k_clamped and (self._refine and self._refinement_rate > 1.0):
            D_cp, I_cp = self._refine_exact(Qg, I_cp, k_clamped)
        else:
            D_cp = None

        if return_distances:
            if D_cp is None:
                D_cp = self._to_cupy_array(d_dev)
                if I_cp.shape[1] != int(D_cp.shape[1]):
                    D_cp = D_cp[:, :I_cp.shape[1]]
            if not D_cp.flags.c_contiguous:
                D_cp = cp.ascontiguousarray(D_cp)

        # Ensure we have distances on device and both outputs are contiguous
        if D_cp is None:
            D_cp = self._to_cupy_array(d_dev)
        if not D_cp.flags.c_contiguous:
            D_cp = cp.ascontiguousarray(D_cp)
        if not I_cp.flags.c_contiguous:
            I_cp = cp.ascontiguousarray(I_cp)

        # cuVS convention: return (distances, neighbors), both device arrays
        return D_cp, I_cp

    def decode(self, neighbors, distances=None):
        """
        Map neighbor ids (+ optional distances) into metadata rows.

        Args:
            neighbors: (nq, k) indices; CuPy or NumPy.
            distances: optional (nq, k) distances; CuPy or NumPy.

        Returns:
            list over queries -> list over k -> (*meta_tuple, score?, row_idx)
        """
        np = self._np
        # host views for Python-side meta access
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
                if D is not None:
                    row.append((*meta, float(D[q, j]), idx))
                else:
                    row.append((*meta, idx))
            out.append(row)
        return out

    # ---------------- save / load ----------------
    def _maybe_serialize_index(self) -> tuple[bytes | None, bool]:
        """
        Try to serialize the index using whatever API exists in this cuVS build.
        Returns (bytes_or_None, serialized_flag).
        """
        try:
            ser = getattr(self.cagra, "serialize", None) if self._build_algo == "nn_descent" else getattr(self.ivf_pq, "serialize", None)
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
        X_dev.get(out=host)   # async copy to pinned; syncs on following CPU use
        return np.asarray(host)

    def enable_memory_pool(self, *, use_rmm: bool = True, initial_pool_size: int | None = None):
        """
        Enable a GPU memory pool to reduce cudaMalloc/cudaFree overhead.
        Call once after constructing the class.
        """
        if use_rmm:
            import rmm, cupy as cp
            rmm.reinitialize(
                pool_allocator=True,
                initial_pool_size=initial_pool_size,  # e.g., 8<<30 = 8GB
            )
            cp.cuda.set_allocator(rmm.rmm_cupy_allocator)
        else:
            # Ensure CuPy's default memory pool is initialized
            import cupy as cp
            _ = cp.get_default_memory_pool()

    def save(
        self,
        out_dir: str | os.PathLike,
        *,
        include_dataset: bool = True,
        fallback_embeddings: Any | None = None,
    ) -> dict:
        """
        Save index + metas + manifest.

        If this cuVS build doesn't support (de)serialization, a small placeholder
        index file is written and we persist embeddings so load() can rebuild.
        """
        cp = self._cp
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        files_dict: dict = {}
        serialized_ok = False

        # Try to save index to file using available API
        index_file = "index.cagra" if self._build_algo == "nn_descent" else "index.ivfpq"
        idx_path = out / index_file

        # Attempt method A: library .save(path, index, include_dataset=...)
        try:
            if self._build_algo == "nn_descent":
                self.cagra.save(str(idx_path), self.index, include_dataset=include_dataset)
            else:
                self.ivf_pq.save(str(idx_path), self.index, include_dataset=include_dataset)
            serialized_ok = True
        except Exception:
            # Attempt method B: serialize() -> bytes, then write
            try:
                ser_bytes, ok = self._maybe_serialize_index()
                if ok and ser_bytes is not None:
                    with open(idx_path, "wb") as f:
                        f.write(ser_bytes)
                    serialized_ok = True
            except Exception:
                serialized_ok = False

        if serialized_ok:
            files_dict["index"] = index_file

        # Metas
        metas_path = out / "metas.json"
        with open(metas_path, "w") as f:
            json.dump(self.metas, f)
        files_dict["metas"] = "metas.json"

        # Persist embeddings to guarantee load() can rebuild when needed
        fallback_normalized = None

        if include_dataset:
            if self._dataset_cp is None:
                raise RuntimeError("No in-memory dataset to persist (include_dataset=True).")
            # Use pinned for potentially faster D2H; fall back to cp.asnumpy if needed
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
        )
        (out / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2))

        return asdict(manifest)

    # ------------- load (classmethod) -------------
    @classmethod
    def load(cls, in_dir: str | os.PathLike, metric: str | None = None) -> "RaftGPU":
        import numpy as np
        """Load an index from a directory created by save()."""
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

        # Load metas (required)
        metas_path = in_dir / files["metas"]
        raft.metas = [tuple(x) for x in json.loads(metas_path.read_text())]

        # Try to deserialize the graph index (only if present)
        deserialized = False
        idx_rel = files.get("index")
        if idx_rel:
            try:
                idx_path = in_dir / idx_rel
                if raft._build_algo == "nn_descent":
                    raft.index = raft.cagra.deserialize(idx_path)
                else:
                    raft.index = raft.ivf_pq.deserialize(idx_path)
                deserialized = True
            except Exception:
                raft.index = None
                deserialized = False

        # Case A: index & dataset were saved together → searchable immediately
        if include_dataset_saved and deserialized:
            return raft

        # Case B: dataset was NOT saved → we must rebuild from embeddings even if
        # the graph structure deserializes, because CAGRA/IVF-PQ need dataset vectors.
        emb_rel = files.get("dataset_embeddings") or files.get("fallback_embeddings")
        if not emb_rel:
            raise RuntimeError(
                "Index was saved without dataset and no embeddings are present to rebuild."
            )
        # Clear any half-loaded index so add() can rebuild fresh
        raft.index = None
        if hasattr(raft, "_X_dev"):
            raft._X_dev = None

        X_host = np.load(in_dir / emb_rel).astype(np.float32, copy=False)
        raft.add(X_host, raft.metas)
        return raft
