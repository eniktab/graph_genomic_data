# HELIndexer.py

from typing import List, Tuple, Dict, Optional
from pathlib import Path

from src.configs import IndexConfig
from src.RaftGPU import RaftGPU

# DNA utilities
_DNA_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq: str) -> str:
    return seq.translate(_DNA_COMP)[::-1]

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

class HELIndexer:
    """
    High-level genome embed + ANN index builder/loader.

    Steps:
      1. Load reference FASTA once (CPU).
      2. Stream windows, embed on GPU in batches, keep device-resident.
      3. Build GPU ANN index (RaftGPU -> cuVS CAGRA).
      4. Save index + metadata + fallback embeddings (if needed).

    Notes on device residency:
      - All embedding tensors are kept on GPU (CUDA, float32, contiguous) by default.
      - We only touch CPU when on-disk formats require it; with the updated RaftGPU.save(),
        we can pass a CUDA tensor directly and it will write via DLPack→CuPy→cp.save.
    """

    def __init__(self, ref_fasta_path: Path | str, cfg: IndexConfig, backend: str = "hyena", emb_batch: int = 256):
        self.ref_fasta_path = Path(ref_fasta_path)
        self.cfg = cfg
        self.backend = backend
        self.EMB_BATCH = int(max(1, emb_batch))

        # Backend contract:
        # - .max_length (optional int)
        # - .embed_best(seq_batch) -> torch.Tensor [batch, dim] (preferably CUDA, fp32)
        # - .fingerprint() -> dict (optional; for compatibility checks)

        # Populated by _load_ref()
        self.ref_dict: Dict[str, str] = {}

        self.index: RaftGPU | None = None
        self.dim: Optional[int] = None
        self.n_vec: Optional[int] = None  # number of indexed vectors

        if backend == "nt":
            from src.NTBackend import NTBackend
            self.embedder = NTBackend(model_name=cfg.model_name, model_dir=cfg.model_dir)
        elif backend == "hyena":
            from src.HyenaBackend import HyenaBackend
            self.embedder = HyenaBackend(
                model_name=cfg.model_name,
                model_dir=cfg.model_dir,
                pooling="mean",
                normalize=False,
                offline=True,
                prefer_cuda=True,
            )
        else:
            from src.DeterministicBackend import DeterministicBackend
            self.embedder = DeterministicBackend(
                dim=256,
                k=7,
                rc_merge=True,
                normalize=False,
                pos_buckets=8,
                num_hashes=2,
                micro_buckets=64,
                power_gamma=0.5,
                center_slices=True,
            )

    # ---------- internal helpers ----------

    def _load_ref(self) -> None:
        """Read FASTA (optionally .gz/.bgz) into memory once (CPU).
        Stores only an id->sequence dict (upper-cased).
        """
        from pathlib import Path
        from Bio import SeqIO
        import gzip

        # Prefer BGZF reader if Biopython provides it (handles .gz and .bgz)
        try:
            from Bio import bgzf
            bgzf_open = bgzf.open  # type: ignore[attr-defined]
        except Exception:
            bgzf_open = None

        p = Path(self.ref_fasta_path)
        is_gz = str(p).lower().endswith((".gz", ".bgz", ".bgzip"))

        if is_gz:
            opener = bgzf_open if bgzf_open is not None else gzip.open
            mode = "rt"  # text mode
            with opener(p, mode) as handle:
                recs = list(SeqIO.parse(handle, "fasta"))
        else:
            with open(p, "rt") as handle:
                recs = list(SeqIO.parse(handle, "fasta"))

        if not recs:
            raise ValueError(f"No sequences in reference FASTA: {self.ref_fasta_path}")

        self.ref_dict = {r.id: str(r.seq).upper() for r in recs}

    def _iter_windows(self, seq: str, window: int, stride: int):
        """Yield (start, subseq) covering the sequence end with a tail window if needed."""
        L = len(seq)
        T = int(window)
        S = int(stride)
        if L <= T:
            yield 0, seq
            return
        # regular sliding
        for start in range(0, L - T + 1, S):
            yield start, seq[start : start + T]
        # ensure tail coverage if not aligned
        last_start = L - T
        if (L - T) % S != 0:
            yield last_start, seq[last_start:]

    def _count_windows_for_len(self, L: int, T: int, S: int) -> int:
        # number of sliding windows with guaranteed tail coverage
        if L <= T:
            return 1
        base = (L - T) // S + 1
        tail = 1 if (L - T) % S != 0 else 0
        return base + tail

    def _estimate_total_windows(self, window: int, stride: int) -> int:
        # exact total windows across chromosomes; account for rc_index duplication (given eff_window/eff_stride)
        T = min(int(window), getattr(self.embedder, "max_length", int(window)))
        S = max(1, int(stride))
        rc = 2 if getattr(self.cfg, "rc_index", False) else 1
        total = 0
        for _, seq in getattr(self, "ref_dict", {}).items():
            L = len(seq)
            total += self._count_windows_for_len(L, T, S) * rc
        return int(total)

    def _index_paths(self, base: Path) -> Dict[str, Path]:
        """Centralize path layout of a HEL index dir."""
        base = Path(base)
        return {
            "dir": base,
            "cagra_index": base / "index.cagra",
            "raft_manifest": base / "manifest.json",
            "indexer_manifest": base / "indexer_manifest.json",
            "fallback_embeddings": base / "embeddings_f32.npy",
            "metas": base / "metas.json",
            "metas_indexer": base / "metas_indexer.json",
        }

    def _looks_like_index(self, base: Path) -> Tuple[bool, str]:
        """Basic filesystem probe so we can reuse an existing index_dir."""
        p = self._index_paths(base)
        if not p["dir"].exists():
            return False, "index dir missing"
        if not p["raft_manifest"].exists():
            return False, "manifest.json missing"
        if not (p["cagra_index"].exists() or p["fallback_embeddings"].exists()):
            return False, "no CAGRA blob or fallback embeddings"
        if not (p["metas"].exists() or p["metas_indexer"].exists()):
            return False, "metas missing"
        return True, "ok"

    def _cfg_or(self, name: str, default):
        """Like getattr but stable for optional config fields."""
        return getattr(self.cfg, name, default)

    def _write_indexer_manifest(self, out_dir: Path, n_vec: int, dim: int, include_dataset: bool) -> None:
        """Save HELIndexer-specific metadata for compatibility checks."""
        import json

        p = self._index_paths(out_dir)
        backend_fp = getattr(self.embedder, "fingerprint", lambda: {"type": "unknown"})()
        manifest = {
            "schema": "HELIndexer.indexer_manifest.v1",
            "created_by": "HELIndexer",
            "backend_name": self.backend,
            "backend_fingerprint": backend_fp,
            "n_vec": int(n_vec),
            "dim": int(dim),
            "cfg": {
                "window": int(self.cfg.window),
                "stride": int(self.cfg.stride),
                "rc_index": bool(self.cfg.rc_index),
                "include_dataset": bool(include_dataset),
                "ann_metric": self.cfg.ann_metric,
            },
        }
        p["indexer_manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @staticmethod
    def _ensure_cuda_fp32(x):
        """Normalize backend output: CUDA, FP32, contiguous, detached (minimize copies)."""
        import torch
        if not isinstance(x, torch.Tensor):
            raise TypeError("Backend must return a torch.Tensor")
        if (x.device.type != "cuda") or (x.dtype != torch.float32) or (not x.is_contiguous()):
            x = x.to(device="cuda", dtype=torch.float32, non_blocking=True).contiguous()
        return x.detach()

    # ---------- public API ----------

    def build_or_load(
        self,
        out_dir: Path | str,
        reuse_existing: bool = True,
        include_dataset: bool = True,
        verbose: bool = True,
    ) -> "HELIndexer":
        """
        Build the index if needed, otherwise load it from disk.

        include_dataset:
            True  => keep raw vectors fused inside the ANN file (bigger file, faster load).
            False => save raw FP32 embeddings separately on disk and stitch them back in on load.
        """
        out_dir = Path(out_dir)
        ok, msg = self._looks_like_index(out_dir)

        if reuse_existing and ok:
            if verbose:
                print(f"[HELIndexer] Reusing index at {out_dir} ({msg}).")
            return self.load_from_dir(out_dir, verbose=verbose)

        if verbose:
            note = "rebuilding" if ok and not reuse_existing else "building"
            print(f"[HELIndexer] {note} at {out_dir} ...")

        return self.build(out_dir, include_dataset=include_dataset, verbose=verbose)

    def build(
        self,
        out_dir: Path | str,
        include_dataset: bool = True,
        verbose: bool = True,
    ) -> "HELIndexer":
        """
        Core build:
          - stream windows from reference
          - embed in EMB_BATCH chunks (torch.inference_mode to avoid autograd graphs)
          - concat/preallocate embeddings on GPU
          - build ANN (RaftGPU)
          - save
        """
        import json
        import torch

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)  # ensure output dir exists early

        if not hasattr(self, "backend") or self.backend is None:
            raise RuntimeError("backend is not set on HELIndexer; set self.backend before build()")

        if not self.ref_dict:
            self._load_ref()

        # respect backend.max_length if present
        eff_window = min(int(self.cfg.window), getattr(self.embedder, "max_length", int(self.cfg.window)))
        eff_stride = max(1, int(self.cfg.stride))

        if verbose and eff_window != self.cfg.window:
            print(f"[HELIndexer] Clamped window {self.cfg.window} -> {eff_window} due to backend.max_length.")

        metas: List[Tuple[str, int, int]] = []  # (chrom, start_bp, orientation [+1 / -1])

        if verbose:
            print(f"[HELIndexer] Embedding windows (streaming, batch={self.EMB_BATCH}) ...")

        # Pre-allocate final embedding matrix on GPU, VRAM-aware.
        total_est = self._estimate_total_windows(eff_window, eff_stride)
        embs = None
        write_pos = 0

        # Stream: we DO NOT hold all sequences in memory, only current batch.
        with torch.inference_mode():
            seq_batch: List[str] = []
            meta_batch: List[Tuple[str, int, int]] = []

            for chrom, seq in self.ref_dict.items():
                for start_bp, sub in self._iter_windows(seq, eff_window, eff_stride):
                    seq_batch.append(sub)
                    meta_batch.append((chrom, start_bp, +1))

                    # reverse complement if requested
                    if self.cfg.rc_index:
                        seq_batch.append(revcomp(sub))
                        meta_batch.append((chrom, start_bp, -1))

                    # flush when batch is full
                    if len(seq_batch) >= self.EMB_BATCH:
                        part = self.embedder.embed_best(seq_batch)
                        part = self._ensure_cuda_fp32(part)  # keep on GPU

                        # lazy init of storage and adaptive batch size
                        if embs is None:
                            self.dim = int(part.shape[1])
                            # VRAM-aware cap
                            try:
                                free_bytes, _ = torch.cuda.mem_get_info()
                            except Exception:
                                free_bytes = 0
                            bpr = part.element_size() * part.shape[1]  # bytes per row (4 * dim for fp32)
                            max_rows_by_mem = int(max(1, (0.60 * free_bytes) // max(1, bpr))) if free_bytes else int(total_est)
                            alloc_n = min(int(total_est), max_rows_by_mem) if total_est > 0 else max_rows_by_mem
                            alloc_n = max(alloc_n, int(part.shape[0]))
                            embs = torch.empty((alloc_n, self.dim), device="cuda", dtype=torch.float32)
                            # also adapt EMB_BATCH upwards if memory allows (under ~20% of free VRAM)
                            try:
                                target = int(0.20 * free_bytes)
                                max_rows = int(max(1, target // max(1, bpr))) if free_bytes else self.EMB_BATCH
                                self.EMB_BATCH = max(1, min(self.EMB_BATCH, max_rows))
                            except Exception:
                                pass

                        n = int(part.shape[0])
                        # ensure capacity; grow on device if underestimated
                        if embs is not None and write_pos + n > embs.shape[0]:
                            new_n = max(write_pos + n, int(embs.shape[0] * 3 // 2))
                            _new = torch.empty((new_n, self.dim), device="cuda", dtype=torch.float32)
                            _new[:write_pos].copy_(embs[:write_pos], non_blocking=True)
                            embs = _new
                        embs[write_pos : write_pos + n].copy_(part, non_blocking=True)
                        write_pos += n
                        metas.extend(meta_batch)
                        seq_batch.clear()
                        meta_batch.clear()

            # flush tail
            if seq_batch:
                part = self.embedder.embed_best(seq_batch)
                part = self._ensure_cuda_fp32(part)  # keep on GPU

                if embs is None:
                    self.dim = int(part.shape[1])
                    alloc_n = int(part.shape[0])
                    embs = torch.empty((alloc_n, self.dim), device="cuda", dtype=torch.float32)

                n = int(part.shape[0])
                # ensure capacity; grow on device if underestimated
                if embs is not None and write_pos + n > embs.shape[0]:
                    new_n = max(write_pos + n, int(embs.shape[0] * 3 // 2))
                    _new = torch.empty((new_n, self.dim), device="cuda", dtype=torch.float32)
                    _new[:write_pos].copy_(embs[:write_pos], non_blocking=True)
                    embs = _new
                embs[write_pos : write_pos + n].copy_(part, non_blocking=True)
                write_pos += n
                metas.extend(meta_batch)

        # Trim allocation to actual size
        if embs is None:
            raise RuntimeError("No embeddings produced; check reference/tiling.")
        if write_pos != embs.shape[0]:
            embs = embs[:write_pos].contiguous()

        if embs.numel() == 0:
            raise RuntimeError("No embeddings produced; check reference/tiling.")
        if embs.device.type != "cuda":
            # Defensive guard: we expect GPU here for RaftGPU interop
            embs = embs.cuda(non_blocking=True)

        self.dim = int(embs.shape[1])
        self.n_vec = int(embs.shape[0])

        if verbose:
            algo = self._cfg_or("build_algo", "nn_descent")
            print(f"[HELIndexer] Building ANN with CAGRA: N={self.n_vec}, dim={self.dim}, algo={algo} ...")

        # Build ANN (ensure metric is cfg.ann_metric)
        self.index = RaftGPU(
            dim=self.dim,
            metric=self.cfg.ann_metric,
            build_algo=self._cfg_or("build_algo", "nn_descent"),
            graph_degree=self._cfg_or("graph_degree", 64),
            intermediate_graph_degree=self._cfg_or("intermediate_graph_degree", 128),
            nn_descent_niter=self._cfg_or("nn_descent_niter", 20),
            ivf_n_lists=self._cfg_or("ivf_n_lists", None),
            ivf_n_probes=self._cfg_or("ivf_n_probes", None),
            ivf_pq_dim=self._cfg_or("ivf_pq_dim", 64),
            ivf_pq_bits=self._cfg_or("ivf_pq_bits", 8),
            refinement_rate=self._cfg_or("refinement_rate", 2.0),
            search_itopk_size=self._cfg_or("search_itopk_size", 128),
        )
        self.index.add(embs, metas)

        if verbose:
            print(f"[HELIndexer] Saving to {out_dir} (include_dataset={include_dataset}) ...")

        # Save to disk: pass GPU tensor directly when not embedding dataset.
        self.index.save(
            out_dir,
            include_dataset=include_dataset,
            fallback_embeddings=None if include_dataset else embs,
        )

        # Write indexer manifest (HELIndexer metadata separate from RAFT manifest)
        self._write_indexer_manifest(out_dir, n_vec=self.n_vec, dim=self.dim, include_dataset=include_dataset)

        # Save metas in a HELIndexer-specific extra file (human-readable)
        (Path(out_dir) / "metas_indexer.json").write_text(
            json.dumps(
                [{"chrom": c, "start": int(s), "orient": int(o)} for (c, s, o) in metas],
                indent=2,
            ),
            encoding="utf-8",
        )

        if verbose:
            print(f"[HELIndexer] Done. Vectors={self.n_vec}; dim={self.dim}.")

        # Attempt early cleanup to release VRAM / RAM immediately
        try:
            del embs
            import torch as _t
            _t.cuda.empty_cache()
        except Exception:
            pass

        return self

    def load_from_dir(self, index_dir: Path | str, verbose: bool = True) -> "HELIndexer":
        """
        Load an already-built index from disk.
        - Reconstruct RaftGPU.
        - Confirm backend compatibility via backend_fingerprint.
        """
        import json

        index_dir = Path(index_dir)
        ok, msg = self._looks_like_index(index_dir)
        if not ok:
            raise FileNotFoundError(f"Invalid index at {index_dir}: {msg}")

        if verbose:
            print(f"[HELIndexer] Loading CAGRA index from {index_dir} ...")

        # 1. Load RaftGPU (ensure metric is consistent with cfg)
        self.index = RaftGPU.load(index_dir, metric=self.cfg.ann_metric)

        # 2. Pull dimension/meta info
        self.dim = int(self.index.dim)
        try:
            self.n_vec = int(len(self.index))
        except Exception:
            self.n_vec = int(len(getattr(self.index, "metas", [])))

        # 3. Sanity: compare backend fingerprint if available
        manifest_path = Path(index_dir) / "indexer_manifest.json"
        if manifest_path.exists():
            idx_manifest = json.loads(manifest_path.read_text())
            saved_fp = idx_manifest.get("backend_fingerprint", {})
            current_fp = getattr(self.embedder, "fingerprint", lambda: {"type": "unknown"})()
            if saved_fp != current_fp:
                raise RuntimeError(
                    "Index backend fingerprint mismatch.\n"
                    f"saved:   {saved_fp}\n"
                    f"current: {current_fp}\n"
                    "Rebuild the index or regenerate embeddings to proceed safely."
                )

        if verbose:
            metas_len = len(getattr(self.index, "metas", []))
            print(f"[HELIndexer] Loaded. dim={self.dim}, metas={metas_len}.")
        return self
