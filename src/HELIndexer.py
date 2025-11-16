# HELIndexer.py (refactored)
from __future__ import annotations

import json
import logging
from pathlib import Path
import weakref
import atexit
from typing import Iterator, List, Optional, Tuple, Dict, Any
from collections.abc import Mapping
from threading import Thread
from queue import Queue

import numpy as np
import torch
import torch.nn.functional as F

try:
    import pysam  # HTSlib-backed FASTA reader
except Exception as _e:  # pragma: no cover
    pysam = None

from src.configs import IndexConfig
from src.RaftGPU import RaftGPU
# NEW: helper encapsulating LUT/IDs logic
from src.dna_tokenizer import DNATok


# ----------------------------- DNA utils -------------------------------------

_DNA_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq: str) -> str:
    return seq.translate(_DNA_COMP)[::-1]


def _l2_normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x: [N, D] CUDA float32/float16
    return F.normalize(x, p=2, dim=1, eps=eps)


def _l2_normalize_rows_inplace(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.dtype != torch.float32:
        norms = x.float().pow(2).sum(dim=1, keepdim=True).sqrt_().clamp_min_(eps)
        x = x / norms
    else:
        norms = x.pow(2).sum(dim=1, keepdim=True).sqrt_().clamp_min_(eps)
        x = x / norms
    return x


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -------------------------- Lazy FASTA access --------------------------------

class LazyReferenceDict(Mapping):
    def __init__(self, fasta_path: Path) -> None:
        if pysam is None:
            raise RuntimeError("pysam is required for LazyReferenceDict.")
        self._path = Path(fasta_path)
        self.ff = pysam.FastaFile(str(self._path))
        self._refs = tuple(self.ff.references)
        self._lengths = {ctg: self.ff.get_reference_length(ctg) for ctg in self._refs}

    def __getitem__(self, key: str) -> str:
        if key not in self._lengths:
            raise KeyError(key)
        return self.ff.fetch(key, 0, self._lengths[key]).upper()

    def __iter__(self):
        return iter(self._refs)

    def __len__(self) -> int:
        return len(self._refs)

    @property
    def lengths(self) -> Dict[str, int]:
        return self._lengths

    def close(self) -> None:
        try:
            self.ff.close()
        except Exception:
            pass


# ----------------------------- HELIndexer ------------------------------------
class HELIndexer:
    MANIFEST_NAME = "manifest.json"
    HEL_META_NAME = "hel_meta.json"
    INT32_MAX = 2_147_483_647
    DEFAULT_IDS_MAX_TOKENS_PER_CALL = 262_144  # 256k

    def __init__(
        self,
        ref_fasta: Path,
        cfg: IndexConfig,
        embedder: str | object = "hyena",
        emb_batch: int = 256,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.ref_fasta = Path(ref_fasta)
        self.cfg = cfg

        if isinstance(embedder, str):
            key = embedder.lower()
            if key == "nt":
                from src.NTBackend import NTBackend
                self.embedder = NTBackend(model_name=getattr(cfg, "model_name", None),
                                          model_dir=getattr(cfg, "model_dir", None))
            elif key == "hyena":
                from src.HyenaBackend import HyenaBackend
                self.embedder = HyenaBackend(
                    model_name=getattr(cfg, "model_name", None),
                    model_dir=getattr(cfg, "model_dir", None),
                    pooling="mean",
                    normalize=False,
                    offline=True,
                    prefer_cuda=True,
                )
            else:
                from src.DeterministicBackend import DeterministicBackend
                self.embedder = DeterministicBackend(
                    dim=getattr(cfg, "det_dim", 256),
                    k=getattr(cfg, "det_k", 7),
                    rc_merge=True,
                    normalize=False,
                    pos_buckets=getattr(cfg, "det_pos_buckets", 8),
                    num_hashes=getattr(cfg, "det_num_hashes", 2),
                    micro_buckets=getattr(cfg, "det_micro_buckets", 64),
                    power_gamma=getattr(cfg, "det_power_gamma", 0.5),
                    center_slices=True,
                )
        else:
            self.embedder = embedder

        self.emb_batch = int(emb_batch)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.log = logger or logging.getLogger("HELIndexer")
        if not self.log.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            self.log.addHandler(ch)
            self.log.setLevel(logging.INFO)

        if self.device != "cuda":
            self.log.warning("HELIndexer instantiated on '%s'; ANN build expects a GPU.", self.device)

        if pysam is None:
            raise RuntimeError(
                "pysam is required for HELIndexer (FASTA streaming via HTSlib). "
                "Please install pysam."
            )

        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        self.raft: Optional[RaftGPU] = None
        self.meta: Dict[str, Any] = {}
        self.dim: Optional[int] = None

        self.ref_dict = LazyReferenceDict(self.ref_fasta)
        self.contig_lengths: Dict[str, int] = self.ref_dict.lengths

        self.ids_helper = DNATok(
            embedder=self.embedder,
            ids_max_tokens_per_call=int(getattr(self.cfg, "ids_max_tokens_per_call", self.DEFAULT_IDS_MAX_TOKENS_PER_CALL)),
            logger=self.log,
        )

        # ensure cleanup runs before modules are torn down
        wr = weakref.ref(self)
        def _cleanup(ref=wr):
            obj = ref()
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        atexit.register(_cleanup)

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Release GPU/extern resources early to avoid noisy __dealloc__ at shutdown."""
        try:
            if hasattr(self, "raft") and self.raft is not None:
                raft = self.raft
                # try common close/free/destroy names without changing behavior if absent
                for m in ("close", "destroy", "free", "release"):
                    fn = getattr(raft, m, None)
                    if callable(fn):
                        try:
                            fn()
                        except Exception:
                            pass
                self.raft = None
        except Exception:
            pass
        try:
            if hasattr(self, "ref_dict") and self.ref_dict is not None:
                self.ref_dict.close()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # --------------------------- helpers -------------------------------------

    def _cfg_or(self, name: str, default):
        return getattr(self.cfg, name, default)

    def _metric_and_norm(self) -> Tuple[str, bool]:
        metric = str(getattr(self.cfg, "metric", "cosine")).lower()
        if metric not in ("cosine", "l2"):
            self.log.warning("Unknown metric '%s'; defaulting to 'cosine'.", metric)
            metric = "cosine"
        do_norm = (metric == "cosine")
        return metric, do_norm

    def _open_ref(self) -> Tuple[pysam.FastaFile, List[Tuple[str, int]]]:
        ff = pysam.FastaFile(str(self.ref_fasta))
        contigs = [(ctg, ff.get_reference_length(ctg)) for ctg in ff.references]
        return ff, contigs

    def _count_windows(self, contigs: List[Tuple[str, int]]) -> int:
        W = int(self.cfg.window)
        S = int(self.cfg.stride)
        rc = bool(getattr(self.cfg, "rc_index", True))
        total = 0
        if W <= 0:
            return 0
        for _, L in contigs:
            if L < W:
                continue
            if S > 0:
                n = 1 + (L - W) // S
            else:
                n = 1
            total += n
        if rc:
            total *= 2
        return total

    def _iter_windows(
        self, ff: pysam.FastaFile, contigs: List[Tuple[str, int]]
    ) -> Iterator[Tuple[str, int, str, int]]:
        W = int(self.cfg.window)
        S = int(self.cfg.stride)
        rc = bool(getattr(self.cfg, "rc_index", True))
        skip_N_frac = float(getattr(self.cfg, "skip_N_frac", 1.0))

        for chrom, clen in contigs:
            if clen < W:
                continue
            last = clen - W
            if S > 0:
                n_tiles = 1 + (last // S)
            else:
                n_tiles = 1
            for i in range(n_tiles):
                start = i * S if S > 0 else 0
                if start > last:
                    break
                seq = ff.fetch(chrom, start, start + W).upper()
                if skip_N_frac < 1.0:
                    n_frac = seq.count("N") / float(W)
                    if n_frac > skip_N_frac:
                        continue

                yield chrom, start, seq, +1
                if rc:
                    yield chrom, start, revcomp(seq), -1

    def _iter_windows_fast(
        self, ff: pysam.FastaFile, contigs: List[Tuple[str, int]]
    ) -> Iterator[Tuple[str, int, str, int]]:
        W = int(self.cfg.window)
        S = int(self.cfg.stride)
        rc = bool(getattr(self.cfg, "rc_index", True))
        skip_N_frac = float(getattr(self.cfg, "skip_N_frac", 1.0))

        for chrom, clen in contigs:
            if clen < W:
                continue

            s = ff.fetch(chrom, 0, clen).upper()
            rc_s = revcomp(s) if rc else None

            use_n_filter = (skip_N_frac < 1.0)
            if use_n_filter:
                n_cum = np.zeros(clen + 1, dtype=np.int32)
                np_s = np.frombuffer(s.encode("ascii", "ignore"), dtype=np.uint8)
                is_N = (np_s == ord('N')).astype(np.int32)
                np.cumsum(is_N, out=n_cum[1:])
            else:
                n_cum = None  # type: ignore

            last = clen - W
            n_tiles = 1 + (last // S) if S > 0 else 1
            for i in range(n_tiles):
                start = i * S if S > 0 else 0
                if start > last:
                    break

                if use_n_filter:
                    n_in_win = int(n_cum[start + W] - n_cum[start])
                    if (n_in_win / float(W)) > skip_N_frac:
                        continue

                yield chrom, start, s[start:start + W], +1

                if rc:
                    rc_start = clen - (start + W)
                    yield chrom, start, rc_s[rc_start:rc_start + W], -1

    def _first_dim_probe(self, batch: List[str]) -> int:
        x = self.embedder.embed_best(batch)  # [B, D] on CUDA
        if not (isinstance(x, torch.Tensor) and x.is_cuda and x.ndim == 2):
            raise TypeError("embedder.embed_best must return CUDA tensor [B, D]")
        D = int(x.shape[1])
        del x
        return D

    # --------------------------- Public API ----------------------------------

    def build_or_load(
            self,
            outdir: Path,
            reuse_existing: bool = True,
            include_dataset: bool = True,
            verbose: bool = True,
    ) -> None:
        outdir = Path(outdir)
        _ensure_dir(outdir)

        manifest_path = outdir / self.MANIFEST_NAME
        hel_meta_path = outdir / self.HEL_META_NAME

        current_metric, _ = self._metric_and_norm()

        if reuse_existing and manifest_path.exists() and hel_meta_path.exists():
            with open(hel_meta_path) as f:
                on_disk = json.load(f)

            if on_disk.get("window") != int(self.cfg.window):
                if verbose:
                    self.log.warning("Window size mismatch (disk=%s, cfg=%s). Rebuilding.",
                                     on_disk.get("window"), int(self.cfg.window))
                self._build(outdir, include_dataset, verbose)
                return

            if on_disk.get("stride") != int(self.cfg.stride):
                if verbose:
                    self.log.warning("Stride mismatch (disk=%s, cfg=%s). Rebuilding.",
                                     on_disk.get("stride"), int(self.cfg.stride))
                self._build(outdir, include_dataset, verbose)
                return

            if on_disk.get("metric") != current_metric:
                if verbose:
                    self.log.warning("Metric mismatch (disk=%s, cfg=%s). Rebuilding.",
                                     on_disk.get("metric"), current_metric)
                self._build(outdir, include_dataset, verbose)
                return

            if verbose:
                self.log.info("Loading existing index from %s ...", str(outdir))
            self.raft = RaftGPU.load(outdir, metric=current_metric)
            self.dim = self.raft.dim
            self.meta = on_disk
            return

        if verbose:
            self.log.info("Rebuilding at %s ...", str(outdir))
        with torch.inference_mode():
            self._build(outdir=outdir, include_dataset=include_dataset, verbose=verbose)

    # --------------------------- Internals -----------------------------------

    def _raft_params(self) -> Dict[str, Any]:
        return {
            "build_algo":          self._cfg_or("build_algo", "nn_descent"),
            "graph_degree":        int(self._cfg_or("graph_degree", 64)),
            "intermediate_graph_degree": int(self._cfg_or("intermediate_graph_degree", 64)),
            "nn_descent_niter":    int(self._cfg_or("nn_descent_niter", 10)),
            "refinement_rate":     float(self._cfg_or("refinement_rate", 2.0)),
            "search_itopk_size":   int(self._cfg_or("search_itopk_size", 128)),
            "ivf_n_lists":         self._cfg_or("ivf_n_lists", None),
            "ivf_n_probes":        self._cfg_or("ivf_n_probes", None),
            "ivf_pq_dim":          self._cfg_or("ivf_pq_dim", None),
            "ivf_pq_bits":         self._cfg_or("ivf_pq_bits", None),
        }

    def _autotune_batch(self, probe: List[str], start: int, max_factor: int = 64) -> int:
        bs = max(1, int(start))
        limit = start * max_factor
        while bs <= limit:
            try:
                tmp = [probe[0]] * bs
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                    out = self.embedder.embed_best(tmp, rc_invariant=False)
                torch.cuda.synchronize()
                bs *= 2
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    bs = max(start, bs // 2)
                    return bs
                raise
        return min(bs // 2, limit)

    def _build(
            self,
            outdir: Path,
            include_dataset: bool,
            verbose: bool,
    ) -> None:
        W = int(self.cfg.window)
        S = int(self.cfg.stride)
        assert W > 0, "cfg.window must be > 0"
        assert self.emb_batch > 0, "emb_batch must be > 0"

        ff, contigs = self._open_ref()
        try:
            # ---- Probe embedding dim ----
            probe_batch: List[str] = []
            for chrom, clen in contigs:
                if clen >= W:
                    probe_batch.append(ff.fetch(chrom, 0, W).upper())
                    break
            if not probe_batch:
                raise RuntimeError("No contigs >= window length; nothing to index.")

            D = self._first_dim_probe(probe_batch)
            self.dim = D

            # ---- Optional batch autotune ----
            try:
                tuned = self._autotune_batch(probe_batch, start=self.emb_batch)
                if tuned != self.emb_batch:
                    self.emb_batch = tuned
                    self.log.info("Auto-tuned emb_batch=%d", self.emb_batch)
            except Exception:
                pass

            metric, do_norm = self._metric_and_norm()
            raft_kwargs = self._raft_params()

            total_windows = self._count_windows(contigs)
            if total_windows <= 0:
                raise RuntimeError("No windows available for the given window/stride settings.")

            # NEW: discover IDs capability (LUT, PAD/N, token_len)
            self.ids_helper.discover()

            self.log.info(
                "Embedding (GPU streaming, batch=%d, ids_path=%s) ... total windows=%d, dim=%d",
                self.emb_batch, str(self.ids_helper.use_ids_path), total_windows, D
            )

            X = torch.empty((total_windows, D), device="cuda", dtype=torch.float32)
            metas: List[Tuple[str, int, int]] = []
            metas_extend = metas.extend

            # ---------------- Producer: CPU side ----------------
            q: Queue = Queue(maxsize=64)

            def _batch_producer():
                seqs: List[str] = []
                metas_local: List[Tuple[str, int, int]] = []

                for chrom, start, seq, strand in self._iter_windows_fast(ff, contigs):
                    seqs.append(seq)
                    metas_local.append((chrom, start, strand))
                    if len(seqs) >= self.emb_batch:
                        if self.ids_helper.use_ids_path:
                            ids_cpu = self.ids_helper.encode_batch_to_ids(seqs)
                            q.put(("ids", ids_cpu, metas_local))
                        else:
                            q.put(("seqs", list(seqs), metas_local))
                        seqs, metas_local = [], []

                if seqs:
                    if self.ids_helper.use_ids_path:
                        ids_cpu = self.ids_helper.encode_batch_to_ids(seqs)
                        q.put(("ids", ids_cpu, metas_local))
                    else:
                        q.put(("seqs", list(seqs), metas_local))

                q.put(("done", None, None))

            producer = Thread(target=_batch_producer, daemon=True)
            producer.start()

            pos = 0
            item = q.get()
            pending = None if item[0] == "done" else item

            while pending is not None:
                nxt = q.get()
                next_pending = None if nxt[0] == "done" else nxt

                kind, payload, metas_batch = pending

                if kind == "ids":
                    for xb in self.ids_helper.iter_embed_tokens_in_slices(payload, emb_batch=self.emb_batch, device="cuda"):
                        bsz = int(xb.shape[0])
                        X[pos:pos + bsz].copy_(xb)
                        pos += bsz
                    metas_extend(metas_batch)
                else:
                    batch_seqs = payload  # List[str]
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                        xb = self.embedder.embed_best(batch_seqs, rc_invariant=False)
                    if xb.device.type != "cuda":
                        xb = xb.to(device="cuda", non_blocking=True)
                    if xb.dtype != torch.float32:
                        xb = xb.float()
                    bsz = int(xb.shape[0])
                    X[pos:pos + bsz].copy_(xb)
                    metas_extend(metas_batch)
                    pos += bsz

                pending = next_pending

            if pos != total_windows:
                X = X[:pos]
            N = int(X.shape[0])
            assert X.shape[1] == D

            self.log.info(
                "Building ANN (CAGRA/RAFT): N=%d, dim=%d, algo=%s ...",
                N, D, raft_kwargs.get("build_algo", "nn_descent")
            )
            raft = RaftGPU(
                dim=D,
                metric=metric,
                build_algo=raft_kwargs["build_algo"],
                graph_degree=raft_kwargs["graph_degree"],
                intermediate_graph_degree=raft_kwargs["intermediate_graph_degree"],
                nn_descent_niter=raft_kwargs["nn_descent_niter"],
                refinement_rate=raft_kwargs["refinement_rate"],
                search_itopk_size=raft_kwargs["search_itopk_size"],
                ivf_n_lists=raft_kwargs["ivf_n_lists"],
                ivf_n_probes=raft_kwargs["ivf_n_probes"],
                ivf_pq_dim=raft_kwargs["ivf_pq_dim"],
                ivf_pq_bits=raft_kwargs["ivf_pq_bits"],
            )

            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                pool = int(free_bytes * 0.85)
                raft.enable_memory_pool(use_rmm=True, initial_pool_size=pool)
                self.log.info("RMM pool initialized to %.2f GB of free VRAM", pool / (1024 ** 3))
            except Exception:
                pass

            raft.add(X, metas)

            try:
                del X
                torch.cuda.empty_cache()
            except Exception:
                pass

            _ensure_dir(outdir)
            self.log.info("Saving to %s (include_dataset=%s) ...", str(outdir), str(include_dataset))
            manifest = raft.save(outdir)

            hel_meta = {
                "schema": "hel-indexer-v1",
                "created_by": "HELIndexer",
                "reference": str(self.ref_fasta),
                "window": W,
                "stride": S,
                "rc_index": bool(getattr(self.cfg, "rc_index", True)),
                "skip_N_frac": float(getattr(self.cfg, "skip_N_frac", 1.0)),
                "metric": metric,
                "scores_higher_better": (metric == "cosine"),
                "normalized": bool(do_norm),
                "embedder_name": getattr(self.embedder, "name", type(self.embedder).__name__),
                "embedding_dim": D,
                "n_vectors": N,
                "ids_path": bool(self.ids_helper.use_ids_path),
                "token_len_fixed": self.ids_helper.token_len,
                "contigs": [{"name": ctg, "length": length} for ctg, length in contigs],
                "raft_params": raft_kwargs,
                "raft_manifest_file": self.MANIFEST_NAME,
                "raft_manifest": manifest,
            }
            with open(outdir / self.HEL_META_NAME, "w") as f:
                json.dump(hel_meta, f, indent=2)

            self.log.info("Done. Vectors=%d; dim=%d.", N, D)

            self.raft = raft
            self.meta = hel_meta

        finally:
            ff.close()

    # --------------------------- Convenience ---------------------------------

    @property
    def index(self) -> RaftGPU:
        if self.raft is None:
            raise RuntimeError("Index not built/loaded yet. Call build_or_load first.")
        return self.raft

    def info(self) -> Dict[str, Any]:
        if not self.meta:
            raise RuntimeError("No metadata. Build or load an index first.")
        return dict(self.meta)