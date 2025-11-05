# HELIndexer.py

from __future__ import annotations

import json
import logging
from pathlib import Path
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


# ----------------------------- DNA utils -------------------------------------

_DNA_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq: str) -> str:
    return seq.translate(_DNA_COMP)[::-1]


def _l2_normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x: [N, D] CUDA float32/float16
    return F.normalize(x, p=2, dim=1, eps=eps)


def _l2_normalize_rows_inplace(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    In-place(ish) row-wise L2 normalization.
    Keeps x's dtype; uses float32 accumulators for stability when needed.
    """
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
    """
    Read-only, dict-like access to a FASTA reference via pysam/HTSlib.

    - __getitem__(contig) returns the FULL uppercase sequence for that contig,
      fetched on demand (no RAM blow-up unless you actually pull large contigs).
    - .lengths gives {contig: length} without fetching sequences.
    - Close the underlying handle with .close() if desired (HELIndexer holds one).
    """

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
    """
    GPU-first reference window embedder and ANN index builder.

    Lifecycle:
      1) build_or_load(outdir, reuse_existing=..., include_dataset=...)
         -> will build if not present or reuse_existing=False
      2) After build/load, self.raft (RaftGPU) and self.meta are available.

    Parameters
    ----------
    ref_fasta : Path
        Path to the reference FASTA (can be bgzipped and indexed; pysam handles it).
    cfg : IndexConfig
        Indexing configuration: .window (int), .stride (int), .rc_index (bool),
        and optionally .metric ('cosine'|'l2'), .skip_N_frac (float), etc.
    embedder : str | object
        If str in {"hyena","nt","det"} it picks a backend. Otherwise any object
        exposing a) embed_best(List[str]) -> torch.cuda.FloatTensor [B, D],
        and (optionally) b) embed_tokens(LongTensor[B,T]) -> torch.cuda.FloatTensor [B, D].
    emb_batch : int
        Batch size fed into the embedder.
    device : str
        'cuda' (default if available) or 'cpu'. Index building requires GPU.
    logger : logging.Logger | None
        Optional shared logger. If None, a default logger is created.
    """

    MANIFEST_NAME = "manifest.json"      # written by RaftGPU.save(...)
    HEL_META_NAME = "hel_meta.json"      # written by this class

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

        # --- pick or accept provided embedder
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
                    normalize=False,    # we normalize here if cosine
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
            self.embedder = embedder  # custom object

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

        # Prefer faster matmul/convs on NVIDIA GPUs when possible
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                # no-op on older PyTorch; improves perf on newer versions
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        # runtime fields
        self.raft: Optional[RaftGPU] = None
        self.meta: Dict[str, Any] = {}
        self.dim: Optional[int] = None

        self.ref_dict = LazyReferenceDict(self.ref_fasta)
        self.contig_lengths: Dict[str, int] = self.ref_dict.lengths

        # ID path wiring (set during _build if available)
        self._use_ids_path: bool = False
        self._ascii_lut: Optional[np.ndarray] = None  # shape [256], dtype=np.int64
        self._id_pad: int = 0
        self._id_N: int = 0
        self._token_len: Optional[int] = None  # if embedder needs fixed T

    # --------------------------- helpers -------------------------------------

    def _cfg_or(self, name: str, default):
        """Like getattr but stable for optional config fields."""
        return getattr(self.cfg, name, default)

    def _metric_and_norm(self) -> Tuple[str, bool]:
        """
        Decide metric ('cosine' or 'l2') and whether we should L2 normalize.
        Default to 'cosine' if cfg.metric absent. Return (metric, do_norm).
        """
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
                n = 1  # stride==0 -> single window at position 0
            total += n
        if rc:
            total *= 2
        return total

    def _iter_windows(
        self, ff: pysam.FastaFile, contigs: List[Tuple[str, int]]
    ) -> Iterator[Tuple[str, int, str, int]]:
        """
        Yield (chrom, start, seq, strand_sign) for each window.
        strand_sign: +1 for forward, -1 for reverse-complement (when rc_index=True)

        NOTE: Kept for compatibility. See _iter_windows_fast for optimized path.
        """
        W = int(self.cfg.window)
        S = int(self.cfg.stride)
        rc = bool(getattr(self.cfg, "rc_index", True))
        skip_N_frac = float(getattr(self.cfg, "skip_N_frac", 1.0))  # 1.0 => keep all

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

    # ---------------------- Optimized window iterator -------------------------

    def _iter_windows_fast(
        self, ff: pysam.FastaFile, contigs: List[Tuple[str, int]]
    ) -> Iterator[Tuple[str, int, str, int]]:
        """
        Optimized iterator:
          - fetch each contig once
          - optional O(1) N-fraction filtering via cumulative counts
          - precompute reverse-complement once per contig
        Yields the same tuples as _iter_windows.
        """
        W = int(self.cfg.window)
        S = int(self.cfg.stride)
        rc = bool(getattr(self.cfg, "rc_index", True))
        skip_N_frac = float(getattr(self.cfg, "skip_N_frac", 1.0))

        for chrom, clen in contigs:
            if clen < W:
                continue

            # Fetch the whole contig once
            s = ff.fetch(chrom, 0, clen).upper()

            # Precompute RC sequence once per contig if needed
            rc_s = revcomp(s) if rc else None

            # Optional cumulative N counts for O(1) in-window N fraction evaluation
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

                # forward strand
                yield chrom, start, s[start:start + W], +1

                if rc:
                    # reverse-complement window slice:
                    # RC(s[start:start+W]) == rc_s[clen-(start+W): clen-start]
                    rc_start = clen - (start + W)
                    yield chrom, start, rc_s[rc_start:rc_start + W], -1

    def _first_dim_probe(self, batch: List[str]) -> int:
        """
        Run a tiny probe batch through the embedder to learn D.
        Assumes embedder returns torch.cuda.FloatTensor [B, D].
        """
        x = self.embedder.embed_best(batch)  # [B, D] on CUDA
        if not (isinstance(x, torch.Tensor) and x.is_cuda and x.ndim == 2):
            raise TypeError("embedder.embed_best must return CUDA tensor [B, D]")
        D = int(x.shape[1])
        del x
        return D

    # ---------------------- IDs path (tokenizer) ------------------------------

    def _discover_ids_capability(self) -> None:
        """
        Detect whether the embedder exposes an IDs path and construct
        a fast ASCII->id LUT if we can discover a compatible vocabulary.

        We consider the IDs path available if:
          - hasattr(embedder, 'embed_tokens')  (callable)
        and we can pick pad_id & N_id. We do *not* depend on special tokens.
        """
        embed_tokens = getattr(self.embedder, "embed_tokens", None)
        if not callable(embed_tokens):
            self._use_ids_path = False
            return

        # Discover mapping for DNA alphabet and PAD/N
        # Strategy:
        #   1) Query common attributes on embedder/tokenizer for pad id and per-char ids
        #   2) Else, build a fallback mapping {A,C,G,T,N} -> {1,2,3,4,0} with pad=0
        #      and warn once (still safe for models that treat unknowns uniformly)
        pad_id = None
        n_id = None
        char_to_id: Dict[str, int] = {}

        # Direct hints
        for name in ("pad_id", "pad_token_id"):
            v = getattr(self.embedder, name, None)
            if isinstance(v, int):
                pad_id = v
                break
        # Tokenizer hints
        tok = getattr(self.embedder, "tokenizer", None)
        if tok is not None:
            v = getattr(tok, "pad_token_id", None)
            if isinstance(v, int):
                pad_id = pad_id if pad_id is not None else v
            # Vocab dict
            get_vocab = getattr(tok, "get_vocab", None)
            vocab = None
            if callable(get_vocab):
                try:
                    vocab = get_vocab()
                except Exception:
                    vocab = None
            elif hasattr(tok, "vocab"):
                vocab = getattr(tok, "vocab", None)
            if isinstance(vocab, dict):
                # Try simple 1-char tokens
                for ch in ("A", "C", "G", "T", "N"):
                    if ch in vocab and isinstance(vocab[ch], int):
                        char_to_id[ch] = int(vocab[ch])

        # Embedder-side helpers
        for helper in ("token_to_id", "id_for_token"):
            fn = getattr(self.embedder, helper, None)
            if callable(fn):
                for ch in ("A", "C", "G", "T", "N"):
                    try:
                        v = fn(ch)
                        if isinstance(v, int):
                            char_to_id[ch] = int(v)
                    except Exception:
                        pass

        # Heuristic fallback if still missing
        if not char_to_id:
            # Common compact mapping; models often reserve 0 for PAD
            char_to_id = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 0}
            if pad_id is None:
                pad_id = 0
            n_id = char_to_id["N"]
            self.log.warning(
                "IDs path: using fallback DNA vocab {A:1,C:2,G:3,T:4,N:0}, pad=%d. "
                "If your model expects different ids, expose embedder.token_to_id() "
                "or tokenizer vocab.", pad_id
            )
        else:
            # Prefer explicit N if present; else map unknowns to PAD
            n_id = char_to_id.get("N", pad_id if pad_id is not None else 0)
            if pad_id is None:
                # Try common pad tokens
                for t in ("<pad>", "[PAD]", "PAD", "pad"):
                    if tok is not None and hasattr(tok, "convert_tokens_to_ids"):
                        try:
                            pid = tok.convert_tokens_to_ids(t)
                            if isinstance(pid, int) and pid >= 0:
                                pad_id = pid
                                break
                        except Exception:
                            pass
                if pad_id is None:
                    pad_id = 0  # last resort

        # Build ASCII LUT (uint16 -> int64 later). Map everything unknown to N (or PAD).
        lut = np.full(256, n_id if n_id is not None else pad_id, dtype=np.int64)
        for ch, idx in char_to_id.items():
            lut[ord(ch)] = int(idx)
            lc = ch.lower()
            if len(lc) == 1 and lc != ch:
                lut[ord(lc)] = int(idx)

        self._ascii_lut = lut
        self._id_pad = int(pad_id)
        self._id_N = int(n_id if n_id is not None else pad_id)
        # Optional fixed token length hint
        for name in ("model_max_length", "max_position_embeddings", "max_seq_len"):
            v = getattr(self.embedder, name, None)
            if isinstance(v, int) and v > 0:
                self._token_len = v
                break

        self._use_ids_path = True
        self.log.info("IDs path enabled (embed_tokens): PAD=%d, N=%d", self._id_pad, self._id_N)

    @staticmethod
    def _encode_batch_ascii_lut(seqs: List[str], lut: np.ndarray) -> np.ndarray:
        """
        Vectorized ASCII->id encoding using a prebuilt LUT.
        Assumes all seqs have equal length T and contain ASCII chars.
        Returns np.ndarray [B, T] (int64).
        """
        assert len(seqs) > 0
        T = len(seqs[0])
        # Safety: ensure equal lengths
        for s in seqs:
            if len(s) != T:
                raise ValueError("All sequences in a batch must have equal length.")
        # Build a single bytes buffer, then reshape
        buf = ("".join(seqs)).encode("ascii", errors="ignore")
        arr = np.frombuffer(buf, dtype=np.uint8)
        if arr.size != len(seqs) * T:
            # Fallback if non-ASCII leaked in
            arr = np.empty((len(seqs), T), dtype=np.uint8)
            for i, s in enumerate(seqs):
                arr[i, :] = np.frombuffer(s.encode("ascii", errors="replace"), dtype=np.uint8)[:T]
        else:
            arr = arr.reshape(len(seqs), T)
        return lut[arr]  # [B, T], int64

    # --------------------------- Public API ----------------------------------

    def build_or_load(
        self,
        outdir: Path,
        reuse_existing: bool = True,
        include_dataset: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Build the ANN index (and save) or load an existing one from outdir.

        Parameters
        ----------
        outdir : Path
            Directory where index and metadata will be saved or loaded from.
        reuse_existing : bool
            If True and an existing manifest is present, load instead of rebuilding.
        include_dataset : bool
            Pass-through to RaftGPU.save(include_dataset=...). If False, RaftGPU
            will omit dataset persistence (useful for very large corpora).
        verbose : bool
            Extra log lines.
        """
        outdir = Path(outdir)
        _ensure_dir(outdir)

        manifest_path = outdir / self.MANIFEST_NAME
        hel_meta_path = outdir / self.HEL_META_NAME

        if reuse_existing and manifest_path.exists() and hel_meta_path.exists():
            if verbose:
                self.log.info("Loading existing index from %s ...", str(outdir))
            self._load(outdir)
            return

        if verbose:
            self.log.info("Rebuilding at %s ...", str(outdir))

        # Build fresh
        with torch.inference_mode():
            self._build(outdir=outdir, include_dataset=include_dataset, verbose=verbose)

    # --------------------------- Internals -----------------------------------

    def _raft_params(self) -> Dict[str, Any]:
        """
        Collect RaftGPU/ANN parameters from cfg with safe defaults.
        (RaftGPU will clamp small-N cases internally; warnings may appear, which is fine.)
        """
        return {
            "build_algo":          self._cfg_or("build_algo", "nn_descent"),
            "graph_degree":        int(self._cfg_or("graph_degree", 128)),
            "intermediate_graph_degree": int(self._cfg_or("intermediate_graph_degree", 192)),
            "nn_descent_niter":    int(self._cfg_or("nn_descent_niter", 10)),
            "refinement_rate":     float(self._cfg_or("refinement_rate", 2.0)),
            "search_itopk_size":   int(self._cfg_or("search_itopk_size", 128)),
            # IVF/PQ (optional)
            "ivf_n_lists":         self._cfg_or("ivf_n_lists", None),
            "ivf_n_probes":        self._cfg_or("ivf_n_probes", None),
            "ivf_pq_dim":          self._cfg_or("ivf_pq_dim", None),
            "ivf_pq_bits":         self._cfg_or("ivf_pq_bits", None),
        }

    def _autotune_batch(self, probe: List[str], start: int, max_factor: int = 64) -> int:
        """
        Very small helper to ramp batch size to better saturate the GPU.
        Returns a batch size >= start, backing off safely on OOM.
        """
        bs = max(1, int(start))
        limit = start * max_factor
        while bs <= limit:
            try:
                tmp = [probe[0]] * bs
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                    out = self.embedder.embed_best(tmp, rc_invariant=False)
                # synchronize to ensure allocation actually happened
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

            # ---- Optional batch autotune (conservative) ----
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

            # IDs capability (ASCII LUT, PAD/N)
            self._discover_ids_capability()

            self.log.info(
                "Embedding (GPU streaming, batch=%d, ids_path=%s) ... total windows=%d, dim=%d",
                self.emb_batch, str(self._use_ids_path), total_windows, D
            )

            # Single large device buffer in fp32 (cuVS requires fp32)
            X = torch.empty((total_windows, D), device="cuda", dtype=torch.float32)
            metas: List[Tuple[str, int, int]] = []
            metas_extend = metas.extend

            # ---------------- Producer: CPU side ----------------
            from queue import Queue
            q: Queue = Queue(maxsize=64)

            def _batch_producer():
                seqs: List[str] = []
                metas_local: List[Tuple[str, int, int]] = []

                for chrom, start, seq, strand in self._iter_windows_fast(ff, contigs):
                    seqs.append(seq)
                    metas_local.append((chrom, start, strand))
                    if len(seqs) >= self.emb_batch:
                        if self._use_ids_path and self._ascii_lut is not None:
                            ids_np = self._encode_batch_ascii_lut(seqs, self._ascii_lut)
                            if self._token_len is not None and self._token_len > ids_np.shape[1]:
                                pad = self._token_len - ids_np.shape[1]
                                # left-pad (matches existing behavior)
                                ids_np = np.pad(ids_np, ((0, 0), (pad, 0)), constant_values=self._id_pad)
                            ids_cpu = torch.as_tensor(ids_np, dtype=torch.long)
                            try:
                                ids_cpu = ids_cpu.pin_memory()
                            except Exception:
                                pass
                            q.put(("ids", ids_cpu, metas_local))
                        else:
                            q.put(("seqs", list(seqs), metas_local))
                        seqs, metas_local = [], []

                # tail
                if seqs:
                    if self._use_ids_path and self._ascii_lut is not None:
                        ids_np = self._encode_batch_ascii_lut(seqs, self._ascii_lut)
                        if self._token_len is not None and self._token_len > ids_np.shape[1]:
                            pad = self._token_len - ids_np.shape[1]
                            ids_np = np.pad(ids_np, ((0, 0), (pad, 0)), constant_values=self._id_pad)
                        ids_cpu = torch.as_tensor(ids_np, dtype=torch.long)
                        try:
                            ids_cpu = ids_cpu.pin_memory()
                        except Exception:
                            pass
                        q.put(("ids", ids_cpu, metas_local))
                    else:
                        q.put(("seqs", list(seqs), metas_local))

                q.put(("done", None, None))

            producer = Thread(target=_batch_producer, daemon=True)
            producer.start()

            # ---------------- Consumer: GPU side with prefetch ----------------
            copy_stream = torch.cuda.Stream()
            compute_stream = torch.cuda.current_stream()

            pos = 0
            pending = None  # (kind, payload_gpu_or_list, metas, copy_event)

            def _prefetch(item):
                kind, payload, metas_batch = item
                if kind == "ids":
                    # Begin HtoD on a separate stream
                    with torch.cuda.stream(copy_stream):
                        ids_dev = payload.to(device="cuda", non_blocking=True)
                        ev = torch.cuda.Event()
                        ev.record(copy_stream)
                    return ("ids", ids_dev, metas_batch, ev)
                else:
                    # Nothing to pre-copy for raw seqs
                    return ("seqs", payload, metas_batch, None)

            # Prime the pipeline
            item = q.get()
            if item[0] == "done":
                pending = None
            else:
                pending = _prefetch(item)

            while pending is not None:
                # Start prefetch for the next batch (if any) before computing current
                nxt = q.get()
                next_prefetched = None if nxt[0] == "done" else _prefetch(nxt)

                kind, payload_dev_or_list, metas_batch, ev = pending

                # Ensure any pending copy completed before we use the buffer
                if ev is not None:
                    compute_stream.wait_event(ev)

                # Run embed on the compute stream
                if kind == "ids":
                    with torch.cuda.stream(compute_stream), torch.amp.autocast("cuda", dtype=torch.float16,
                                                                               enabled=True):
                        xb = self.embedder.embed_tokens(payload_dev_or_list, rc_invariant=False)  # [B, D] CUDA
                else:
                    batch_seqs = payload_dev_or_list  # List[str]
                    with torch.cuda.stream(compute_stream), torch.amp.autocast("cuda", dtype=torch.float16,
                                                                               enabled=True):
                        xb = self.embedder.embed_best(batch_seqs, rc_invariant=False)

                if xb.device.type != "cuda":
                    xb = xb.to(device="cuda", non_blocking=True)
                if xb.dtype != torch.float32:
                    xb = xb.float()

                bsz = int(xb.shape[0])
                X[pos:pos + bsz].copy_(xb)  # device→device; stays on default stream
                metas_extend(metas_batch)
                pos += bsz

                # Advance
                pending = next_prefetched

            # Adjust for potential N-filtered drops
            if pos != total_windows:
                X = X[:pos]
            N = int(X.shape[0])
            assert X.shape[1] == D

            # ---------------- Build ANN (cuVS) ----------------
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

            # Size RMM pool to CURRENT free memory, not total
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                # keep some headroom for temporary kernels & stream syncs
                pool = int(free_bytes * 0.85)
                raft.enable_memory_pool(use_rmm=True, initial_pool_size=pool)
                self.log.info("RMM pool initialized to %.2f GB of free VRAM", pool / (1024 ** 3))
            except Exception:
                pass

            # Add (zero-copy via DLPack inside RaftGPU) and persist
            raft.add(X, metas)

            # Release Torch handle; dataset now lives in cuVS/CuPy
            try:
                del X
                torch.cuda.empty_cache()
            except Exception:
                pass

            _ensure_dir(outdir)
            self.log.info("Saving to %s (include_dataset=%s) ...", str(outdir), str(include_dataset))
            manifest = raft.save(outdir, include_dataset=include_dataset)

            # HEL-side metadata
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
                "ids_path": bool(self._use_ids_path),
                "token_len_fixed": self._token_len,
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
