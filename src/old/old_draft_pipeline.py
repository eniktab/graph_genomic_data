from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
import re
import logging

import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/g/data/te53/en9803/workspace/sync/ANU/graph_genomic_data/'])

# Try to enable GPU path (optional dependency). We only use CuPy if present.
try:
    import cupy as cp  # type: ignore
    _HAS_CUPY = True
except Exception:
    cp = None  # type: ignore
    _HAS_CUPY = False

# ===============================================================
# Utilities
# ===============================================================

_DNA_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

def revcomp(seq: str) -> str:
    """Reverse-complement a DNA string (A/C/G/T/N only)."""
    return seq.translate(_DNA_COMP)[::-1]

_CIGAR_TOK_RE = re.compile(r"(\d+)([MIDNSHP=XBmidnshp=xb])")


def cigar_tokens(cigar: Optional[str]) -> List[Tuple[int, str]]:
    """Parse a CIGAR string into (length, op) tokens. Upper-cases ops."""
    if not cigar:
        return []
    return [(int(n), op.upper()) for n, op in _CIGAR_TOK_RE.findall(cigar)]


def affine_distance_from_cigar(cigar: Optional[str], x: int, o: int, e: int) -> int:
    """Compute affine *distance* from a CIGAR.
    - mismatches: 'X' → cost x per base
    - gaps: contiguous runs of 'I' or 'D' → o + e * run_len
    - 'M' or '=' → 0
    If the CIGAR uses only 'M' (no 'X'), only gap costs are counted.
    Returns an integer distance (lower is better).
    """
    if not cigar:
        return 0

    toks = cigar_tokens(cigar)
    dist = sum(n for n, op in toks if op == "X") * int(x)

    prev = None
    run = 0
    # Sentinel to flush a trailing gap run
    for n, op in toks + [(0, None)]:
        if op in ("I", "D"):
            if prev == op:
                run += n
            else:
                if prev in ("I", "D"):
                    dist += int(o) + int(e) * run
                prev, run = op, n
        else:
            if prev in ("I", "D"):
                dist += int(o) + int(e) * run
            prev, run = op, 0
    return int(dist)


def span_from_cigar(cigar: Optional[str]) -> tuple[int, int, int, int]:
    """Derive (begin_query, end_query, begin_ref, end_ref) from CIGAR.
    - Leading I/S advances query start; D/N advances ref start
    - H/P/B consume neither
    - Trailing S/H/P/B do not extend the aligned span
    Coordinates are 0-based half-open relative to the input strings.
    """
    toks = cigar_tokens(cigar)
    if not toks:
        return 0, 0, 0, 0

    bq = br = 0

    # Skip through leading ops until first aligned column (M/= /X)
    i = 0
    while i < len(toks):
        n, op = toks[i]
        if op in ("M", "=", "X"):
            break
        if op in ("I", "S"):
            bq += n
        elif op in ("D", "N"):
            br += n
        # H/P/B consume neither
        i += 1

    q_used = t_used = 0
    for n, op in toks[i:]:
        if op in ("M", "=", "X"):
            q_used += n
            t_used += n
        elif op == "I":
            q_used += n
        elif op in ("D", "N"):
            t_used += n
        # S/H/P/B after alignment start do not extend the core span

    return bq, bq + q_used, br, br + t_used


# ===============================================================
# Hit + chaining
# ===============================================================

@dataclass(slots=True)
class Hit:
    q_start: int      # query(read) start bp for this ~window-bp tile
    q_end: int        # query(read) end bp
    chrom: str        # reference contig/chrom ID
    s_start: int      # reference start for the matching ref tile
    s_end: int        # reference end
    strand: int       # +1 or -1
    score: float      # ANN similarity score (higher = better)


def _gather_hits(
    q_starts: List[int],
    window: int,
    D,
    I,
    metas: List[Tuple[str, int, int]],
    *,
    higher_better: bool,
) -> List[Hit]:
    """CPU fallback: Convert ANN results into a flat list of Hit objects efficiently.

    If D/I are CuPy arrays, we copy them to host **once** (bulk) and avoid per-element
    device syncs. This is the fallback path when GPU grouping/DP is not taken.
    """
    # If CuPy, cross to host ONCE (metas holds Python strings; GPU join isn’t feasible here)
    is_cp = _HAS_CUPY and isinstance(D, cp.ndarray) and isinstance(I, cp.ndarray)  # type: ignore

    D_h = (cp.asnumpy(D) if is_cp else np.asarray(D))
    I_h = (cp.asnumpy(I) if is_cp else np.asarray(I))

    if D_h.shape != I_h.shape:
        raise ValueError(f"D and I shapes must match, got {D_h.shape} vs {I_h.shape}")
    if D_h.shape[0] != len(q_starts):
        raise ValueError("Row count of D/I must equal number of query tiles")

    nq, k = I_h.shape
    tile_idx = np.repeat(np.arange(nq, dtype=np.int32), k)          # which tile per (row,col)
    flat_I = I_h.reshape(-1).astype(np.int64, copy=False)
    flat_D = D_h.reshape(-1).astype(np.float32, copy=False)
    mask = flat_I >= 0

    tile_idx = tile_idx[mask]
    flat_I = flat_I[mask]
    flat_D = flat_D[mask]

    q_starts_arr = np.asarray(q_starts, dtype=np.int32)
    q_start_flat = q_starts_arr[tile_idx]
    q_end_flat = q_start_flat + int(window)

    hits: List[Hit] = []
    # Vectorized prep above; now a tight Python loop to attach metas (strings)
    for idx, q_start, q_end, raw in zip(flat_I, q_start_flat, q_end_flat, flat_D):
        try:
            chrom, s_start, strand = metas[int(idx)]
        except Exception as e:
            raise IndexError(f"metas lookup failed for idx={int(idx)}: {e}")
        s_end = int(s_start) + int(window)
        score = float(raw) if higher_better else -float(raw)
        hits.append(
            Hit(
                q_start=int(q_start),
                q_end=int(q_end),
                chrom=str(chrom),
                s_start=int(s_start),
                s_end=int(s_end),
                strand=1 if int(strand) >= 0 else -1,
                score=score,
            )
        )
    return hits


def _group_hits_arrays(hits: List[Hit]):
    """CPU: Group hits by (chrom, strand) and prepare arrays for DP chaining.
    Returns list of tuples: ((chrom, strand), q_arr, s_arr, sc_arr, idx_arr)
    Arrays are sorted by (q, then s) for stable chaining.
    """
    from collections import defaultdict
    groups = defaultdict(lambda: {"q": [], "s": [], "sc": [], "idx": []})

    for i, h in enumerate(hits):
        key = (h.chrom, h.strand)
        g = groups[key]
        g["q"].append(h.q_start)
        g["s"].append(h.s_start)
        g["sc"].append(h.score)
        g["idx"].append(i)

    out = []
    for key, g in groups.items():
        q = np.asarray(g["q"], dtype=np.int32)
        s = np.asarray(g["s"], dtype=np.int32)
        sc = np.asarray(g["sc"], dtype=np.float32)
        idx = np.asarray(g["idx"], dtype=np.int32)

        order = np.lexsort((s, q))  # sort by q then s
        out.append((key, q[order], s[order], sc[order], idx[order]))
    return out


def _dp_chain_single(
    q: np.ndarray,      # query coords (sorted)
    s: np.ndarray,      # subject coords (sorted alongside q)
    sc: np.ndarray,     # ANN scores (higher better)
    strand: int,        # +1 / -1 orientation
    gap_lambda: float,  # penalty scaling
    *,
    max_q_gap: int = 40_000,
    diag_band: int = 10_000,
):
    """CPU quadratic DP chaining with monotonic constraints and banding."""
    n = int(len(q))
    if n == 0:
        return 0.0, np.asarray([], dtype=np.int32)

    dp = np.asarray(sc, dtype=np.float32).copy()
    prv = np.full(n, -1, dtype=np.int32)

    for i in range(n):
        qi = int(q[i])
        si = int(s[i])
        # binary search to find earliest j with q[j] >= qi - max_q_gap
        j_lo = np.searchsorted(q, qi - int(max_q_gap), side='left')
        for j in range(j_lo, i):
            # monotonic constraints
            if strand == 1 and s[j] >= si:
                continue
            if strand == -1 and s[j] <= si:
                continue

            dq = qi - int(q[j])
            ds = (si - int(s[j])) * strand
            dev = (ds - dq)

            # band pruning
            if abs(dev) > diag_band:
                continue

            gap_pen = float(gap_lambda) * (dev * dev) / (1000.0 + abs(dq))
            cand = dp[j] + sc[i] - gap_pen
            if cand > dp[i]:
                dp[i] = cand
                prv[i] = j

    k = int(np.argmax(dp))
    best = float(dp[k])

    order = []
    while k != -1:
        order.append(k)
        k = int(prv[k])
    order.reverse()
    return best, np.asarray(order, dtype=np.int32)


# ---------------- GPU-accelerated gather + chaining ----------------

@dataclass(slots=True)
class _GPUHits:
    q_start: "cp.ndarray"
    s_start: "cp.ndarray"
    score: "cp.ndarray"
    chrom_id: "cp.ndarray"
    strand_sign: "cp.ndarray"  # +1 or -1 (int8/int32)
    chrom_table: List[str]     # id -> chrom string
    window: int

def _ensure_meta_arrays(metas: List[Tuple[str, int, int]]):
    """Build CPU & GPU side meta arrays once: chrom_id_of_meta, s_start_of_meta, strand_of_meta, and chrom table."""
    # Build chrom string table
    chrom_to_id: Dict[str, int] = {}
    chrom_table: List[str] = []
    chrom_ids = np.empty(len(metas), dtype=np.int32)
    s_starts = np.empty(len(metas), dtype=np.int32)
    strands = np.empty(len(metas), dtype=np.int8)
    for i, (chrom, s_start, strand) in enumerate(metas):
        if chrom not in chrom_to_id:
            chrom_to_id[chrom] = len(chrom_table)
            chrom_table.append(chrom)
        chrom_ids[i] = chrom_to_id[chrom]
        s_starts[i] = int(s_start)
        strands[i] = 1 if int(strand) >= 0 else -1
    return chrom_table, chrom_ids, s_starts, strands

def _gather_hits_arrays_gpu(
    q_starts: List[int],
    window: int,
    D_cp: "cp.ndarray",
    I_cp: "cp.ndarray",
    metas: List[Tuple[str, int, int]],
    *,
    higher_better: bool,
    meta_cache_gpu: Optional[Tuple[List[str], "cp.ndarray", "cp.ndarray", "cp.ndarray"]] = None,
) -> _GPUHits:
    """GPU gather: materialize hit fields as CuPy arrays; no per-element host sync."""
    assert _HAS_CUPY
    if D_cp.shape != I_cp.shape:
        raise ValueError(f"D and I shapes must match, got {D_cp.shape} vs {I_cp.shape}")
    if D_cp.shape[0] != len(q_starts):
        raise ValueError("Row count of D/I must equal number of query tiles")

    nq, k = D_cp.shape
    # Flatten and mask valid indices (>=0)
    flat_I = I_cp.reshape(-1).astype(cp.int64, copy=False)
    flat_D = D_cp.reshape(-1).astype(cp.float32, copy=False)
    mask = flat_I >= 0
    flat_I = flat_I[mask]
    flat_D = flat_D[mask]

    # Tile index per (row,col)
    tile_idx = cp.repeat(cp.arange(nq, dtype=cp.int32), k)[mask]
    q_starts_cp = cp.asarray(q_starts, dtype=cp.int32)
    q_start_flat = q_starts_cp[tile_idx]
    sgn = 1.0 if higher_better else -1.0
    score_flat = flat_D * sgn

    # Prepare meta arrays on GPU (from cache if available)
    if meta_cache_gpu is not None:
        chrom_table, chrom_ids_cp, s_starts_cp, strands_cp = meta_cache_gpu
    else:
        chrom_table, chrom_ids_np, s_starts_np, strands_np = _ensure_meta_arrays(metas)
        chrom_ids_cp = cp.asarray(chrom_ids_np, dtype=cp.int32)
        s_starts_cp = cp.asarray(s_starts_np, dtype=cp.int32)
        strands_cp = cp.asarray(strands_np, dtype=cp.int8)

    # Gather per-hit metadata on GPU
    chrom_id_flat = chrom_ids_cp[flat_I]
    s_start_flat = s_starts_cp[flat_I]
    strand_flat = strands_cp[flat_I].astype(cp.int32, copy=False)  # +/-1

    return _GPUHits(
        q_start=q_start_flat,
        s_start=s_start_flat,
        score=score_flat,
        chrom_id=chrom_id_flat,
        strand_sign=strand_flat,
        chrom_table=chrom_table,
        window=int(window),
    )

def _dp_chain_single_gpu(q: "cp.ndarray",
                         s: "cp.ndarray",
                         sc: "cp.ndarray",
                         strand_sign: int,
                         gap_lambda: float,
                         *,
                         max_q_gap: int = 40_000,
                         diag_band: int = 10_000):
    """GPU quadratic DP chaining for one (chrom,strand) group with banding.
    q, s, sc are 1D CuPy arrays already sorted by (q, then s).
    Returns (best_score: float, order_idx: np.ndarray[int32] on CPU).
    """
    assert _HAS_CUPY
    n = int(q.size)
    if n == 0:
        return 0.0, np.asarray([], dtype=np.int32)

    dp  = sc.astype(cp.float32, copy=True)
    prv = cp.full(n, -1, dtype=cp.int32)
    lam = cp.float32(gap_lambda)
    sgn_val = cp.int32(1 if int(strand_sign) >= 0 else -1)

    for i in range(1, n):
        qi = q[i]; si = s[i]
        qj = q[:i]; sj = s[:i]

        # masks
        mask_q = (qj < qi)
        # max_q_gap band
        mask_q = mask_q & (qj >= qi - cp.int32(max_q_gap))
        mask_s = (sj < si) if int(strand_sign) >= 0 else (sj > si)
        dq = qi - qj
        ds = (si - sj) * sgn_val
        dev = ds - dq
        mask_diag = (cp.abs(dev) <= diag_band)
        mask = mask_q & mask_s & mask_diag

        if not mask.any().item():
            continue

        gap_pen = lam * (dev * dev) / (cp.float32(1000.0) + cp.abs(dq))
        cand = dp[:i] + sc[i] - gap_pen
        cand = cp.where(mask, cand, -cp.inf)

        j_best = cp.argmax(cand)
        v_best = cand[j_best]
        take = v_best > dp[i]
        # in-place update without host sync
        dp[i]  = cp.where(take, v_best.astype(cp.float32), dp[i])
        prv[i] = cp.where(take, j_best.astype(cp.int32), prv[i])

    k = int(cp.argmax(dp).item())
    best = float(dp[k].item())

    # backtrack on device, then move to CPU once
    order_gpu = []
    while k != -1:
        order_gpu.append(k)
        k = int(prv[k].item())
    order_gpu.reverse()
    order = np.asarray(order_gpu, dtype=np.int32)
    return best, order

def _chain_from_gpu_hits(
    gpuh: _GPUHits,
    chain_gap_lambda: float,
    chain_min_hits: int,
    take_top_chains: int,
    dp_max_q_gap: int,
    dp_diag_band: int,
    debug_snapshot: Callable[[str, dict], None] | None = None,
) -> List["ChainCandidate"]:
    """Group by (chrom_id, strand) on GPU, sort by (q,s), run GPU DP per group, and
    return top ChainCandidates. Only small summaries cross to CPU.
    """
    assert _HAS_CUPY
    q = gpuh.q_start
    s = gpuh.s_start
    sc = gpuh.score
    cid = gpuh.chrom_id
    sgn = gpuh.strand_sign  # per-hit ±1

    # Composite group key: cid * 2 + (sgn==-1 ? 1 : 0)
    grp = cid * 2 + (sgn < 0).astype(cp.int32)

    # new: stack keys into a 2D array (k, N); last row is the primary key
    # We want order by (grp, q, s) ⇒ keys rows = [s, q, grp]
    keys = cp.stack((s, q, grp), axis=0)
    order = cp.lexsort(keys)
    grp_sorted = grp[order]
    q_sorted = q[order]
    s_sorted = s[order]
    sc_sorted = sc[order]
    cid_sorted = cid[order]
    sgn_sorted = sgn[order]

    # Find group boundaries where grp changes
    grp_shift = cp.concatenate([cp.asarray([-1], dtype=grp_sorted.dtype), grp_sorted[:-1]])
    new_group = grp_sorted != grp_shift
    grp_starts = cp.where(new_group)[0]
    grp_starts_h = grp_starts.get().tolist()
    grp_starts_h.append(int(grp_sorted.size))  # sentinel end

    chains: List[ChainCandidate] = []
    for gi in range(len(grp_starts_h) - 1):
        lo = grp_starts_h[gi]
        hi = grp_starts_h[gi + 1]
        if hi - lo <= 0:
            continue

        chrom_id = int(cid_sorted[lo].item())
        strand_sign = int(sgn_sorted[lo].item())

        # Already stable sorted (q then s) by global order
        q_seg = q_sorted[lo:hi]
        s_seg = s_sorted[lo:hi]
        sc_seg = sc_sorted[lo:hi]

        best, ord_idx = _dp_chain_single_gpu(
            q_seg, s_seg, sc_seg, strand_sign, chain_gap_lambda,
            max_q_gap=dp_max_q_gap, diag_band=dp_diag_band
        )
        if ord_idx.size == 0:
            continue
        if ord_idx.size < int(chain_min_hits):
            continue

        # Convert chosen hits to CPU Hit objects *for this chain only*
        abs_idx = (np.asarray(lo, dtype=np.int64) + ord_idx).astype(np.int64)
        q_cpu = cp.asnumpy(q_sorted[abs_idx])
        s_cpu = cp.asnumpy(s_sorted[abs_idx])
        sc_cpu = cp.asnumpy(sc_sorted[abs_idx])

        hits_list: List[Hit] = []
        chrom = gpuh.chrom_table[chrom_id]
        w = int(gpuh.window)
        for qq, ss, sco in zip(q_cpu, s_cpu, sc_cpu):
            hits_list.append(
                Hit(
                    q_start=int(qq),
                    q_end=int(qq) + w,
                    chrom=chrom,
                    s_start=int(ss),
                    s_end=int(ss) + w,
                    strand=1 if strand_sign >= 0 else -1,
                    score=float(sco),
                )
            )

        chains.append(
            ChainCandidate(
                chain_score=float(best),
                chrom=chrom,
                strand=(1 if strand_sign >= 0 else -1),
                hits=hits_list,
            )
        )

    # Rank and trim
    chains.sort(key=lambda c: c.chain_score, reverse=True)
    out_chains = chains[: int(take_top_chains)]

    # snapshot summary (small)
    if debug_snapshot is not None:
        summary = []
        for ch in out_chains[: min(5, len(out_chains))]:
            q_min = min(h.q_start for h in ch.hits)
            q_max = max(h.q_end for h in ch.hits)
            s_min = min(h.s_start for h in ch.hits)
            s_max = max(h.s_end for h in ch.hits)
            summary.append(
                {
                    "chrom": ch.chrom,
                    "strand": ch.strand,
                    "score": round(ch.chain_score, 2),
                    "n": len(ch.hits),
                    "q_span": [q_min, q_max],
                    "s_span": [s_min, s_max],
                }
            )
        debug_snapshot("chaining_gpu", {"n_groups": len(grp_starts_h) - 1, "n_chains": len(out_chains), "top": summary})

    return out_chains


# ===============================================================
# Chain + placement records
# ===============================================================

@dataclass(slots=True)
class ChainCandidate:
    chain_score: float
    chrom: str
    strand: int          # +1 / -1
    hits: List[Hit]      # ordered hits in this chain


@dataclass(slots=True)
class Placement:
    read_id: str
    chrom: str
    strand: str          # "+" or "-"
    start: int           # genomic start (0-based inclusive)
    end: int             # genomic end   (0-based exclusive-ish)
    aln_score: int       # fine aligner score (higher is better)
    chain_score: float   # coarse chain score
    n_chain_hits: int
    cigar: Optional[str]
    debug: Dict[str, object]


@dataclass
class QueryConfig:
    # chaining / DP
    chain_gap_lambda: float = 1.0
    chain_min_hits: int = 2
    take_top_chains: int = 4

    # ANN search / refinement
    top_k: int = 5
    flank: int = 2000
    ann_scores_higher_better: bool = True  # set False if your index returns distances

    # refinement selection
    refine_top_chains: int = 2
    competitive_gap: float = 50.0

    # WFA-GPU penalties / knobs
    wfa_x: int = 2         # mismatch penalty
    wfa_o: int = 3         # gap open
    wfa_e: int = 1         # gap extend
    wfa_batch_size: Optional[int] = None  # if set, passed through; else auto

    # Debugging / logging (DEBUG by default as requested)
    debug_level: int = logging.DEBUG
    debug_head_k: int = 5
    debug_max_print: int = 50
    debug_sink: Optional[Callable[[dict], None]] = None

    # --- new tuning knobs (defaults conservative) ---
    use_gpu_for_numpy_hits_threshold: int = 50_000  # promote NumPy D/I to CuPy if elements exceed this
    dp_max_q_gap: int = 50_000                      # predecessor window in query (bp)
    dp_diag_band: int = 20_000                      # |(ds - dq)| band (bp)


@dataclass(slots=True)
class _FineResult:
    score: int
    cigar: Optional[str]
    begin_query: int
    end_query: int
    begin_ref: int
    end_ref: int


# ===============================================================
# ReadAssembler
# ===============================================================

class ReadAssembler:
    """
    Pipeline:
      1. Tile read into overlapping windows (~10kb, stride ~3kb by default)
      2. Embed tiles using the same backend as the HEL indexer
      3. ANN search (RaftGPU / CAGRA) to get nearest reference windows
      4. DP chaining across tiles to get candidate placements (GPU-accelerated if possible)
      5. If chaining fails (short read), build pseudo-chains from ANN hits
      6. Fine-align (WFA-GPU) competing candidates, pick best by alignment score
      7. Return Placement (and debug info with all tried loci)
    """

    # --- constructor ---
    def __init__(
        self,
        hel_indexer,   # HELIndexer (already built/loaded)
        cfg: QueryConfig,
        window: int = 10_000,
        stride: Optional[int] = None,
    ):
        self.hel = hel_indexer
        self.cfg = cfg

        # match HELIndexer tiling
        eff_backend_max = int(getattr(self.hel.embedder, "max_length", window))
        self.window = int(min(int(window), eff_backend_max))

        # default stride ~1/3 window for overlap
        self.stride = int(max(1, (self.window // 3) if stride is None else int(stride)))

        # reference dictionary: chrom -> sequence
        self.ref_dict: Dict[str, str] = self.hel.ref_dict

        # meta cache (CPU + GPU arrays) to avoid rebuilding every read
        # (chrom_table, chrom_ids_np, s_starts_np, strands_np, chrom_ids_cp, s_starts_cp, strands_cp)
        self._meta_cache: Optional[Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, Optional["cp.ndarray"], Optional["cp.ndarray"], Optional["cp.ndarray"]]] = None

        # --- logging ---
        self._logger = logging.getLogger("ReadAssembler")
        if not self._logger.handlers:
            _h = logging.StreamHandler()
            _h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
            self._logger.addHandler(_h)
        self._logger.setLevel(getattr(self.cfg, 'debug_level', logging.DEBUG))

    # ---------- meta cache helper ----------

    def _get_meta_cache(self):
        if self._meta_cache is not None:
            return self._meta_cache
        chrom_table, chrom_ids_np, s_starts_np, strands_np = _ensure_meta_arrays(self.hel.index.metas)
        if _HAS_CUPY:
            chrom_ids_cp = cp.asarray(chrom_ids_np, dtype=cp.int32)
            s_starts_cp  = cp.asarray(s_starts_np, dtype=cp.int32)
            strands_cp   = cp.asarray(strands_np, dtype=cp.int8)
        else:
            chrom_ids_cp = s_starts_cp = strands_cp = None
        self._meta_cache = (chrom_table, chrom_ids_np, s_starts_np, strands_np,
                            chrom_ids_cp, s_starts_cp, strands_cp)
        return self._meta_cache

    # ---------- debug helpers ----------

    def _log(self, level: int, msg: str, **kw):
        if self._logger.isEnabledFor(level):
            extras = ", ".join(f"{k}={v}" for k, v in kw.items())
            self._logger.log(level, f"{msg}" + (" | " + extras if extras else ""))

    def _snapshot(self, name: str, **kw):
        # emit to logger
        self._log(logging.DEBUG, f"[SNAP] {name}", **kw)
        # emit to optional sink
        sink = getattr(self.cfg, 'debug_sink', None)
        if callable(sink):
            try:
                sink({"name": name, **kw})
            except Exception:
                pass

    # ---------- tiling / ANN ----------

    def _iter_read_tiles(self, read_seq: str) -> Tuple[List[int], List[str]]:
        """Break the read into overlapping tiles of length self.window stepping by self.stride.
        If len(read_seq) <= window you'll still only get one tile.
        """
        q_starts: List[int] = []
        q_tiles: List[str] = []

        L = len(read_seq)
        win = self.window
        st = self.stride

        if L == 0:
            self._snapshot("tiling", L=0, window=win, stride=st, n_tiles=0)
            return q_starts, q_tiles

        # sliding windows
        for i in range(0, max(L - win, 0) + 1, st):
            q_starts.append(i)
            q_tiles.append(read_seq[i:i + win])

        # ensure tail coverage
        last_start = max(L - win, 0)
        if not q_starts or last_start != q_starts[-1]:
            q_starts.append(last_start)
            q_tiles.append(read_seq[last_start:])

        self._snapshot("tiling", L=L, window=win, stride=st, n_tiles=len(q_starts), first_start=(q_starts[0] if q_starts else None), last_start=(q_starts[-1] if q_starts else None))
        return q_starts, q_tiles

    def _search_tiles(self, q_tiles: List[str]) -> Tuple[object, object]:
        """Embed tiles with the same model the index used, then ANN search.
        Accepts either return order: (D, I) or (I, D). Returns (D, I).
        D and I remain CuPy if the index returns CuPy; otherwise NumPy.
        D: float32 [nq, top_k], I: int64 [nq, top_k].
        """
        import torch

        # No tiles → empty, correctly shaped outputs
        if not q_tiles:
            zD = np.zeros((0, self.cfg.top_k), dtype=np.float32)
            zI = np.full((0, self.cfg.top_k), -1, dtype=np.int64)
            self._snapshot("embed", backend=type(self.hel.embedder).__name__, emb_shape=(0,))
            self._snapshot("ann_search", index=type(self.hel.index).__name__, D_shape=tuple(zD.shape),
                           I_shape=tuple(zI.shape))
            return zD, zI

        # 1) Embed (stay on device if possible)
        with torch.inference_mode():
            Q_emb = self.hel.embedder.embed_best(q_tiles)

        # torch -> CuPy via DLPack (only if available and useful)
        try:
            from torch.utils import dlpack as _dl
            if _HAS_CUPY and isinstance(Q_emb, torch.Tensor):
                if not Q_emb.is_cuda:
                    Q_emb = Q_emb.cuda(non_blocking=True)
                Q_emb = cp.from_dlpack(_dl.to_dlpack(Q_emb))  # keep on device
        except Exception:
            pass

        self._snapshot(
            "embed",
            backend=type(self.hel.embedder).__name__,
            emb_shape=tuple(getattr(Q_emb, 'shape', ())),
            emb_dtype=str(getattr(getattr(Q_emb, 'dtype', ''), 'name', ''))
        )

        # 2) ANN search
        out = self.hel.index.search(Q_emb, self.cfg.top_k)
        if not (isinstance(out, (list, tuple)) and len(out) == 2):
            raise RuntimeError("index.search must return a tuple/list of (D, I) or (I, D)")
        a, b = out[0], out[1]

        # 3) Normalize to (D: float32, I: int64) without crossing device unnecessarily
        is_cp = _HAS_CUPY and isinstance(a, cp.ndarray) and isinstance(b, cp.ndarray)  # type: ignore
        A = a if is_cp else np.asarray(a)
        B = b if is_cp else np.asarray(b)

        if np.issubdtype(A.dtype, np.integer) and np.issubdtype(B.dtype, np.floating):
            I = A.astype(np.int64, copy=False)
            D = B.astype(np.float32, copy=False)
        elif np.issubdtype(A.dtype, np.floating) and np.issubdtype(B.dtype, np.integer):
            D = A.astype(np.float32, copy=False)
            I = B.astype(np.int64, copy=False)
        else:
            D = A.astype(np.float32, copy=False)
            I = B.astype(np.int64, copy=False)

        if D.shape != I.shape:
            raise RuntimeError(f"index.search returned mismatched shapes: {D.shape} vs {I.shape}")

        # 4) Pad/trim to top_k using the same array library (prealloc, no hstack)
        lib = cp if is_cp else np
        nq = D.shape[0]
        head_k = min(D.shape[1], self.cfg.top_k)
        D2 = lib.full((nq, self.cfg.top_k), -lib.inf, dtype=D.dtype)
        I2 = lib.full((nq, self.cfg.top_k), -1, dtype=I.dtype)
        if head_k > 0:
            D2[:, :head_k] = D[:, :head_k]
            I2[:, :head_k] = I[:, :head_k]
        D, I = D2, I2

        # 5) Log small heads (unchanged logging)
        if is_cp:
            D_head = D[0, :min(self.cfg.debug_head_k, self.cfg.top_k)].get().tolist() if D.size else []
            I_head = I[0, :min(self.cfg.debug_head_k, self.cfg.top_k)].get().tolist() if I.size else []
        else:
            D_head = D[0, :min(self.cfg.debug_head_k, self.cfg.top_k)].tolist() if D.size else []
            I_head = I[0, :min(self.cfg.debug_head_k, self.cfg.top_k)].tolist() if I.size else []

        self._snapshot(
            "ann_search",
            index=type(self.hel.index).__name__,
            D_shape=tuple(D.shape), I_shape=tuple(I.shape),
            D_dtype=str(D.dtype), I_dtype=str(I.dtype), top_k=self.cfg.top_k,
            D_head=D_head, I_head=I_head,
            metas_len=len(getattr(self.hel.index, 'metas', [])),
        )

        # 6) Prune neighbors when top-1 is (approximately) 1.0 — AFTER logging
        if self.cfg.top_k > 1 and nq > 0:
            top1 = D[:, 0]
            if is_cp:
                mask = cp.isclose(top1, lib.float32(1.0), rtol=1e-6, atol=1e-6)
                if mask.any().item():
                    rows = cp.where(mask)[0]
                    D[rows, 1:] = -lib.inf
                    I[rows, 1:] = -1
            else:
                mask = np.isclose(top1, 1.0, rtol=1e-6, atol=1e-6)
                if mask.any():
                    D[mask, 1:] = -lib.inf
                    I[mask, 1:] = -1

        return D, I

    # ---------- chaining (CPU path) ----------

    def _chain_hits(self, hits: List[Hit]) -> List[ChainCandidate]:
        """CPU DP chaining for each (chrom,strand) bucket. Keep only chains with ≥ chain_min_hits."""
        groups = _group_hits_arrays(hits)
        chains: List[ChainCandidate] = []

        for (chrom, strand), q_arr, s_arr, sc_arr, idx_map in groups:
            if q_arr.size == 0:
                continue

            best_score, order = _dp_chain_single(
                q_arr, s_arr, sc_arr, int(strand), float(self.cfg.chain_gap_lambda),
                max_q_gap=int(self.cfg.dp_max_q_gap), diag_band=int(self.cfg.dp_diag_band)
            )
            if order.size == 0:
                continue
            chain_hits_list: List[Hit] = [hits[int(idx_map[o])] for o in order]

            if len(chain_hits_list) >= self.cfg.chain_min_hits:
                chains.append(
                    ChainCandidate(
                        chain_score=float(best_score),
                        chrom=str(chrom),
                        strand=int(strand),
                        hits=chain_hits_list,
                    )
                )

        chains.sort(key=lambda c: c.chain_score, reverse=True)
        out_chains = chains[: self.cfg.take_top_chains]
        # snapshot summary
        summary = []
        for ch in out_chains[:min(5, len(out_chains))]:
            q_min = min(h.q_start for h in ch.hits)
            q_max = max(h.q_end for h in ch.hits)
            s_min = min(h.s_start for h in ch.hits)
            s_max = max(h.s_end for h in ch.hits)
            summary.append({"chrom": ch.chrom, "strand": ch.strand, "score": round(ch.chain_score, 2), "n": len(ch.hits), "q_span": [q_min, q_max], "s_span": [s_min, s_max]})
        self._snapshot("chaining_cpu", n_groups=len(groups), n_chains=len(out_chains), top=summary)
        return out_chains

    def _fallback_chains_from_hits(self, hits: List[Hit]) -> List[ChainCandidate]:
        """If chaining failed (e.g. short read ⇒ only 1 tile), take top ANN hits and
        turn them into pseudo-chains. Cluster nearby windows on the same chrom/strand.
        """
        if not hits:
            return []

        sorted_hits = sorted(hits, key=lambda h: h.score, reverse=True)
        cluster_radius = max(self.window // 2, self.stride * 2)

        reps: List[Hit] = []
        for h in sorted_hits:
            placed = False
            for rep in reps:
                if (
                    rep.chrom == h.chrom
                    and rep.strand == h.strand
                    and abs(rep.s_start - h.s_start) <= cluster_radius
                ):
                    placed = True
                    break
            if not placed:
                reps.append(h)
            if len(reps) >= self.cfg.take_top_chains:
                break

        chains: List[ChainCandidate] = [
            ChainCandidate(chain_score=float(h.score), chrom=h.chrom, strand=h.strand, hits=[h])
            for h in reps
        ]
        self._snapshot("fallback_chains", n_hits=len(hits), n_reps=len(reps),
                       reps=[{"chrom": r.chrom, "strand": r.strand,
                              "s_start": r.s_start, "score": round(r.score, 2)} for r in reps[:min(5, len(reps))]])
        return chains

    # ---------- refinement prep ----------

    def _extract_ref_window(self, chrom: str, start_bp: int, end_bp: int) -> str:
        """Slice reference [start_bp:end_bp] safely."""
        ref_seq = self.ref_dict[chrom]
        start_bp = max(0, int(start_bp))
        end_bp = min(len(ref_seq), int(end_bp))
        if start_bp >= end_bp:
            return ""
        return ref_seq[start_bp:end_bp]

    def _pick_chains_for_refinement(self, chains: List[ChainCandidate]) -> List[ChainCandidate]:
        """Decide which chain candidates get passed to fine alignment."""
        if not chains:
            return []
        if len(chains) == 1:
            return chains[:1]

        best, second = chains[0], chains[1]
        gap = best.chain_score - second.chain_score
        return chains[: self.cfg.refine_top_chains] if gap < self.cfg.competitive_gap else chains[:1]

    def _prepare_refinement_batch(self, qseq: str, chains_for_refine: List[ChainCandidate]) -> Tuple[List[str], List[str], List[Dict]]:
        """For each chosen candidate chain:
          - compute coarse genomic span from min..max hit
          - add flank
          - reverse-complement if chain.strand == -1
          - create Q (query read) and S (target slice) for GPU batch
        Returns (Qs, Ss, meta)
        """
        Qs: List[str] = []
        Ss: List[str] = []
        meta: List[Dict] = []

        for rank, ch in enumerate(chains_for_refine):
            chrom = ch.chrom
            strand = int(ch.strand)
            s_start = min(h.s_start for h in ch.hits)
            s_end = max(h.s_end for h in ch.hits)

            s_lo = max(0, s_start - self.cfg.flank)
            s_hi = s_end + self.cfg.flank

            ref_sub = self._extract_ref_window(chrom, s_lo, s_hi)
            tgt_seq = revcomp(ref_sub) if strand == -1 else ref_sub

            Qs.append(qseq)
            Ss.append(tgt_seq)
            meta.append(
                {
                    "rank": rank,
                    "chrom": chrom,
                    "strand_sign": strand,
                    "s_lo": int(s_lo),
                    "s_hi": int(s_hi),
                    "chain_score": float(ch.chain_score),
                    "n_hits": len(ch.hits),
                    "ref_len": len(ref_sub),
                }
            )

        self._snapshot("refine_prep", n=len(meta), items=[{k: (v if k in ("chrom", "strand_sign", "n_hits") else int(v)) for k, v in m.items()} for m in meta[:min(5, len(meta))]])
        return Qs, Ss, meta

    @staticmethod
    def _revcomp_coord_transform(ref_start_bp: int, ref_end_bp: int, aln_begin_ref: int, aln_end_ref: int) -> Tuple[int, int]:
        """Convert alignment coords from reverse-complement slice back to forward-genome coords."""
        L = int(ref_end_bp) - int(ref_start_bp)
        start_genome = int(ref_start_bp) + (L - int(aln_end_ref))
        end_genome = int(ref_start_bp) + (L - int(aln_begin_ref))
        return int(start_genome), int(end_genome)

    # ---------- WFA-GPU adapter ----------

    def _wfa_gpu_align_batch(self, Qs: List[str], Ss: List[str]) -> List[_FineResult]:
        """WFA-GPU adapter that returns objects with (score, cigar, begin/end for query & ref).
        Uses affine penalties from self.cfg (wfa_x, wfa_o, wfa_e). Score is *negative distance*.
        """
        try:
            from wfa_gpu.wfagpu import wfa_gpu as WFA_GPU  # your Python wrapper
        except Exception as e:
            raise ImportError(
                "Failed to import WFA-GPU Python wrapper 'wfa_gpu.wfagpu'. "
                "Ensure the CAPI .so is importable and the package is installed."
            ) from e

        x = int(getattr(self.cfg, "wfa_x", 2))
        o = int(getattr(self.cfg, "wfa_o", 3))
        e = int(getattr(self.cfg, "wfa_e", 1))

        results: List[Dict] = []
        # Prefer a true batched entrypoint if present
        if hasattr(WFA_GPU, "align_batch"):
            bsz = int(getattr(self.cfg, "wfa_batch_size", 0) or max(1, len(Qs)))
            results = WFA_GPU.align_batch(Qs, Ss, x=x, o=o, e=e, compute_cigar=True, batch_size=bsz)
            if not isinstance(results, list):
                results = [results]
        else:
            # Not batched: avoid oversized per-call batch
            bsz = 1
            for q, s in zip(Qs, Ss):
                results.append(WFA_GPU.align(q, s, x=x, o=o, e=e, compute_cigar=True, batch_size=bsz))

        out: List[_FineResult] = []
        for q, s, r in zip(Qs, Ss, results):
            cigar = r.get("cigar")
            distance = affine_distance_from_cigar(cigar, x, o, e)
            score_for_ranking = -int(distance)

            bq, eq, br, er = span_from_cigar(cigar)
            if eq <= bq:  # guard
                eq = len(q)
            if er <= br:
                er = len(s)

            out.append(
                _FineResult(
                    score=score_for_ranking,
                    cigar=cigar,
                    begin_query=int(bq),
                    end_query=int(eq),
                    begin_ref=int(br),
                    end_ref=int(er),
                )
            )

        self._snapshot(
            "wfa_align",
            n=len(out),
            first={
                "score": (out[0].score if out else None),
                "cigar_len": (len(out[0].cigar) if out and out[0].cigar else 0),
                "begin_ref": (out[0].begin_ref if out else None),
                "end_ref": (out[0].end_ref if out else None),
            },
        )
        return out

    # ---------- refinement + ranking ----------

    def _rank_and_choose_best(self, cands: List[Placement]) -> Placement:
        """Rank fine-aligned candidates and return the best.
        Sort priority: aln_score desc, chain_score desc, n_chain_hits desc, genomic start asc.
        Also attaches ALL candidates in best.debug["all_candidates"].
        """
        if not cands:
            raise RuntimeError("No candidates to rank.")

        def sort_key(p: Placement):
            return (-p.aln_score, -p.chain_score, -p.n_chain_hits, p.start)

        cands_sorted = sorted(cands, key=sort_key)
        best = cands_sorted[0]

        best.debug["all_candidates"] = [
            {
                "chrom": p.chrom,
                "strand": p.strand,
                "start": p.start,
                "end": p.end,
                "aln_score": p.aln_score,
                "chain_score": p.chain_score,
                "n_chain_hits": p.n_chain_hits,
            }
            for p in cands_sorted
        ]
        self._snapshot("rank", chosen={"chrom": best.chrom, "strand": best.strand,
                                       "start": best.start, "end": best.end,
                                       "aln_score": best.aln_score,
                                       "chain_score": round(best.chain_score, 2),
                                       "n_hits": best.n_chain_hits})
        return best

    def _choose_best_alignment(self, qseq: str, chains: List[ChainCandidate]) -> Optional[Placement]:
        """Refine best/ambiguous chains with GPU alignment and choose the winner."""
        if not chains:
            return None

        refine_list = self._pick_chains_for_refinement(chains)
        if not refine_list:
            return None

        Qs, Ss, meta = self._prepare_refinement_batch(qseq, refine_list)
        gpu_results = self._wfa_gpu_align_batch(Qs, Ss)
        assert len(gpu_results) == len(meta)

        refined_candidates: List[Placement] = []
        for m, r in zip(meta, gpu_results):
            chrom = m["chrom"]
            strand_sign = int(m["strand_sign"])
            s_lo = int(m["s_lo"])
            s_hi = int(m["s_hi"])

            if strand_sign == +1:
                g_start = s_lo + int(r.begin_ref)
                g_end = s_lo + int(r.end_ref)
                strand_symbol = "+"
            else:
                g_start, g_end = self._revcomp_coord_transform(s_lo, s_hi, int(r.begin_ref), int(r.end_ref))
                strand_symbol = "-"

            refined_candidates.append(
                Placement(
                    read_id="",  # filled in by caller
                    chrom=str(chrom),
                    strand=strand_symbol,
                    start=min(g_start, g_end),
                    end=max(g_start, g_end),
                    aln_score=int(r.score),
                    chain_score=float(m["chain_score"]),
                    n_chain_hits=int(m["n_hits"]),
                    cigar=r.cigar,
                    debug={
                        "begin_query": int(r.begin_query),
                        "end_query": int(r.end_query),
                        "begin_ref_local": int(r.begin_ref),
                        "end_ref_local": int(r.end_ref),
                        "coarse_start": s_lo,
                        "coarse_end": s_hi,
                    },
                )
            )

        return self._rank_and_choose_best(refined_candidates)

    # ---------- public API ----------

    def place_one_read(self, read_seq: str, read_id: str) -> Optional[Placement]:
        """Full mapping pipeline for one read.
        Returns best Placement or None if nothing plausible was found.
        """
        # tile + ANN
        q_starts, q_tiles = self._iter_read_tiles(read_seq)
        D, I = self._search_tiles(q_tiles)

        # Promote to GPU if NumPy and large enough (to use GPU pipeline end-to-end)
        if _HAS_CUPY and isinstance(D, np.ndarray) and isinstance(I, np.ndarray):
            if int(D.size) > int(getattr(self.cfg, "use_gpu_for_numpy_hits_threshold", 50_000)):
                D = cp.asarray(D)
                I = cp.asarray(I)

        # Try GPU pipeline end-to-end if D/I are on device
        is_cp = _HAS_CUPY and isinstance(D, cp.ndarray) and isinstance(I, cp.ndarray)  # type: ignore
        chains: List[ChainCandidate] = []
        gpuh: Optional[_GPUHits] = None

        if is_cp:
            # GPU gather + chaining
            try:
                chrom_table, chrom_ids_np, s_starts_np, strands_np, chrom_ids_cp, s_starts_cp, strands_cp = self._get_meta_cache()
                meta_cache_gpu = (chrom_table, chrom_ids_cp, s_starts_cp, strands_cp)  # type: ignore[arg-type]
                gpuh = _gather_hits_arrays_gpu(
                    q_starts, self.window, D, I, self.hel.index.metas,
                    higher_better=self.cfg.ann_scores_higher_better,
                    meta_cache_gpu=meta_cache_gpu
                )
                chains = _chain_from_gpu_hits(
                    gpuh,
                    chain_gap_lambda=float(self.cfg.chain_gap_lambda),
                    chain_min_hits=int(self.cfg.chain_min_hits),
                    take_top_chains=int(self.cfg.take_top_chains),
                    dp_max_q_gap=int(self.cfg.dp_max_q_gap),
                    dp_diag_band=int(self.cfg.dp_diag_band),
                    debug_snapshot=lambda n, d: self._snapshot(n, **d),
                )
            except Exception as e:
                self._snapshot("gpu_chain_fallback", error=str(e))

        # If GPU path not taken or produced no chains, fallback to CPU path
        if not chains:
            hits = _gather_hits(
                q_starts, self.window, D, I, self.hel.index.metas,
                higher_better=self.cfg.ann_scores_higher_better,
            )
            # snapshot hits summary
            if hits:
                from collections import Counter
                bucket = Counter(f"{h.chrom}:{h.strand}" for h in hits)
                self._snapshot(
                    "hits",
                    n_hits=len(hits),
                    score_min=round(min(h.score for h in hits), 4),
                    score_max=round(max(h.score for h in hits), 4),
                    buckets=dict(bucket),
                    sample=[{"chrom": hits[0].chrom, "strand": hits[0].strand, "q_start": hits[0].q_start, "s_start": hits[0].s_start, "score": round(hits[0].score, 3)}]
                )
            if not hits:
                return None
            chains = self._chain_hits(hits)

        # fallback if chaining produced no multi-hit chains
        if not chains:
            # If we came via GPU path, reconstruct a few top hits for fallback
            if is_cp and gpuh is not None:
                # Create small CPU Hit list from top scores (vectorized copies)
                flat_scores = gpuh.score
                top_n = int(min(self.cfg.take_top_chains * 4, int(flat_scores.size)))
                if top_n > 0:
                    top_idx = cp.asnumpy(cp.argsort(-flat_scores)[:top_n])
                    sel = cp.asarray(top_idx)
                    qq  = cp.asnumpy(gpuh.q_start[sel])
                    ss  = cp.asnumpy(gpuh.s_start[sel])
                    cid = cp.asnumpy(gpuh.chrom_id[sel])
                    sgn = cp.asnumpy(gpuh.strand_sign[sel])
                    sco = cp.asnumpy(gpuh.score[sel])
                    hits_small: List[Hit] = []
                    chrom_table = gpuh.chrom_table
                    w = gpuh.window
                    for qv, sv, cv, gv, dv in zip(qq, ss, cid, sgn, sco):
                        hits_small.append(Hit(int(qv), int(qv) + w, chrom_table[int(cv)], int(sv), int(sv) + w, (1 if int(gv) >= 0 else -1), float(dv)))
                    chains = self._fallback_chains_from_hits(hits_small)
            if not chains:
                return None

        # fine alignment refinement + ranking
        placement = self._choose_best_alignment(read_seq, chains)
        if placement is None:
            return None

        placement.read_id = read_id
        return placement

    def assemble(self, reads: List[str]) -> List[Placement]:
        """Map a batch of reads and return their best placements."""
        out: List[Placement] = []
        for ridx, read_seq in enumerate(reads):
            rid = f"read_{ridx}"
            p = self.place_one_read(read_seq, rid)
            if p is not None:
                out.append(p)
        self._snapshot("assemble_done", n_reads=len(reads), n_placed=len(out))
        return out


# ===============================================================
# Example wiring (kept minimal; adapt to your environment)
# ===============================================================
if __name__ == "__main__":
    from pathlib import Path
    import pysam
    from src.old.old_HELIndexer import HELIndexer
    from src.configs import IndexConfig

    WINDOW = 12_000
    N_TILES = 10
    TARGET_LEN = WINDOW * N_TILES


    work = Path("/g/data/te53/en9803/sandpit/graph_genomics/chr22")
    work.mkdir(parents=True, exist_ok=True)
    ref_fa = work / "chm13v2_chr22.fa.gz"          # this should already exist

    fasta = pysam.FastaFile(str(ref_fa))
    if "chr22" in fasta.references:
        seq = fasta.fetch("chr22")
        out_header = ">chrA\n"
        fasta.close()
        chrom22_seq_clean = seq.upper().replace("N", "")[:TARGET_LEN]
        if not chrom22_seq_clean:
            raise RuntimeError("chr22 sequence is empty after N removal. Check input FASTA?")
    """
        # Write tiny.fa in 60-column FASTA lines
        tiny_fa = work / "tiny.fa"
        gz = tiny_fa.with_suffix(tiny_fa.suffix + ".gz")
        with tiny_fa.open("w") as out:
            out.write(out_header)
            for i in range(0, WINDOW*N_TILES, 60):
                out.write(chrom22_seq_clean[i:i + 60] + "\n")
            # BGZF-compress (creates chm13v2_chr22.fa.gz)
            pysam.tabix_compress(str(tiny_fa), str(gz), force=True)
            pysam.faidx(str(gz))
            fa = pysam.FastaFile(str(gz))
            try:
                print("refs:", fa.references[:3], "…")
                print("chrA length:", fa.get_reference_length("chrA"))
            finally:
                fa.close()
        print(f"wrote {tiny_fa} with {len(chrom22_seq_clean)} bp from chr22 (no Ns); source={ref_fa.name}")

    """

    tiny_fa = Path("/g/data/te53/en9803/sandpit/graph_genomics/chr22/tiny.fa")

    cfg_index = IndexConfig(window=WINDOW, stride=6000, rc_index=True)
    hel = HELIndexer(tiny_fa, cfg_index, emb_batch=256)
    hel.build_or_load(tiny_fa.parent / "index", reuse_existing=False, include_dataset=True, verbose=True)

    # DEBUG is ON by default in QueryConfig; you can still override here
    cfg_query = QueryConfig(
        chain_gap_lambda=1.0,
        chain_min_hits=2,
        take_top_chains=2,
        top_k=5,
        flank=2000,
        refine_top_chains=5,
        competitive_gap=50.0,
        wfa_x=2,
        wfa_o=3,
        wfa_e=1,
        wfa_batch_size=None,
        ann_scores_higher_better=True,
        debug_level=logging.DEBUG,

        # new knobs (tune if desired)
        use_gpu_for_numpy_hits_threshold=50_000,
        dp_max_q_gap=5 * WINDOW,       # ~5 windows
        dp_diag_band=2 * WINDOW,       # ~2 windows
    )

    assembler = ReadAssembler(hel_indexer=hel, cfg=cfg_query, window=WINDOW, stride=None)

    reads = [revcomp(chrom22_seq_clean[i:i+WINDOW]) for i in range(0, WINDOW*3, WINDOW)]
    placements = assembler.assemble(reads)
    for p in placements:
        print(p)
