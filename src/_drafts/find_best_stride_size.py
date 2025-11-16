#!/usr/bin/env python3
from __future__ import annotations
import os
import gc
import re
import time
import json
import math
import csv
import random
import logging
import multiprocessing as mp
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from contextlib import suppress, contextmanager

import pysam

# Project imports
from src.HELIndexer import HELIndexer
from src.configs import IndexConfig
from src.ReadAssembler import ReadAssembler, QueryConfig, revcomp
from src.HyenaBackend import HyenaBackend

# Optional GPU libs
try:
    import torch
    _HAVE_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAVE_TORCH = False

try:
    import cupy as cp  # type: ignore
    _HAVE_CUPY = True
except Exception:
    cp = None  # type: ignore
    _HAVE_CUPY = False

try:
    import rmm  # type: ignore
    _HAVE_RMM = True
except Exception:
    rmm = None  # type: ignore
    _HAVE_RMM = False


# ============================================================
# Config
# ============================================================

WINDOW = 10_000
# 1,000..10,000 inclusive, now stepping by 500
STRIDES = list(range(500, 10001, 500))

READS_PER_STRIDE = 300
# 20% perfect, 80% random
PCT_PERFECT = 0.20
SEED = 1337

N_TILES = 120                      # 1.2 Mbp slice (fast/deterministic)
TARGET_LEN = N_TILES * WINDOW

FASTA_PATH = Path(
    os.environ.get(
        "HEL_FASTA",
        "/g/data/te53/en9803/sandpit/graph_genomics/chr22/chm13v2_chr22.fa.gz",
    )
).expanduser()

WORK_ROOT = Path(
    os.environ.get(
        "HEL_WORK",
        "/g/data/te53/en9803/sandpit/graph_genomics/chr22",
    )
).expanduser()

INDEX_CACHE_ROOT = WORK_ROOT / "hel_cache_sweep"
INDEX_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = INDEX_CACHE_ROOT / "sweep_results.csv"
RESULTS_JSONL = INDEX_CACHE_ROOT / "sweep_results.jsonl"

# Reduce thread oversubscription noise
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


# ============================================================
# Memory & resource management
# ============================================================

def _free_cuda_sync():
    """Synchronize CUDA streams to avoid freeing in-flight buffers."""
    with suppress(Exception):
        if _HAVE_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
    with suppress(Exception):
        if _HAVE_CUPY:
            cp.cuda.runtime.deviceSynchronize()  # type: ignore[attr-defined]

def _free_cuda_heaps():
    """Aggressively return GPU memory to the OS where possible."""
    # CuPy memory pools
    if _HAVE_CUPY:
        with suppress(Exception):
            cp.get_default_memory_pool().free_all_blocks()
        with suppress(Exception):
            cp.get_default_pinned_memory_pool().free_all_blocks()
    # RAPIDS RMM (if used by RAFT/cuVS)
    if _HAVE_RMM:
        with suppress(Exception):
            mr = rmm.mr.get_current_device_resource()
            if hasattr(mr, "flush"):
                mr.flush()  # type: ignore
            if hasattr(mr, "release"):
                mr.release()  # type: ignore
    # PyTorch caches (and shared CUDA IPC arenas)
    if _HAVE_TORCH and torch.cuda.is_available():
        with suppress(Exception):
            torch.cuda.empty_cache()
        with suppress(Exception):
            torch.cuda.ipc_collect()  # type: ignore[attr-defined]

def _teardown_cuda_and_cpu():
    _free_cuda_sync()
    _free_cuda_heaps()
    for _ in range(2):
        gc.collect()
        time.sleep(0.02)

def _close_if_possible(obj):
    with suppress(Exception):
        if obj is not None and hasattr(obj, "close"):
            obj.close()

@contextmanager
def closing_hel(hel: HELIndexer):
    try:
        yield hel
    finally:
        _close_if_possible(hel)
        del hel
        _teardown_cuda_and_cpu()

@contextmanager
def closing_assembler(asm: ReadAssembler):
    try:
        yield asm
    finally:
        _close_if_possible(asm)
        del asm
        _teardown_cuda_and_cpu()


# ============================================================
# Helpers
# ============================================================

def _cfg_signature(window: int, stride: int, metric: str, rc_index: bool, fasta_path: Path) -> str:
    fasta_tag = Path(fasta_path).stem
    rc_tag = "rc1" if rc_index else "rc0"
    return f"{fasta_tag}_w{window}_s{stride}_{metric}_{rc_tag}"

def _load_chr22_slice(fa_path: Path, target_len: int) -> str:
    assert fa_path.exists(), f"FASTA not found: {fa_path}"
    with pysam.FastaFile(str(fa_path)) as fa:
        chrom = "chr22" if "chr22" in fa.references else "22"
        seq = fa.fetch(chrom)
    seq = seq.upper().replace("N", "")
    if len(seq) < target_len:
        raise RuntimeError(f"{fa_path} {chrom} length {len(seq)} < {target_len}")
    return seq[:target_len]

def _rng(seed: Optional[int]) -> random.Random:
    return random.Random(seed) if seed is not None else random

def make_reads_mixed(
    genome: str,
    window: int,
    n_reads: int,
    pct_perfect: float,
    seed: Optional[int],
    rc_fraction_random: float = 0.5,
) -> Tuple[List[str], List[int], List[int], List[bool]]:
    """
    Generate reads of length = window.
      - 'pct_perfect' perfect reads start at exact window multiples: 0, 10k, 20k, ...
      - Remaining are random starts anywhere valid.
      - Perfect reads: strand '+'; Random reads: rc_fraction_random proportion RC.
    Returns:
      reads, true_starts, true_strands(+1/-1), is_perfect (bool list)
    """
    assert 0 <= pct_perfect <= 1
    R = _rng(seed)
    L = len(genome)
    max_start = L - window
    if max_start <= 0:
        raise ValueError("Genome too short for requested window")

    n_perfect = int(round(n_reads * pct_perfect))
    n_random = n_reads - n_perfect

    max_bin = max_start // window
    perfect_bins = list(range(min(n_perfect, max_bin + 1)))
    perfect_starts = [b * window for b in perfect_bins]
    while len(perfect_starts) < n_perfect and max_bin >= 0:
        perfect_starts.append((len(perfect_starts) % (max_bin + 1)) * window)

    random_starts = [R.randrange(0, max_start + 1) for _ in range(n_random)]

    starts = perfect_starts + random_starts
    is_perfect = [True] * len(perfect_starts) + [False] * len(random_starts)

    reads: List[str] = []
    strands: List[int] = []

    for i, s in enumerate(starts):
        frag = genome[s:s + window]
        if is_perfect[i]:
            reads.append(frag)
            strands.append(+1)
        else:
            if R.random() < rc_fraction_random:
                reads.append(revcomp(frag))
                strands.append(-1)
            else:
                reads.append(frag)
                strands.append(+1)

    return reads, starts, strands, is_perfect

def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))

def choose_bin_start_for_read(start: int, read_len: int, genome_len: int, window: int, stride: int) -> Tuple[int, int]:
    """Map [start,start+len) to stride-grid bin [b,b+window) maximizing overlap."""
    read0, read1 = start, start + read_len
    base = (start // stride) * stride
    candidates = []
    for off in (-2, -1, 0, 1, 2):
        b0 = base + off * stride
        if 0 <= b0 <= genome_len - window:
            o = _overlap(read0, read1, b0, b0 + window)
            center_diff = abs((read0 + read1)/2 - (b0 + (b0 + window))/2)
            candidates.append((o, -center_diff, b0))
    if not candidates:
        b0 = min(max(0, base), max(0, genome_len - window))
        return b0, b0 // stride
    b0 = max(candidates)[2]
    return b0, b0 // stride

def bin_from_start(start: Optional[int], genome_len: int, window: int, stride: int) -> Optional[Tuple[int, int]]:
    if start is None or start < 0:
        return None
    b0, bid = choose_bin_start_for_read(start, window, genome_len, window, stride)
    return (b0, bid)

# -------- placement extractors --------
_CIGAR_TOK_RE = re.compile(r"(\d+)([MIDNSHP=XBmidnshp=xb])")

def cigar_counts(cigar: Optional[str]) -> Dict[str, int]:
    counts = defaultdict(int)
    if not cigar:
        return counts
    for n, op in _CIGAR_TOK_RE.findall(cigar):
        counts[op.upper()] += int(n)
    return counts

def _read_attr(obj: Any, name: str, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default

def extract_anchor_start(p: Any) -> Optional[int]:
    if p is None:
        return None
    dbg = _read_attr(p, "debug")
    if isinstance(dbg, dict):
        v = dbg.get("coarse_start")
        if isinstance(v, int):
            return v
        v = dbg.get("begin_ref_local")
        if isinstance(v, int):
            return v
    for k in ("anchor_start", "pos", "start", "ref_pos"):
        v = _read_attr(p, k)
        if isinstance(v, int):
            return v
    return None

def extract_refined_start(p: Any) -> Optional[int]:
    if p is None:
        return None
    dbg = _read_attr(p, "debug")
    if isinstance(dbg, dict):
        for k in ("wfa_ref_start", "aln_ref_start", "dp_ref_start", "refined_start", "ref_start", "ref_begin"):
            v = dbg.get(k)
            if isinstance(v, int):
                return v
    v = _read_attr(p, "start")
    return int(v) if isinstance(v, int) else None

def extract_anchor_score(p: Any) -> Optional[float]:
    if p is None:
        return None
    for k in ("chain_score", "ann_score", "anchor_score"):
        v = _read_attr(p, k)
        if isinstance(v, (int, float)):
            return float(v)
    dbg = _read_attr(p, "debug")
    if isinstance(dbg, dict):
        cands = dbg.get("all_candidates")
        if isinstance(cands, list) and cands:
            v = cands[0].get("chain_score")
            if isinstance(v, (int, float)):
                return float(v)
    return None

def extract_alignment_score(p: Any) -> Optional[float]:
    if p is None:
        return None
    for k in ("aln_score", "wfa_score", "dp_score"):
        v = _read_attr(p, k)
        if isinstance(v, (int, float)):
            return float(v)
    dbg = _read_attr(p, "debug")
    if isinstance(dbg, dict):
        cands = dbg.get("all_candidates")
        if isinstance(cands, list) and cands:
            v = cands[0].get("aln_score")
            if isinstance(v, (int, float)):
                return float(v)
    return None

def extract_cigar(p: Any) -> Optional[str]:
    if p is None:
        return None
    cig = _read_attr(p, "cigar")
    if isinstance(cig, str) and cig:
        return cig
    dbg = _read_attr(p, "debug")
    if isinstance(dbg, dict):
        cands = dbg.get("all_candidates")
        if isinstance(cands, list) and cands:
            cig = cands[0].get("cigar")
            if isinstance(cig, str) and cig:
                return cig
    return None

# -------- assembler config (enable refinement to get CIGAR) --------
def query_cfg_with_refine(window: int) -> QueryConfig:
    return QueryConfig(
        top_k=64,
        take_top_chains=8,
        chain_min_hits=1,
        chain_gap_lambda=0.8,
        competitive_gap=20.0,
        flank=500,
        refine_top_chains=4,
        wfa_x=2, wfa_o=4, wfa_e=1,
        wfa_batch_size=64,
        dp_max_q_gap=1500,
        dp_diag_band=512,
        ann_scores_higher_better=True,
        debug_level=logging.WARNING,
        use_gpu_for_numpy_hits_threshold=50_000,
    )


# ============================================================
# Index build/load + size
# ============================================================

def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            with suppress(Exception):
                total += p.stat().st_size
    return total

def _manifest_rows(manifest_path: Path) -> Optional[int]:
    try:
        with open(manifest_path, "r") as f:
            m = json.load(f)
        # Try a few common keys
        for k in ("n_vectors", "n_rows", "n_windows", "n_items"):
            if isinstance(m.get(k), int):
                return int(m[k])
        # Sometimes nested
        ds = m.get("dataset") or m.get("meta") or {}
        for k in ("n", "rows", "windows"):
            if isinstance(ds.get(k), int):
                return int(ds[k])
    except Exception:
        pass
    return None

def ensure_index_cached(fasta_path: Path | str, window: int, stride: int, *, rc_index: bool = True) -> Dict[str, Any]:
    """Build-or-load the index and return metrics, without keeping state alive."""
    fasta_path = Path(fasta_path)
    cfg_index = IndexConfig(window=window, stride=stride, rc_index=rc_index)
    metric_name = getattr(cfg_index, "metric", "cosine")
    sig = _cfg_signature(window, stride, metric_name, cfg_index.rc_index, fasta_path)
    index_dir = INDEX_CACHE_ROOT / sig
    index_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = index_dir / "manifest.json"
    existed_before = manifest_path.exists()

    t0 = time.perf_counter()
    hyena_last = HyenaBackend(
        model_name="hyenadna-large-1m-seqlen-hf",
        model_dir="/g/data/te53/en9803/data/scratch/hf-cache/models",
        pooling="mean",
        normalize=False,
        offline=True,
        prefer_cuda=True,
    )
    hel = HELIndexer(fasta_path, cfg_index, embedder=hyena_last, emb_batch=512)
    with closing_hel(hel) as h:
        h.build_or_load(index_dir, reuse_existing=True, verbose=False)
    build_or_load_time = time.perf_counter() - t0

    built_now = (not existed_before) and manifest_path.exists()
    size_bytes = _dir_size_bytes(index_dir)
    n_rows = _manifest_rows(manifest_path)

    print(f"[INDEX] ready: {index_dir} | time={build_or_load_time:.3f}s | size={size_bytes/1e6:.2f}MB | rows={n_rows}")

    return {
        "stride": stride,
        "index_dir": str(index_dir),
        "index_build_or_load_time_sec": build_or_load_time,
        "index_size_bytes": size_bytes,
        "index_rows": n_rows,
        "index_built_now": built_now,
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
    }


# ============================================================
# Evaluation
# ============================================================

@dataclass
class StrideReport:
    stride: int
    n_reads: int
    n_perfect: int
    n_random: int

    # anchor (index/bin) stats - overall
    bin_correct: int
    bin_adjacent: int
    bin_wrong: int

    # refined alignment stats - overall
    refined_correct: int
    refined_adjacent: int
    refined_wrong: int

    # diagnostics
    rescued_by_alignment: int
    anchor_ok_align_poor: int
    similar_region_wrong_bin: int

    # per-class (perfect)
    bin_correct_perfect: int
    bin_adjacent_perfect: int
    bin_wrong_perfect: int
    refined_correct_perfect: int
    refined_adjacent_perfect: int
    refined_wrong_perfect: int

    # per-class (random)
    bin_correct_random: int
    bin_adjacent_random: int
    bin_wrong_random: int
    refined_correct_random: int
    refined_adjacent_random: int
    refined_wrong_random: int

    # score tracking
    anchor_scores: List[float]
    aln_scores: List[float]

    # timing for querying
    query_time_sec_total: float
    query_time_ms_per_read: float

def _evaluate_stride_worker(args) -> Dict[str, Any]:
    """Runs in a fresh subprocess (spawn) to isolate native/GPU state."""
    (fasta_path_str, stride, window, reads_per_stride, pct_perfect, seed, target_len) = args

    # Deterministic seeds per worker
    random.seed(seed + stride)
    if _HAVE_TORCH:
        with suppress(Exception):
            torch.manual_seed(seed + stride)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + stride)

    fasta_path = Path(fasta_path_str)

    # Build/load index in this process (makes sure RAFT state lives/dies here)
    _ = ensure_index_cached(fasta_path, window, stride, rc_index=True)

    # Load genome slice locally in worker to avoid shipping large objects through pickle
    genome = _load_chr22_slice(fasta_path, target_len)
    genome_len = len(genome)

    cfg = query_cfg_with_refine(window)
    hyena_last = HyenaBackend(
        model_name="hyenadna-large-1m-seqlen-hf",  # or your specific HF name if you set one elsewhere
        model_dir="/g/data/te53/en9803/data/scratch/hf-cache/models",
        pooling="mean",  # <--- key change
        normalize=False,  # HELIndexer handles cosine normalization
        offline=True,
        prefer_cuda=True,
    )
    hel = HELIndexer(
        fasta_path,
        IndexConfig(window=window, stride=stride, rc_index=True),
        embedder=hyena_last,  # <--- pass the object, not the string
        emb_batch=512,
    )

    # Assemble reads and time the query path
    with closing_hel(hel) as h:
        h.build_or_load(INDEX_CACHE_ROOT / _cfg_signature(window, stride, getattr(cfg, "metric", "cosine"), True, fasta_path), reuse_existing=True, verbose=False)
        asm = ReadAssembler(hel_indexer=h, cfg=cfg, window=window, stride=None)
        with closing_assembler(asm):
            reads, true_starts, true_strands, is_perfect = make_reads_mixed(
                genome, window, n_reads=reads_per_stride, pct_perfect=pct_perfect, seed=seed, rc_fraction_random=0.5
            )
            t0 = time.perf_counter()
            placements = asm.assemble(reads)
            t1 = time.perf_counter()
            if len(placements) < len(reads):
                placements = list(placements) + [None] * (len(reads) - len(placements))
            query_time_total = t1 - t0
            query_time_ms_per_read = (query_time_total / max(1, len(reads))) * 1e3

    # At this point HELIndexer/ReadAssembler are closed; only pure-Python remains.
    bin_correct = bin_adjacent = 0
    refined_correct = refined_adjacent = 0
    rescued_by_alignment = 0
    anchor_ok_align_poor = 0
    similar_region_wrong_bin = 0
    anchor_scores: List[float] = []
    aln_scores: List[float] = []

    # per-class counters
    pc = {"bin_c": 0, "bin_a": 0, "ref_c": 0, "ref_a": 0}  # perfect
    rcnt = {"bin_c": 0, "bin_a": 0, "ref_c": 0, "ref_a": 0}  # random

    HIGH_MATCH_THRESH = int(0.90 * window)

    for i in range(len(reads)):
        tru = int(true_starts[i])
        p = placements[i]
        cls_perfect = bool(is_perfect[i])

        anchor_start = extract_anchor_start(p)
        refined_start = extract_refined_start(p)

        true_bin = bin_from_start(tru, genome_len, window, stride)
        anchor_bin = bin_from_start(anchor_start, genome_len, window, stride)
        refined_bin = bin_from_start(refined_start, genome_len, window, stride)

        anchor_ok = anchor_adj = False
        if true_bin and anchor_bin:
            if anchor_bin[1] == true_bin[1]:
                anchor_ok = True
            elif abs(anchor_bin[1] - true_bin[1]) == 1:
                anchor_adj = True

        if anchor_ok:
            bin_correct += 1
            if cls_perfect: pc["bin_c"] += 1
            else: rcnt["bin_c"] += 1
        elif anchor_adj:
            bin_adjacent += 1
            if cls_perfect: pc["bin_a"] += 1
            else: rcnt["bin_a"] += 1

        ref_ok = ref_adj = False
        if true_bin and refined_bin:
            if refined_bin[1] == true_bin[1]:
                ref_ok = True
            elif abs(refined_bin[1] - true_bin[1]) == 1:
                ref_adj = True

        if ref_ok:
            refined_correct += 1
            if cls_perfect: pc["ref_c"] += 1
            else: rcnt["ref_c"] += 1
        elif ref_adj:
            refined_adjacent += 1
            if cls_perfect: pc["ref_a"] += 1
            else: rcnt["ref_a"] += 1

        if (not anchor_ok) and ref_ok:
            rescued_by_alignment += 1
        if anchor_ok and (not ref_ok):
            anchor_ok_align_poor += 1

        a_score = extract_anchor_score(p)
        if isinstance(a_score, float):
            anchor_scores.append(a_score)
        s_score = extract_alignment_score(p)
        if isinstance(s_score, float):
            aln_scores.append(s_score)

        cig = extract_cigar(p)
        cc = cigar_counts(cig)
        match_bp = cc.get("M", 0) + cc.get("=", 0)
        if (not anchor_ok) and (match_bp >= HIGH_MATCH_THRESH):
            similar_region_wrong_bin += 1

    total = len(reads)
    n_perfect = sum(1 for b in is_perfect if b)
    n_random = total - n_perfect

    bin_wrong = total - (bin_correct + bin_adjacent)
    refined_wrong = total - (refined_correct + refined_adjacent)

    bin_wrong_perfect = max(0, n_perfect - (pc["bin_c"] + pc["bin_a"]))
    refined_wrong_perfect = max(0, n_perfect - (pc["ref_c"] + pc["ref_a"]))
    bin_wrong_random = max(0, n_random - (rcnt["bin_c"] + rcnt["bin_a"]))
    refined_wrong_random = max(0, n_random - (rcnt["ref_c"] + rcnt["ref_a"]))

    # Final aggressive cleanup before process returns result
    _teardown_cuda_and_cpu()

    rep = StrideReport(
        stride=stride,
        n_reads=total,
        n_perfect=n_perfect,
        n_random=n_random,
        bin_correct=bin_correct,
        bin_adjacent=bin_adjacent,
        bin_wrong=bin_wrong,
        refined_correct=refined_correct,
        refined_adjacent=refined_adjacent,
        refined_wrong=refined_wrong,
        rescued_by_alignment=rescued_by_alignment,
        anchor_ok_align_poor=anchor_ok_align_poor,
        similar_region_wrong_bin=similar_region_wrong_bin,
        bin_correct_perfect=pc["bin_c"],
        bin_adjacent_perfect=pc["bin_a"],
        bin_wrong_perfect=bin_wrong_perfect,
        refined_correct_perfect=pc["ref_c"],
        refined_adjacent_perfect=pc["ref_a"],
        refined_wrong_perfect=refined_wrong_perfect,
        bin_correct_random=rcnt["bin_c"],
        bin_adjacent_random=rcnt["bin_a"],
        bin_wrong_random=bin_wrong_random,
        refined_correct_random=rcnt["ref_c"],
        refined_adjacent_random=rcnt["ref_a"],
        refined_wrong_random=refined_wrong_random,
        anchor_scores=anchor_scores,
        aln_scores=aln_scores,
        query_time_sec_total=query_time_total,
        query_time_ms_per_read=query_time_ms_per_read,
    )
    return asdict(rep)


# ============================================================
# Main
# ============================================================

def _acc(ok: int, adj: int, tot: int) -> Tuple[float, float]:
    if tot <= 0:
        return (float("nan"), float("nan"))
    a1 = ok / tot
    a1p = (ok + adj) / tot
    return a1, a1p

def _write_results_csv_jsonl(rows: List[Dict[str, Any]], csv_path: Path, jsonl_path: Path):
    # Flatten and keep a stable header order
    if not rows:
        return
    # Determine headers as union of keys
    headers: List[str] = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(jsonl_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[RESULTS] wrote {csv_path} and {jsonl_path}")

def main():
    print(f"[SETUP] FASTA={FASTA_PATH}")
    genome = _load_chr22_slice(FASTA_PATH, TARGET_LEN)
    print(f"[INFO] Using chr22 slice: {len(genome):,} bp (window={WINDOW})")
    print(f"[INFO] Strides: {STRIDES}")
    print(f"[INFO] Reads/stride: {READS_PER_STRIDE} (perfect={int(PCT_PERFECT*READS_PER_STRIDE)}, random={READS_PER_STRIDE - int(PCT_PERFECT*READS_PER_STRIDE)})")

    # Phase 1: ensure indices exist (each in its own fresh worker to avoid lingering native state)
    print("\n[PHASE 1] Ensure indices (build-or-load once; isolated workers)")
    ctx = mp.get_context("spawn")
    build_info_by_stride: Dict[int, Dict[str, Any]] = {}
    with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
        for s in STRIDES:
            print(f"[PHASE 1] stride={s} …")
            info = pool.apply(ensure_index_cached, (str(FASTA_PATH), WINDOW, s), {"rc_index": True})  # type: ignore[arg-type]
            build_info_by_stride[s] = info
            time.sleep(0.05)

    # Phase 2: evaluate (again, isolate each stride in a fresh worker)
    print("\n[PHASE 2] Evaluate binning vs alignment (separate perfect vs random; isolated workers)")
    reports: List[StrideReport] = []
    merged_rows: List[Dict[str, Any]] = []

    with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
        for s in STRIDES:
            print(f"\n[SWEEP] stride={s}")
            rep_dict = pool.apply(
                _evaluate_stride_worker,
                ((
                    str(FASTA_PATH),
                    s,
                    WINDOW,
                    READS_PER_STRIDE,
                    PCT_PERFECT,
                    SEED,
                    TARGET_LEN,
                ),)
            )
            rep = StrideReport(**rep_dict)  # type: ignore[arg-type]
            reports.append(rep)

            # Overall
            a1, a1p = _acc(rep.bin_correct, rep.bin_adjacent, rep.n_reads)
            ra1, ra1p = _acc(rep.refined_correct, rep.refined_adjacent, rep.n_reads)

            # Perfect-only
            a1_p, a1p_p = _acc(rep.bin_correct_perfect, rep.bin_adjacent_perfect, rep.n_perfect)
            ra1_p, ra1p_p = _acc(rep.refined_correct_perfect, rep.refined_adjacent_perfect, rep.n_perfect)

            # Random-only
            a1_r, a1p_r = _acc(rep.bin_correct_random, rep.bin_adjacent_random, rep.n_random)
            ra1_r, ra1p_r = _acc(rep.refined_correct_random, rep.refined_adjacent_random, rep.n_random)

            bi = build_info_by_stride.get(s, {})
            size_mb = (bi.get("index_size_bytes") or 0) / 1e6

            print(
                f"  index : build/load={bi.get('index_build_or_load_time_sec', float('nan')):.3f}s | size={size_mb:.2f}MB | rows={bi.get('index_rows')}"
            )
            print(
                f"  query : total={rep.query_time_sec_total:.3f}s | per_read={rep.query_time_ms_per_read:.2f} ms"
            )
            print(
                f"  bins  : OVERALL acc@1={a1:.3f} acc@1+adj={a1p:.3f} | ok/adj/wrong={rep.bin_correct}/{rep.bin_adjacent}/{rep.bin_wrong}"
            )
            print(
                f"          PERFECT acc@1={a1_p:.3f} acc@1+adj={a1p_p:.3f} | ok/adj/wrong={rep.bin_correct_perfect}/{rep.bin_adjacent_perfect}/{rep.bin_wrong_perfect}"
            )
            print(
                f"          RANDOM  acc@1={a1_r:.3f} acc@1+adj={a1p_r:.3f} | ok/adj/wrong={rep.bin_correct_random}/{rep.bin_adjacent_random}/{rep.bin_wrong_random}"
            )
            print(
                f"  align : OVERALL acc@1={ra1:.3f} acc@1+adj={ra1p:.3f} | ok/adj/wrong={rep.refined_correct}/{rep.refined_adjacent}/{rep.refined_wrong} | rescued={rep.rescued_by_alignment} anchor_ok_align_poor={rep.anchor_ok_align_poor}"
            )
            print(
                f"          PERFECT acc@1={ra1_p:.3f} acc@1+adj={ra1p_p:.3f} | ok/adj/wrong={rep.refined_correct_perfect}/{rep.refined_adjacent_perfect}/{rep.refined_wrong_perfect}"
            )
            print(
                f"          RANDOM  acc@1={ra1_r:.3f} acc@1+adj={ra1p_r:.3f} | ok/adj/wrong={rep.refined_correct_random}/{rep.refined_adjacent_random}/{rep.refined_wrong_random}"
            )
            print(
                f"  diag  : similar_region_wrong_bin(≥90%M)={rep.similar_region_wrong_bin:>3} | n_scores(anchor,aln)=({len(rep.anchor_scores)},{len(rep.aln_scores)})"
            )

            # Merge into a single flat row for CSV/JSONL
            row: Dict[str, Any] = dict(asdict(rep))
            # add computed metrics and index metrics
            row.update({
                "acc_bin_overall@1": a1,
                "acc_bin_overall@1+adj": a1p,
                "acc_bin_perfect@1": a1_p,
                "acc_bin_perfect@1+adj": a1p_p,
                "acc_bin_random@1": a1_r,
                "acc_bin_random@1+adj": a1p_r,
                "acc_refined_overall@1": ra1,
                "acc_refined_overall@1+adj": ra1p,
                "acc_refined_perfect@1": ra1_p,
                "acc_refined_perfect@1+adj": ra1p_p,
                "acc_refined_random@1": ra1_r,
                "acc_refined_random@1+adj": ra1p_r,
                "index_build_or_load_time_sec": bi.get("index_build_or_load_time_sec"),
                "index_size_bytes": bi.get("index_size_bytes"),
                "index_size_mb": size_mb,
                "index_rows": bi.get("index_rows"),
                "index_built_now": bi.get("index_built_now"),
                "index_dir": bi.get("index_dir"),
            })
            merged_rows.append(row)

            _teardown_cuda_and_cpu()
            time.sleep(0.02)

    # Rank best stride (bin acc@1, tie-breaker acc@1+adj)
    def sort_key(r: StrideReport):
        a1 = r.bin_correct / max(1, r.n_reads)
        a1p = (r.bin_correct + r.bin_adjacent) / max(1, r.n_reads)
        return (a1, a1p)

    reports_sorted = sorted(reports, key=sort_key, reverse=True)
    best = reports_sorted[0]

    print("\n==================== SUMMARY (by bin acc@1 OVERALL) ====================")
    for r in reports_sorted:
        a1 = r.bin_correct / r.n_reads
        a1p = (r.bin_correct + r.bin_adjacent) / r.n_reads
        print(f"stride={r.stride:>4} | acc@1={a1:.3f} | acc@1+adj={a1p:.3f} "
              f"| bins(ok/adj/wrong)={r.bin_correct}/{r.bin_adjacent}/{r.bin_wrong} "
              f"| align(ok/adj/wrong)={r.refined_correct}/{r.refined_adjacent}/{r.refined_wrong} "
              f"| per_read_query_ms={r.query_time_ms_per_read:.2f}")

    print("\n[WINNER]")
    print(f"Best stride: {best.stride} "
          f"(acc@1={best.bin_correct/best.n_reads:.3f}, acc@1+adj={(best.bin_correct+best.bin_adjacent)/best.n_reads:.3f})")

    # Write results to files
    _write_results_csv_jsonl(merged_rows, RESULTS_CSV, RESULTS_JSONL)

    # Final parent-side cleanup
    _teardown_cuda_and_cpu()


if __name__ == "__main__":
    mp.freeze_support()
    main()
