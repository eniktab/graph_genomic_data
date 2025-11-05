from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# ---------- helpers ----------
_RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")
def revcomp(s: str) -> str:
    return s.translate(_RC)[::-1]

def _iter_query_windows(seq: str, window: int, stride: int):
    """Yield (q_start, subseq) covering the tail exactly once."""
    L = len(seq); T = min(window, L); S = max(1, stride)
    if L <= T:
        yield 0, seq; return
    for i in range(0, L - T + 1, S):
        yield i, seq[i:i+T]
    if (L - T) % S != 0:
        last = L - T
        yield last, seq[last:]

def _best_of_two(Da: np.ndarray, Ia: np.ndarray, Db: np.ndarray, Ib: np.ndarray, metric_internal: str):
    """
    Pick per-row best between A and B (e.g., fwd vs RC).
    For 'inner_product' higher is better; for 'sqeuclidean' lower is better.
    """
    if metric_internal == 'inner_product':
        mask = Da >= Db
    else:
        mask = Da <= Db
    D = np.where(mask, Da, Db)
    I = np.where(mask, Ia, Ib)
    ori_used = np.where(mask, +1, -1)  # +1 = A (fwd), -1 = B (RC)
    return D, I, ori_used

def _decode_hits(q_starts,
                 I,
                 S,
                 metas,
                 tile_len,
                 query_orient=None):
    """
    Map (indices, scores) to rich hit dicts per query window.

    Accepts query_orient in any of:
      - None                  -> assume +1 for all
      - shape (B,)            -> one orientation per row
      - shape (B, K)          -> one orientation per (row, rank) hit
    Handles I/S of shape (B,) by auto-expanding to (B,1).
    """
    import numpy as np

    I = np.asarray(I)
    S = np.asarray(S)

    # Ensure 2D (B, K)
    if I.ndim == 1:
        I = I[:, None]
    if S.ndim == 1:
        S = S[:, None]

    B, K = I.shape
    out = []

    # Normalize query_orient to one of the three supported cases
    if query_orient is not None:
        query_orient = np.asarray(query_orient)
        if query_orient.ndim == 1 and query_orient.shape[0] != B:
            raise ValueError(f"query_orient shape {query_orient.shape} incompatible with B={B}")
        if query_orient.ndim == 2 and query_orient.shape != (B, K):
            # tolerate (B,1) -> (B,K) broadcast
            if query_orient.shape == (B, 1):
                query_orient = np.repeat(query_orient, K, axis=1)
            else:
                raise ValueError(f"query_orient shape {query_orient.shape} incompatible with (B,K)=({B},{K})")

    for r in range(B):
        row_hits = []
        q_start = int(q_starts[r])
        q_end = q_start + tile_len
        for c in range(K):
            j = int(I[r, c])
            s = float(S[r, c])
            try:
                chrom, ref_start, ref_orient = metas[j]
            except Exception as e:
                raise IndexError(f"Bad index_id {j} for metas of len {len(metas)}") from e

            # choose orientation per hit if available, else per row, else +1
            if query_orient is None:
                q_ori = +1
            elif query_orient.ndim == 1:
                q_ori = int(query_orient[r])
            else:
                q_ori = int(query_orient[r, c])

            row_hits.append({
                "chrom": chrom,
                "ref_start": int(ref_start),
                "ref_end": int(ref_start + tile_len),
                "ref_orient": int(ref_orient),
                "query_start": q_start,
                "query_end": q_end,
                "query_orient": q_ori,   # +1 used forward embedding; -1 used RC embedding
                "score": s,              # cosine: similarity; L2: similarity if you converted
                "index_id": j,
                "rank": c
            })
        out.append(row_hits)
    return out

def _aggregate_overall_by_offset(hits_per_row: List[List[Dict[str, Any]]],
                                 q_starts: List[int],
                                 top_n: int,
                                 delta_bin: int = 1000,
                                 use_top_m_per_row: int = 3) -> List[Dict[str, Any]]:
    """
    Aggregate hits by the offset Δ = ref_start - q_start, binned by `delta_bin`.
    Per (chrom, ref_orient, Δ_bin), sum weights (scores). Return top_n clusters with
    a representative coordinate (weighted median of ref_start).
    """
    assert len(hits_per_row) == len(q_starts)
    votes: Dict[Tuple[str,int,int,int], Dict[str,Any]] = {}
    # key = (chrom, ref_orient, delta_bin_idx, sign) ; sign not strictly needed

    for row_idx, (row_hits, q0) in enumerate(zip(hits_per_row, q_starts)):
        m = min(use_top_m_per_row, len(row_hits))
        for h in row_hits[:m]:
            chrom = h["chrom"]
            ref_start = int(h["ref_start"])
            ref_orient = int(h["ref_orient"])
            score = float(h["score"])
            delta = ref_start - int(q0)
            b = int(np.floor(delta / float(delta_bin)))  # Δ bin
            key = (chrom, ref_orient, b, 0)

            bucket = votes.get(key)
            if bucket is None:
                votes[key] = {
                    "chrom": chrom,
                    "ref_orient": ref_orient,
                    "delta_bin": b,
                    "delta_bin_width": delta_bin,
                    "total_score": 0.0,
                    "members": []  # (ref_start, score, row_idx, index_id)
                }
                bucket = votes[key]
            bucket["total_score"] += score
            bucket["members"].append((ref_start, score, row_idx, h["index_id"]))

    # turn buckets into candidates
    candidates: List[Dict[str,Any]] = []
    for bucket in votes.values():
        mem = bucket["members"]
        if not mem:
            continue
        # weighted median of ref_start for representitive coordinate
        ref_starts = np.array([x[0] for x in mem], dtype=np.int64)
        weights = np.array([x[1] for x in mem], dtype=np.float64)
        order = np.argsort(ref_starts)
        ref_sorted = ref_starts[order]; w_sorted = weights[order]
        csum = np.cumsum(w_sorted)
        wmed = ref_sorted[np.searchsorted(csum, 0.5 * csum[-1])]
        # occurrences = number of distinct query windows that voted
        q_rows = {x[2] for x in mem}
        candidates.append({
            "chrom": bucket["chrom"],
            "ref_start": int(wmed),
            "ref_end":   int(wmed) + (mem[0][0] - mem[0][0] + 1),  # placeholder width; your tile len is known outside
            "ref_orient": bucket["ref_orient"],
            "delta_bin": int(bucket["delta_bin"]),
            "delta_span": (int(bucket["delta_bin"]*bucket["delta_bin_width"]),
                           int((bucket["delta_bin"]+1)*bucket["delta_bin_width"]-1)),
            "total_score": float(bucket["total_score"]),
            "occurrences": len(q_rows),
            "n_votes": len(mem)
        })

    candidates.sort(key=lambda x: (x["total_score"], x["occurrences"]), reverse=True)
    return candidates[:min(top_n, len(candidates))]

# ---------- configs ----------
@dataclass
class QueryConfig:
    window: int = 10_000           # will be clamped to backend.max_length
    q_stride: int = 1_000
    topk_per_chunk: int = 8
    rc_search: bool = True
    return_similarity: bool = True # if L2 internally, convert dist->sim via exp(-d)
    overall_top_n: int = 5         # how many overall options to return

@dataclass
class AlignConfig:
    flank: int = 0  # retained for API compatibility; unused here

# ---------- HEL (simple search only) ----------
class HEL:
    def __init__(self, indexer: 'HELIndexer', qcfg: QueryConfig, acfg: AlignConfig, emb_batch: int = 256):
        self.indexer = indexer
        self.EMB_BATCH = int(max(1, emb_batch))
        self.qcfg = qcfg
        self._ref = self.indexer.ref_dict  # not used here, but kept for parity
        self._metric_internal = self.indexer.index._metric_name  # 'inner_product' or 'sqeuclidean'
        self._tile_len = min(self.qcfg.window, self.indexer.backend.max_length)

    def _embed_batch(self, seqs: List[str]):
        import torch
        parts = []
        for i in range(0, len(seqs), self.EMB_BATCH):
            parts.append(self.indexer.backend.embed_best(seqs[i:i+self.EMB_BATCH]))
        return torch.cat(parts, dim=0)

    def locate(self, query_fasta: Path | str) -> List[Dict[str, Any]]:
        """
        Returns per-query dicts with:
          - 'query_id'
          - 'tile_len'
          - 'per_window_hits': list of lists of hit dicts (topK per window)
          - 'overall_top': aggregated top few options across all windows
        """
        from Bio import SeqIO

        results: List[Dict[str, Any]] = []
        for rec in SeqIO.parse(str(query_fasta), "fasta"):
            qid = rec.id
            qseq = str(rec.seq).upper()
            # 1) windows
            q_windows = list(_iter_query_windows(qseq, self._tile_len, self.qcfg.q_stride))
            q_starts = [s for s, _ in q_windows]
            q_seqs   = [t for _, t in q_windows]
            if not q_seqs:
                results.append({"query_id": qid, "tile_len": self._tile_len, "per_window_hits": [], "overall_top": []})
                continue
            # 2) embed fwd (+ RC)
            Qf = self._embed_batch(q_seqs)
            if self.qcfg.rc_search:
                Qr = self._embed_batch([revcomp(s) for s in q_seqs])
            else:
                Qr = None
            # 3) search
            Df, If = self.indexer.index.search(Qf, self.qcfg.topk_per_chunk)  # numpy arrays
            if Qr is not None:
                Dr, Ir = self.indexer.index.search(Qr, self.qcfg.topk_per_chunk)
                S, I, chosen_orient = _best_of_two(Df, If, Dr, Ir, self._metric_internal)
            else:
                S, I = Df, If
                chosen_orient = np.ones((Df.shape[0],), dtype=np.int32)
            # 4) L2 -> similarity if desired
            if self._metric_internal == 'sqeuclidean' and self.qcfg.return_similarity:
                S = np.exp(-S)  # monotone transform for readability
            # 5) decode & aggregate
            metas = self.indexer.index.metas
            hits_per_row = _decode_hits(q_starts, I, S, metas, self._tile_len, chosen_orient)
            overall_top = _aggregate_overall_by_offset(
                hits_per_row,
                q_starts=q_starts,  # <— pass window starts
                top_n=self.qcfg.overall_top_n,
                delta_bin=1000,  # 1 kb bins (tune 500–2000)
                use_top_m_per_row=3  # vote with top-3 per window
            )
            # give ref_end a meaningful value (tile length)
            for c in overall_top:
                c["ref_end"] = c["ref_start"] + self._tile_len

            results.append({
                "query_id": qid,
                "tile_len": self._tile_len,
                "per_window_hits": hits_per_row,
                "overall_top": overall_top
            })
        if not results:
            raise ValueError("No sequences in query FASTA")
        return results