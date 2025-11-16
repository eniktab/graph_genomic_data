# make_chr_windows_hyena_raw_fn.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import pysam
import torch

from src.HyenaBackend import HyenaBackend
try:
    from src.HyenaBackend import HyenaDNAPooler
except Exception:
    HyenaDNAPooler = None


# -------------------- basic utils --------------------

def _load_contig_sequence(fasta_path: Path, contig: str) -> str:
    with pysam.FastaFile(str(fasta_path)) as fa:
        if contig not in fa.references:
            raise ValueError(f"Contig '{contig}' not found in {fasta_path}.")
        return fa.fetch(contig, 0, fa.get_reference_length(contig)).upper()

def _starts_aligned(L: int, window: int, stride: int) -> List[int]:
    last = L - window
    if last < 0: return []
    n = 1 + (last // stride)
    return [i * stride for i in range(n)]

def _starts_misaligned(L: int, window: int, stride: int, offset: int) -> List[int]:
    if not (1 <= offset <= stride - 1):
        raise ValueError(f"offset must be in [1, {stride-1}] (got {offset})")
    last = L - window
    if last < 0: return []
    return list(range(offset, last + 1, stride))

def _batched(seq: List[str], bsz: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), bsz):
        yield seq[i:i + bsz]

def _write_meta(outdir: Path, tag: str, contig: str, starts: List[int], window: int, label: int) -> None:
    rows = [(i, contig, s, s + window, label) for i, s in enumerate(starts)]
    df = pd.DataFrame(rows, columns=["id", "contig", "start", "end", "label"])
    (outdir / tag).mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / tag / "meta.csv", index=False)


# -------------------- embedding helpers --------------------

@torch.inference_mode()
def _probe_T_H(backend: HyenaBackend, seq_example: str) -> Tuple[int, int]:
    """Forward a single sequence to learn token length T and hidden size H."""
    tok = backend.tokenizer
    dev = backend.device
    enc = tok([seq_example], padding=True, truncation=True,
              max_length=getattr(backend, "max_length", None), return_tensors="pt")
    ids = enc["input_ids"].to(dev)
    out = backend.model(ids)
    hs_list = out.hidden_states if hasattr(out, "hidden_states") else [out.last_hidden_state]
    T = hs_list[-1].shape[1]
    H = hs_list[-1].shape[2]
    del out, hs_list, ids
    return int(T), int(H)

@torch.inference_mode()
def _embed_two_layers_raw_list(
    backend: HyenaBackend,
    sequences: List[str],
    batch_size: int,
    best_layer_spec: int,
    store_dtype: torch.dtype,
):
    """
    Returns lists (one tensor per sample) to avoid T-mismatch across mini-batches:
      last_list: [T,H] tensors
      best_list: [T,H] tensors
      ids_list:  [T]   long tensors
      msk_list:  [T]   bool tensors
    """
    tok = backend.tokenizer
    dev = backend.device

    last_list, best_list, ids_list, msk_list = [], [], [], []

    for chunk in _batched(sequences, batch_size):
        enc = tok(chunk, padding="longest", truncation=True,
                  max_length=getattr(backend, "max_length", None), return_tensors="pt")
        ids = enc["input_ids"].to(dev, non_blocking=True)           # [B,T]
        msk = (ids != int(tok.pad_token_id)).to(dev)                # [B,T] bool

        out = backend.model(ids)
        hs = out.hidden_states if hasattr(out, "hidden_states") else [out.last_hidden_state]

        # Resolve best layer index safely (supports negatives)
        try:
            hs_best = hs[best_layer_spec]
        except Exception:
            idx = (len(hs) + best_layer_spec) if best_layer_spec < 0 else best_layer_spec
            idx = max(0, min(len(hs) - 1, idx))
            hs_best = hs[idx]

        hs_last = hs[-1]  # [B,T,H]

        # split into per-sample tensors (CPU + cast)
        B = ids.shape[0]
        for b in range(B):
            last_list.append(hs_last[b].to("cpu", non_blocking=True).to(dtype=store_dtype).contiguous())
            best_list.append(hs_best[b].to("cpu", non_blocking=True).to(dtype=store_dtype).contiguous())
            ids_list.append(ids[b].to("cpu", non_blocking=True).contiguous())
            msk_list.append(msk[b].to("cpu", non_blocking=True).to(torch.bool).contiguous())

        del out, hs_last, hs_best, hs, ids, msk

    return last_list, best_list, ids_list, msk_list


def _pad_stack_hidden(tensors: List[torch.Tensor], T_pad: int, H: int, dtype: torch.dtype) -> torch.Tensor:
    """tensors: list of [T_i, H] → [N, T_pad, H] with zero pad."""
    N = len(tensors)
    out = torch.zeros((N, T_pad, H), dtype=dtype)
    for i, t in enumerate(tensors):
        Ti = t.shape[0]
        out[i, :Ti, :] = t
    return out

def _pad_stack_ids(ids_list: List[torch.Tensor], T_pad: int, pad_id: int) -> torch.Tensor:
    N = len(ids_list)
    out = torch.full((N, T_pad), int(pad_id), dtype=torch.long)
    for i, t in enumerate(ids_list):
        Ti = t.shape[0]
        out[i, :Ti] = t
    return out

def _pad_stack_mask(msk_list: List[torch.Tensor], T_pad: int) -> torch.Tensor:
    N = len(msk_list)
    out = torch.zeros((N, T_pad), dtype=torch.bool)
    for i, t in enumerate(msk_list):
        Ti = t.shape[0]
        out[i, :Ti] = t
    return out


# -------------------- auto shard sizing --------------------

def _estimate_bytes_per_sample(T: int, H: int, store_dtype: torch.dtype) -> int:
    """
    Rough size of one sample when we store:
      hidden_last [T,H] + hidden_best [T,H]  (both in store_dtype)
      input_ids   [T]   (int64)
      attention   [T]   (bool, 1 byte)
      labels, starts, ends (tiny)
    """
    bytes_per_el = 2 if store_dtype == torch.float16 else 4
    hidden = 2 * T * H * bytes_per_el       # two layers
    ids    = T * 8                          # int64
    mask   = T * 1                          # bool
    misc   = 8 + 8 + 8                      # labels/start/end overhead per sample (upper bound)
    return hidden + ids + mask + misc


# -------------------- writer --------------------

def _save_shards(
    outdir: Path,
    tag: str,
    seq: str,
    window: int,
    starts: List[int],
    backend: HyenaBackend,
    *,
    store_dtype: torch.dtype,
    batch_size: int,
    best_layer_spec: int,
    shard_size: int,
) -> int:
    """
    Packs MANY samples per shard; pads to shard-local max T.
    Returns number of shards written.
    """
    od = outdir / tag / "shards"
    od.mkdir(parents=True, exist_ok=True)

    windows = [seq[s:s + window] for s in starts]
    N = len(windows)
    if N == 0:
        return 0

    tok = backend.tokenizer
    pad_id = int(tok.pad_token_id)

    num_shards, cursor = 0, 0
    while cursor < N:
        upto = min(cursor + shard_size, N)
        win_chunk = windows[cursor:upto]
        start_chunk = starts[cursor:upto]

        last_list, best_list, ids_list, msk_list = _embed_two_layers_raw_list(
            backend, win_chunk, batch_size=batch_size,
            best_layer_spec=best_layer_spec, store_dtype=store_dtype
        )

        # find shard-local T_pad and H
        T_pad = max(t.shape[0] for t in last_list)
        H = last_list[0].shape[1]

        hidden_last = _pad_stack_hidden(last_list, T_pad, H, store_dtype)
        hidden_best = _pad_stack_hidden(best_list, T_pad, H, store_dtype)
        input_ids   = _pad_stack_ids(ids_list, T_pad, pad_id)
        attn_mask   = _pad_stack_mask(msk_list, T_pad)

        shard = {
            "input_ids":    input_ids,                  # [n, T_pad] long
            "attention_mask": attn_mask,               # [n, T_pad] bool
            "hidden_last":  hidden_last,               # [n, T_pad, H] f16/f32
            "hidden_best":  hidden_best,               # [n, T_pad, H] f16/f32
            "labels":       torch.full((len(win_chunk),), 1 if tag=="aligned" else 0, dtype=torch.long),
            "starts":       torch.as_tensor(start_chunk, dtype=torch.long),
            "ends":         torch.as_tensor([s + window for s in start_chunk], dtype=torch.long),
        }

        torch.save(shard, od / f"{tag}-{num_shards:05d}.pt")
        num_shards += 1
        cursor = upto

        # free CPU buffers promptly
        del hidden_last, hidden_best, input_ids, attn_mask, shard

    return num_shards


# ==================== PUBLIC ENTRY ====================

def make_chr_windows_hyena_raw(
    fasta: str | Path,
    contig: str,
    outdir: str | Path,
    *,
    window: int = 10_000,
    stride: int = 5_000,
    offset: Optional[int] = None,
    max_n_frac: float = 1.0,
    batch: int = 128,
    # choose ONE of the two knobs below:
    shard_size: Optional[int] = None,      # exact samples per shard (if set)
    target_shard_mb: int = 1024,           # else: auto choose shard_size to hit ~this MB
    dtype: str = "fp16",                   # "fp16" or "fp32"
    model_name: str = "hyenadna-small-32k-seqlen-hf",
    model_dir: str = "hf-cache/models/",
    preset_name: str = "cluster_max_sep",
    best_layer_spec: Optional[int] = None, # default inferred from preset, else e.g. -7
) -> Dict[str, Any]:
    """
    Build aligned/misaligned window datasets. Save **many samples per shard**, with:
      - hidden_last [N, T_pad, H]
      - hidden_best [N, T_pad, H]
    No pooling. No normalization.

    Use `shard_size` to force samples per shard, or leave None and set `target_shard_mb`
    to automatically pick a large shard size (few big files).
    """
    fasta  = Path(fasta)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    seq = _load_contig_sequence(fasta, contig)
    L = len(seq)

    aligned = _starts_aligned(L, window, stride)

    if offset is None:
        rng = np.random.default_rng()
        off = int(rng.integers(low=1, high=max(2, stride)))
        if off >= stride: off = max(1, stride - 1)
    else:
        off = int(offset)
    misaligned = _starts_misaligned(L, window, stride, off)

    # optional N filter
    if max_n_frac < 1.0:
        def keep(s: int) -> bool:
            w = seq[s:s + window]
            return (w.count("N") / float(window)) <= max_n_frac
        aligned   = [s for s in aligned   if keep(s)]
        misaligned = [s for s in misaligned if keep(s)]

    # backend: raw per-token states
    backend = HyenaBackend(
        model_name=model_name,
        model_dir=model_dir,
        pooling="none",
        normalize=False,
        offline=True,
        prefer_cuda=True,
    )

    # infer "best" layer if needed
    if best_layer_spec is None:
        inferred = -7
        try:
            if HyenaDNAPooler is not None:
                pooler = HyenaDNAPooler.from_preset(backend, preset_name)
                inferred = int(getattr(pooler, "layer_spec", inferred))
        except Exception:
            pass
        best_layer_spec = inferred

    store_dtype = torch.float16 if dtype.lower() == "fp16" else torch.float32

    # auto shard sizing if shard_size not given
    if shard_size is None:
        # probe on one real window for accurate T,H
        probe_seq = seq[aligned[0]:aligned[0]+window] if aligned else seq[0:window]
        T_est, H = _probe_T_H(backend, probe_seq)
        bps = _estimate_bytes_per_sample(T_est, H, store_dtype)
        target_bytes = int(target_shard_mb * (1024**2))
        shard_size = max(1, target_bytes // max(1, bps))

    # write metas
    _write_meta(outdir, "aligned",   contig, aligned,   window, label=1)
    _write_meta(outdir, "misaligned", contig, misaligned, window, label=0)

    # save few big shards
    n_shards_aln = _save_shards(
        outdir, "aligned", seq, window, aligned, backend,
        store_dtype=store_dtype, batch_size=batch, best_layer_spec=best_layer_spec,
        shard_size=shard_size,
    )
    n_shards_mis = _save_shards(
        outdir, "misaligned", seq, window, misaligned, backend,
        store_dtype=store_dtype, batch_size=batch, best_layer_spec=best_layer_spec,
        shard_size=shard_size,
    )

    manifest = {
        "contig": contig,
        "window": window,
        "stride": stride,
        "misalign_offset": off,
        "n_aligned": len(aligned),
        "n_misaligned": len(misaligned),
        "n_shards_aligned": n_shards_aln,
        "n_shards_misaligned": n_shards_mis,
        "storage_dtype": "fp16" if store_dtype == torch.float16 else "fp32",
        "model_name": model_name,
        "model_dir": model_dir,
        "hidden_states_saved": ["last", f"best:{best_layer_spec}"],
        "batch": batch,
        "max_n_frac": max_n_frac,
        "target_shard_mb": target_shard_mb,
        "effective_shard_size": shard_size,
    }
    (outdir / "manifest.json").write_text((pd.Series(manifest).to_json(indent=2)), encoding="utf-8")
    return manifest

if __name__ == "__main__":
    manifest = make_chr_windows_hyena_raw(
        fasta="/g/data/te53/en9803/sandpit/graph_genomics/chr22/chm13v2_chr22.fa.gz",
        contig="chr22",
        outdir="/g/data/te53/en9803/sandpit/graph_genomics/chr22/windows_raw",
        window=10_000, stride=5_000,
        # leave shard_size=None so we auto-size:
        target_shard_mb=1024,        # ~1 GB per shard (adjust to taste)
        batch=128, dtype="fp16",
        model_name = "hyenadna-large-1m-seqlen-hf",
        model_dir = "/g/data/te53/en9803/data/scratch/hf-cache/models/",
        # best_layer_spec=-7,        # optional override

    )