from src.configs import IndexConfig
from src.HyenaBackend import HyenaBackend
cfg = IndexConfig()

backend =    backend = HyenaBackend(
                model_name=cfg.model_name,
                model_dir=cfg.model_dir,
                pooling="mean",
                normalize=False,
                offline=True,
                prefer_cuda=True,
            )

pooler = backend.build_pooler(
    direction="exp_left",
    tau=64.0,
    pooling_axis="position",
    layer_spec=-7,
    rc_average=False,     # set True for RC-averaged sequence embeddings
)

seqs = ["ACGT"*1000, "TTTT"*800 + "AC", "NNNNACGT"]
X = pooler.embed(seqs)      # -> torch.FloatTensor [N, D], L2-normalized rows
print(X.shape, X.dtype, X.device)


# tests/test_hel_end_to_end.py
import os
import math
import tempfile
from pathlib import Path

import pytest
import torch
import numpy as np

from src.configs import IndexConfig
from src.HELIndexer import HELIndexer
from src.HyenaBackend import HyenaBackend
from src.DNALUTEncoder import DNALUTEncoder
from src.CudaWindower import CudaWindower


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for this end-to-end test."
)

cfg = IndexConfig()

def _make_fasta(tmpdir: Path, name: str = "chrTest", length: int = 4096) -> Path:
    # Deterministic synthetic sequence (no Ns to keep comparisons clean)
    bases = ("A", "C", "G", "T")
    seq = "".join(bases[i % 4] for i in range(length))
    fa = tmpdir / "tiny.fa"
    fa.write_text(f">{name}\n{seq}\n", encoding="utf-8")
    return fa

def _make_multi_fasta(tmpdir: Path, specs):
    """
    specs: List[Tuple[str, int, int]] -> (name, length, seed)
    Returns path to fasta and dict name->seq
    """
    bases = np.array(list("ACGT"))
    fa = tmpdir / "multi.fa"
    seqs = {}
    with fa.open("w", encoding="utf-8") as fh:
        for name, length, seed in specs:
            rng = np.random.default_rng(seed)
            seq = "".join(rng.choice(bases, size=length).tolist())
            fh.write(f">{name}\n{seq}\n")
            seqs[name] = seq
    return fa, seqs


def _starts_with_tail(L: int, win: int, stride: int):
    """Match HELIndexer forward tail coverage starts."""
    if L <= win:
        return [0]
    starts = list(range(0, L - win + 1, stride))
    if (L - win) % stride != 0:
        starts.append(L - win)
    return starts


def _windows_ids_with_tail(ids: torch.Tensor, win: int, stride: int, pad_id: int) -> torch.Tensor:
    """Forward strand windows with HELIndexer-compatible tail coverage."""
    assert ids.is_cuda and ids.dtype == torch.long and ids.ndim == 1
    L = int(ids.numel())
    dev = ids.device
    wind = CudaWindower(device=dev)

    if L < win:
        w = torch.full((win,), int(pad_id), dtype=torch.long, device=dev)
        w[-L:] = ids
        return w.unsqueeze(0)

    base = 1 + (L - win) // stride
    windows = wind.as_windows(ids, win, stride)  # [base, win]
    if (L - win) % stride != 0:
        windows = torch.cat([windows, ids[-win:].unsqueeze(0)], dim=0)
    return windows.contiguous()


def test_embeddings_ids_equal_embed_best_tensor_smoke():
    """
    Verify that embeddings produced by:
      - backend.embed_tokens(ids, pooling='mean', rc_invariant=False)
      - backend.embed_best(ids,   pooling='mean', rc_invariant=False)   (tensor path)
    are equal (within tolerance). This tests the 'ids' path wiring.
    """
    backend = HyenaBackend(
                model_name=cfg.model_name,
                model_dir=cfg.model_dir,
                pooling="mean",
                normalize=False,
                offline=True,
                prefer_cuda=True,
            )
    enc = DNALUTEncoder.from_backend(backend)

    seq = ("ACGT" * 700)[:2048]  # 2048 bp
    ids = enc.encode_to_cuda_ids(seq).unsqueeze(0)  # [1, T] cuda:long

    emb_tokens = backend.embed_tokens(ids, pooling="mean", rc_invariant=False)
    emb_best   = backend.embed_best(ids,  pooling="mean", rc_invariant=False)

    assert emb_tokens.shape == emb_best.shape
    torch.testing.assert_close(emb_tokens, emb_best, rtol=1e-4, atol=1e-4)


def test_end_to_end_retrieval_with_embed_best(tmp_path: Path):
    specs = [
        ("chrA", 3073, 1),   # force tail
        ("chrB", 4097, 2),   # force tail
        ("chrC", 2049, 3),   # force tail
    ]
    # tiny FASTA to force a tail window
    fasta, seqs = _make_multi_fasta(tmp_path, specs)
    # Multi-contig FASTA (different RNG seeds so windows aren't identical across contigs)

    # Build index
    index_dir = tmp_path / "index"
    idx = HELIndexer(fasta, cfg, backend="hyena", emb_batch=512).build(
        index_dir, include_dataset=True, verbose=False
    )
    backend: HyenaBackend = idx.backend  # type: ignore
    metas = getattr(idx.index, "metas", None)
    assert metas is not None and len(metas) == idx.n_vec

    # Encoder and pad for CUDA windows
    enc = DNALUTEncoder.from_backend(backend)
    pad_id = int(getattr(backend.tokenizer, "pad_token_id", 0))

    # Collect queries from each contig: first, middle-ish, last (tail) window
    q_ids_list = []
    expected_meta = []  # list of (chrom, start)
    for chrom, length, _seed in specs:
        seq = idx.ref_dict[chrom]
        ids_1d = enc.encode_to_cuda_ids(seq)  # [L] cuda:long

        starts = _starts_with_tail(len(seq), cfg.window, cfg.stride)
        # choose up to 3 windows per contig: first / mid / last
        picks = []
        if len(starts) >= 1:
            picks.append(starts[0])
        if len(starts) >= 3:
            picks.append(starts[len(starts) // 2])
        if len(starts) >= 2:
            picks.append(starts[-1])

        windows_all = _windows_ids_with_tail(ids_1d, cfg.window, cfg.stride, pad_id)
        start_to_row = {s: i for i, s in enumerate(starts)}
        rows = torch.tensor([start_to_row[s] for s in picks], device=ids_1d.device, dtype=torch.long)
        q_ids = windows_all.index_select(0, rows)  # [Qc, win] cuda:long

        q_ids_list.append(q_ids)
        expected_meta.extend([(chrom, int(s)) for s in picks])

    # Concatenate all queries across contigs
    q_ids_all = torch.cat(q_ids_list, dim=0)  # [Q, win] cuda:long

    # Query via embed_best (tensor path, forward only)
    queries = backend.embed_best(q_ids_all, pooling="mean", rc_invariant=False)  # [Q, D] cuda:float
    assert queries.is_cuda

    # Search top-1
    topk = 1
    if hasattr(idx.index, "search"):
        inds, dists = idx.index.search(queries, k=topk)
    elif hasattr(idx.index, "query"):
        inds, dists = idx.index.query(queries, topk=topk)
    else:
        raise RuntimeError("RaftGPU index does not expose search/query")

    inds = torch.as_tensor(inds, device=queries.device)
    dists = torch.as_tensor(dists, device=queries.device)

    # Orientation rule
    allow_orients = (+1,) if not idx.cfg.rc_index else (+1, -1)

    # Verify each query maps back to the correct (chrom, start)
    for qi, (exp_chrom, exp_start) in enumerate(expected_meta):
        hit = int(inds[qi, 0].item())
        dist = float(dists[qi, 0].item())
        chrom, s_bp, orient = metas[hit]
        assert chrom == exp_chrom
        assert int(s_bp) == int(exp_start)
        assert int(orient) in allow_orients, f"unexpected orient={orient} with rc_index={idx.cfg.rc_index}"
        # identical vectors ⇒ near-zero cosine distance
        assert dist <= 1e-5, f"expected near-zero cosine distance, got {dist}"