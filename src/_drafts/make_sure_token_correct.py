# ids_path_vs_tokenizer_benchmark.py (improved)
from __future__ import annotations
import time, random, math
from typing import List, Tuple, Iterator

import torch

try:
    from evo2 import Evo2  # only for tokenizer; compute shim below
except Exception:
    Evo2 = None

# If your helper is a local file, ensure it's on PYTHONPATH or in the CWD.
from src.dna_tokenizer import IDsPathHelper

# --------------------------- config ---------------------------------
DNA = "ACGTNacgtn"

# Workload
B = 4096             # batch size (number of sequences)
T = 512              # tokens per sequence (equal length)
REPS = 3             # repetitions for timing
DEVICE = "cuda:0"

# Compute shim knobs (to demonstrate overlap benefits)
USE_COMPUTE_SHIM = True      # if False, embedder is identity-like (very light)
EMB_DIM = 128                # embedding dimension for the shim
VOCAB_MOD = 8192             # reduce token ids modulo this for Embedding table

# IDsPathHelper performance knobs
IDS_MAX_TOKENS_PER_CALL = 1_048_576  # raised ceiling
PREFER_INT32_H2D = True              # host→device as int32, cast to int64 on device
OVERLAP_COPY_AND_COMPUTE = True      # enable pipelined streamer

# --------------------------- utils ---------------------------------

def make_equal_len_sequences(B: int, T: int, seed: int = 0) -> List[str]:
    random.seed(seed)
    return ["".join(random.choice(DNA) for _ in range(T)) for _ in range(B)]

# ---------------------- robust tokenizer probes --------------------

def _safe_get_vocab(tokenizer):
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        try:
            v = get_vocab()
            if isinstance(v, dict):
                return v
        except Exception:
            pass
    try:
        v = getattr(tokenizer, "vocab")
        if isinstance(v, dict):
            return v
    except Exception:
        pass
    return None

def _safe_token_to_id(tokenizer, token_str: str):
    fn = getattr(tokenizer, "token_to_id", None)
    if callable(fn):
        try:
            out = fn(token_str)
            if isinstance(out, int) and out >= 0:
                return out
        except Exception:
            pass
    cti = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(cti):
        try:
            out = cti(token_str)
            if isinstance(out, int) and out >= 0:
                return out
        except Exception:
            pass
    vocab = _safe_get_vocab(tokenizer)
    if vocab and token_str in vocab and isinstance(vocab[token_str], int):
        return int(vocab[token_str])
    return None

def _encode_one_char_to_id(tokenizer, ch: str):
    enc = getattr(tokenizer, "encode", None)
    if not callable(enc):
        return None
    try:
        try:
            out = enc(ch, add_special_tokens=False)
        except TypeError:
            out = enc(ch)
        ids = out.ids if hasattr(out, "ids") else out
        if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], int):
            return ids[0]
    except Exception:
        pass
    return None

def _discover_unknown_id(tokenizer) -> int:
    # Prefer explicit N, fall back to pad, else 0.
    for cand in ("N", "n"):
        tid = _safe_token_to_id(tokenizer, cand)
        if isinstance(tid, int):
            return tid
    for pad_tok in ("<pad>", "[PAD]", "PAD", "pad"):
        tid = _safe_token_to_id(tokenizer, pad_tok)
        if isinstance(tid, int):
            return tid
    return 0

# ---------------------- canonical ID getters -----------------------

def _tok2ids_via_encode_batch(tokenizer, seqs: List[str]) -> List[List[int]]:
    if hasattr(tokenizer, "encode_batch"):
        try:
            encs = tokenizer.encode_batch(seqs, add_special_tokens=False)
        except TypeError:
            encs = tokenizer.encode_batch(seqs)
        out = []
        for e in encs:
            if hasattr(e, "ids"):
                out.append(e.ids)
            elif isinstance(e, list) and all(isinstance(x, int) for x in e):
                out.append(e)
            else:
                raise RuntimeError("encode_batch returned unsupported element.")
        return out
    return None

def _tok2ids_via_encode(tokenizer, s: str) -> List[int] | None:
    enc = getattr(tokenizer, "encode", None)
    if callable(enc):
        try:
            try:
                out = enc(s, add_special_tokens=False)
            except TypeError:
                out = enc(s)
            if hasattr(out, "ids"):
                return list(out.ids)
            if isinstance(out, list) and all(isinstance(x, int) for x in out):
                return out
        except Exception:
            pass
    return None

def _tok2ids_via_tokenize_map(tokenizer, s: str) -> List[int] | None:
    # tokenizer.tokenize returns tokens; map them to ids via convert_tokens_to_ids or token_to_id
    tokf = getattr(tokenizer, "tokenize", None)
    if not callable(tokf):
        return None
    try:
        toks = tokf(s)
        if isinstance(toks, torch.Tensor):
            return toks.long().tolist()
        if not isinstance(toks, list):
            return None
        if toks and all(isinstance(x, int) for x in toks):
            return toks
        out = []
        for t in toks:
            if isinstance(t, int):
                out.append(int(t))
            else:
                tid = _safe_token_to_id(tokenizer, t)
                if tid is None:
                    return None
                out.append(int(tid))
        return out
    except Exception:
        return None

def _tok2ids_via_charwise(tokenizer, s: str) -> List[int]:
    unknown_id = _discover_unknown_id(tokenizer)
    out: List[int] = []
    for ch in s:
        tid = _safe_token_to_id(tokenizer, ch)
        if tid is None:
            code = ord(ch)
            for key in (f"<0x{code:02X}>", f"<0x{code:02x}>"):
                tid = _safe_token_to_id(tokenizer, key)
                if isinstance(tid, int):
                    break
        if tid is None:
            tid = _encode_one_char_to_id(tokenizer, ch)
        out.append(int(tid if isinstance(tid, int) else unknown_id))
    return out

def _tok2ids_user_way(tokenizer, s: str) -> List[int]:
    encb = _tok2ids_via_encode_batch(tokenizer, [s])
    if encb is not None and len(encb) == 1:
        return list(encb[0])
    enc1 = _tok2ids_via_encode(tokenizer, s)
    if enc1 is not None:
        return enc1
    tmap = _tok2ids_via_tokenize_map(tokenizer, s)
    if tmap is not None:
        return tmap
    return _tok2ids_via_charwise(tokenizer, s)

def _tok2ids_batch_encode(tokenizer, seqs: List[str]) -> List[List[int]]:
    encb = _tok2ids_via_encode_batch(tokenizer, seqs)
    if encb is not None:
        return encb
    return [_tok2ids_user_way(tokenizer, s) for s in seqs]

# ----------------------- tensor helpers --------------------------------------

def as_cuda(ids_2d, device: str = DEVICE) -> torch.Tensor:
    """Accepts list[list[int]] or CPU tensor; returns CUDA int64; pins when possible."""
    if isinstance(ids_2d, torch.Tensor):
        t = ids_2d
        if t.dtype != torch.int64:
            t = t.to(torch.int64)
        if t.device.type != "cpu":
            return t.to(device, non_blocking=True)
        try:
            t = t.pin_memory()
        except Exception:
            pass
        return t.to(device, non_blocking=True)
    t = torch.tensor(ids_2d, dtype=torch.int64)
    try:
        t = t.pin_memory()
    except Exception:
        pass
    return t.to(device, non_blocking=True)

# --------------------------- compute shim ----------------------------------

class _NoComputeShim:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    @torch.no_grad()
    def embed_tokens(self, x, **_):
        return x.float()

class _ComputeShim:
    """Adds meaningful GPU compute to expose overlap benefits.
    Maps token IDs (modded) to embeddings and pushes through a tiny MLP.
    Output: [B, EMB_DIM]
    """
    def __init__(self, tokenizer, device: str):
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.emb = torch.nn.Embedding(VOCAB_MOD, EMB_DIM, device=self.device)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(EMB_DIM, EMB_DIM * 2, device=self.device),
            torch.nn.ReLU(),
            torch.nn.Linear(EMB_DIM * 2, EMB_DIM, device=self.device),
        )
        torch.nn.init.normal_(self.emb.weight, std=0.02)
        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    @torch.no_grad()
    def embed_tokens(self, x, **_):
        # x: [B, T] Long; When using int32 H2D path we cast on-device before call.
        x = x.to(self.device, non_blocking=True)
        x = x % VOCAB_MOD
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            e = self.emb(x)        # [B, T, D]
            y = e.mean(dim=1)     # [B, D]
            out = self.net(y)     # [B, D]
        return out.float()

# --------------------------- main ----------------------------------

def main():
    # Prepare tokenizer
    if Evo2 is not None:
        evo2_model = Evo2("evo2_7b")
        tok = getattr(evo2_model, "tokenizer", evo2_model)
    else:
        class _DummyTok:
            def encode(self, s, add_special_tokens=False):
                return [ord(c) for c in s]
        tok = _DummyTok()

    # Choose compute shim
    embedder = _ComputeShim(tok, DEVICE) if USE_COMPUTE_SHIM else _NoComputeShim(tok)

    helper = IDsPathHelper(
        embedder,
        ids_max_tokens_per_call=IDS_MAX_TOKENS_PER_CALL,
        prefer_int32_h2d=PREFER_INT32_H2D,
        overlap_h2d_compute=OVERLAP_COPY_AND_COMPUTE,
    )

    try:
        helper.discover()
    except Exception as e:
        print("IDsPathHelper.discover() warning:", e)

    # Prepare data
    seqs = make_equal_len_sequences(B, T, seed=42)

    # ---------------------- correctness ---------------------------------
    s = "ACGTNt"
    ids_user = _tok2ids_user_way(tok, s)
    ids_batch = _tok2ids_batch_encode(tok, [s])[0]

    def _helper_safe_encode_batch(sequences: List[str]) -> List[List[int]]:
        try:
            ids = helper.encode_batch_to_ids(sequences)  # Tensor int64 CPU pinned
            return ids.tolist()
        except Exception:
            return _tok2ids_batch_encode(tok, sequences)

    ids_helper = _helper_safe_encode_batch([s])[0]

    assert ids_user == ids_batch, "Tokenizer 'user_way' vs 'encode_batch' mismatch."
    assert ids_user == ids_helper, "Tokenizer vs IDsPathHelper mismatch on single string."
    print("✅ Single-string equality OK.")

    idsA = _tok2ids_batch_encode(tok, seqs)
    idsB = _helper_safe_encode_batch(seqs)
    assert idsA == idsB, "Batch mismatch: tokenizer vs IDsPathHelper."
    print("✅ Batch equality OK.")

    # bytes→ids device path equality
    ids_dev_check = helper.ids_from_ascii_bytes_cuda(helper.encode_batch_to_ascii_bytes(seqs), DEVICE)
    assert torch.equal(ids_dev_check.cpu(), torch.tensor(idsB, dtype=torch.long)), \
        "bytes→ids (device) mismatch vs helper/tokenizer."
    print("✅ Device bytes→ids equality OK.")

    # ---------------------- performance primitives ---------------------

    total_tokens = B * T

    def time_encode(fn, label: str) -> Tuple[float, float]:
        t0 = time.perf_counter(); ids2d = fn(); t1 = time.perf_counter()
        enc_rate = total_tokens / max(1e-9, (t1 - t0))
        torch.cuda.synchronize()
        c0 = time.perf_counter(); x = as_cuda(ids2d, DEVICE); torch.cuda.synchronize(); c1 = time.perf_counter()
        copy_rate = total_tokens / max(1e-9, (c1 - c0))
        del x
        print(f"{label:28s} encode: {enc_rate:,.0f} tok/s | H2D: {copy_rate:,.0f} tok/s")
        return enc_rate, copy_rate

    def run_reps_encode(get_ids2d, label: str):
        enc_rates, h2d_rates = [], []
        for _ in range(REPS):
            er, hr = time_encode(get_ids2d, label)
            enc_rates.append(er); h2d_rates.append(hr)
        print(f"{label:28s} avg encode: {sum(enc_rates)/len(enc_rates):,.0f} tok/s | avg H2D: {sum(h2d_rates)/len(h2d_rates):,.0f} tok/s\n")
        return sum(enc_rates)/len(enc_rates), sum(h2d_rates)/len(h2d_rates)

    # Warm-up
    for _ in range(2):
        _ = _tok2ids_batch_encode(tok, seqs)
        _ = _helper_safe_encode_batch(seqs)

    # A′: user-way (robust per-string)
    def A_prime():
        return [_tok2ids_user_way(tok, s) for s in seqs]

    # A: encode_batch (fast path or robust fallback)
    def A_batch():
        return _tok2ids_batch_encode(tok, seqs)

    # B: IDsPathHelper classic path (int64, fresh pin each call)
    def B_helper_encode_only():
        return helper.encode_batch_to_ids(seqs)

    # B(staging,i64): persistent pinned staging (int64), no overlap
    def B_staging_i64():
        return helper.encode_batch_to_ids_staging(seqs, dtype=torch.int64)

    # B(staging,i32): persistent pinned staging (int32), no overlap
    def B_staging_i32():
        return helper.encode_batch_to_ids_staging(seqs, dtype=torch.int32)

    # ---------------- H2D-only comparison (i64 vs i32) -------------------

    def time_h2d_only(cpu_tensor: torch.Tensor, label: str) -> float:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        dev = cpu_tensor.to(DEVICE, non_blocking=True)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        rate = total_tokens / max(1e-9, (t1 - t0))
        del dev
        print(f"{label:28s} H2D-only: {rate:,.0f} tok/s")
        return rate

    # ---------------- End-to-end streaming measurements -----------------

    @torch.no_grad()
    def tokens_per_sec_stream_baseline(ids_cpu_long: torch.Tensor, emb_batch: int) -> float:
        # Non-overlap path
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _out in helper.iter_embed_tokens_in_slices(ids_cpu_long, emb_batch, device=DEVICE):
            pass
        torch.cuda.synchronize(); t1 = time.perf_counter()
        return total_tokens / max(1e-9, (t1 - t0))

    @torch.no_grad()
    def tokens_per_sec_stream_pipelined(ids_cpu_any: torch.Tensor, emb_batch: int) -> float:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _out in helper.iter_embed_tokens_pipelined(ids_cpu_any, emb_batch, device=DEVICE, use_int32_h2d=PREFER_INT32_H2D):
            pass
        torch.cuda.synchronize(); t1 = time.perf_counter()
        return total_tokens / max(1e-9, (t1 - t0))

    # ---------------- Run CPU encode + H2D microbench --------------------

    a1, a1c = run_reps_encode(A_prime,  "A′ user-way (robust)")
    a2, a2c = run_reps_encode(A_batch,  "A  encode_batch")
    b1, b1c = run_reps_encode(B_helper_encode_only, "B  IDsPathHelper (i64)")
    b2, b2c = run_reps_encode(B_staging_i64,       "B  staging i64 (pinned)")
    b3, b3c = run_reps_encode(B_staging_i32,       "B  staging i32 (pinned)")

    # Direct H2D-only comparison (best of REPS)
    i64 = helper.encode_batch_to_ids_staging(seqs, dtype=torch.int64)
    i32 = helper.encode_batch_to_ids_staging(seqs, dtype=torch.int32)
    h2d_i64_rates = [time_h2d_only(i64, "H2D int64")] + [time_h2d_only(i64, "H2D int64 (rep)") for _ in range(REPS-1)]
    h2d_i32_rates = [time_h2d_only(i32, "H2D int32")] + [time_h2d_only(i32, "H2D int32 (rep)") for _ in range(REPS-1)]
    best_i64 = max(h2d_i64_rates); best_i32 = max(h2d_i32_rates)
    print(f"\nBest H2D int64: {best_i64:,.0f} tok/s | Best H2D int32: {best_i32:,.0f} tok/s | Δ: {(best_i32/best_i64 - 1.0)*100:.1f}%")

    # ---------------- End-to-end streaming (overlap vs baseline) --------

    # Baseline requires LongTensor CPU ids
    ids_cpu_long = helper.encode_batch_to_ids_staging(seqs, dtype=torch.int64)
    ids_cpu_i32  = helper.encode_batch_to_ids_staging(seqs, dtype=torch.int32)

    EMB_BATCH = 2048  # tune for your model

    # Warmup
    _ = tokens_per_sec_stream_baseline(ids_cpu_long, EMB_BATCH)
    _ = tokens_per_sec_stream_pipelined(ids_cpu_i32 if PREFER_INT32_H2D else ids_cpu_long, EMB_BATCH)

    base_rates = [tokens_per_sec_stream_baseline(ids_cpu_long, EMB_BATCH) for _ in range(REPS)]
    pipe_rates = [tokens_per_sec_stream_pipelined(ids_cpu_i32 if PREFER_INT32_H2D else ids_cpu_long, EMB_BATCH) for _ in range(REPS)]

    base_avg = sum(base_rates)/len(base_rates)
    pipe_avg = sum(pipe_rates)/len(pipe_rates)

    print(f"\nStreaming (end-to-end) baseline: {base_avg:,.0f} tok/s")
    print(f"Streaming (end-to-end) pipelined: {pipe_avg:,.0f} tok/s")
    print(f"Streaming speedup: {(pipe_avg/base_avg - 1.0)*100:.1f}% (OVERLAP={'on' if OVERLAP_COPY_AND_COMPUTE else 'off'}, INT32_H2D={'on' if PREFER_INT32_H2D else 'off'})")

    # Keep prior final CUDA ids shape print for continuity
    final_cuda = as_cuda(ids_cpu_long, DEVICE)
    print("Final CUDA ids shape:", tuple(final_cuda.shape))
    del final_cuda

if __name__ == "__main__":
    main()
