# --------------------------------------------
# DeterministicBackend: stable k-mer sketching (collision-hardened)
# (CUDA-compliant: embed_best returns CUDA float32 [B, D])
# --------------------------------------------
from __future__ import annotations
from typing import List, Optional, Tuple
import torch

_RC_MAP = str.maketrans({"A": "T", "C": "G", "G": "C", "T": "A"})


def revcomp(seq: str) -> str:
    return seq.translate(_RC_MAP)[::-1]


def _mutate(s: str, i: int) -> str:
    comp = {"A": "C", "C": "G", "G": "T", "T": "A"}
    s = list(s)
    s[i] = comp.get(s[i], "A")
    return "".join(s)


def _non_rc_symmetric_seq(L: int = 10_000) -> str:
    s = ("ACGT" * ((L // 4) + 2))[:L]
    s = "A" + s[1:-1] + "C"
    if revcomp(s) == s:
        s = "G" + s[1:-1] + "T"
    return s


class DeterministicBackend:
    """
    Deterministic, model-free DNA embedding.

    Returns **CUDA** float32 tensors to satisfy HELIndexer._first_dim_probe.

    - Encodes A,C,G,T -> 0..3, streams a rolling k-mer code.
    - Canonical k-mers if rc_merge=True (min(code, rc_code)).
    - CountSketch each (k-mer, coarse-pos-bucket, micro-pos) into the slice
      dedicated to that coarse position bucket with ±1 signed updates.
    - Per-slice companding (signed |x|^gamma) and centering (optional),
      then slice-wise L2 normalization, then optional global L2 normalization.

    If rc_merge=False, we rotate within-slice by half the slice length to make
    strands distinguishable.
    """

    # constants
    _MASK64 = 0xFFFFFFFFFFFFFFFF
    _GOLDEN = 0x9E3779B97F4A7C15
    _MICRO_SALT = 0x9FB21C651E98DF25

    # ---- init ----------------------------------------------------------------
    def __init__(
        self,
        dim: int = 1024,
        k: int = 7,
        rc_merge: bool = True,
        normalize: bool = True,
        *,
        max_length: int = 1_000_000,
        pos_buckets: int = 64,
        num_hashes: int = 2,
        micro_buckets: int = 64,
        power_gamma: Optional[float] = 0.5,   # signed |x|^gamma per-slice
        center_slices: bool = True,           # mean-center per slice before L2
        device: Optional[torch.device | str] = "cuda",
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be > 0")
        if not (1 <= k <= 15):
            raise ValueError("k must be in [1, 15]")
        if pos_buckets <= 0:
            raise ValueError("pos_buckets must be > 0")
        if num_hashes <= 0:
            raise ValueError("num_hashes must be >= 1")
        if micro_buckets <= 0:
            raise ValueError("micro_buckets must be > 0")
        if power_gamma is not None and not (0.0 < power_gamma <= 1.0):
            raise ValueError("power_gamma must be in (0, 1] or None")

        # --- device: HELIndexer expects CUDA tensors from embed_best ----------
        if device is None:
            device = "cuda"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DeterministicBackend requires CUDA, but no GPU is available.")
        if self.device.type != "cuda":
            # HELIndexer._first_dim_probe requires x.is_cuda == True
            raise RuntimeError("DeterministicBackend must run on CUDA (device='cuda').")

        self.dim = int(dim)
        self.k = int(k)
        self.rc_merge = bool(rc_merge)
        self.normalize = bool(normalize)
        self.max_length = int(max_length)
        self.pos_buckets = int(pos_buckets)
        self.num_hashes = int(num_hashes)
        self.micro_buckets = int(micro_buckets)
        self.power_gamma = power_gamma
        self.center_slices = bool(center_slices)

        # Rolling k-mer params
        self._KMOD = 1 << (2 * self.k)
        self._KMASK = self._KMOD - 1

        # seeds for multi-hash
        self._seeds = (
            0x243F6A8885A308D3,
            0x13198A2E03707344,
            0xA4093822299F31D0,
            0x082EFA98EC4E6C89,
        )

        # base map
        self._map = {-1: -1, ord("A"): 0, ord("C"): 1, ord("G"): 2, ord("T"): 3}

        # precompute slice bounds for speed
        self._slice_bounds_cache: List[Tuple[int, int]] = []
        for p in range(self.pos_buckets):
            start = (p * self.dim) // self.pos_buckets
            end = ((p + 1) * self.dim) // self.pos_buckets
            self._slice_bounds_cache.append((start, end))

    def fingerprint(self) -> dict:
        return {
            "type": "DeterministicBackend",
            "dim": self.dim,
            "k": self.k,
            "rc_merge": self.rc_merge,
            "normalize": self.normalize,
            "pos_buckets": self.pos_buckets,
            "num_hashes": self.num_hashes,
            "micro_buckets": self.micro_buckets,
            "power_gamma": self.power_gamma,
            "center_slices": self.center_slices,
            "max_length": self.max_length,
            "device": str(self.device),
        }

    # ---- utils ---------------------------------------------------------------
    @staticmethod
    def _splitmix64(x: int) -> int:
        x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = x
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        z ^= (z >> 31)
        return z & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def _l2_normalize(vec: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        denom = torch.linalg.vector_norm(vec).clamp_min(eps)
        return vec / denom

    def _char_code(self, ch: str) -> int:
        return self._map.get(ord(ch), -1)

    def _roll_codes(self, seq: str):
        k = self.k
        mask = self._KMASK
        code = 0
        have = 0
        for i, ch in enumerate(seq):
            v = self._char_code(ch)
            if v < 0:
                code = 0
                have = 0
                continue
            code = ((code << 2) | v) & mask
            have = min(have + 1, k)
            if have == k:
                yield code, i  # k-mer ends at i

    def _revcomp_code(self, code: int) -> int:
        rc = 0
        for _ in range(self.k):
            two = code & 0b11
            rc = (rc << 2) | (two ^ 0b11)
            code >>= 2
        return rc & self._KMASK

    def _pos_bucket(self, i_end: int, L: int) -> int:
        if L <= 0 or self.pos_buckets <= 1:
            return 0
        b = min(self.pos_buckets - 1, ((i_end + 1) * self.pos_buckets) // L)
        if self.rc_merge:
            b = min(b, (self.pos_buckets - 1) - b)  # strand-invariant
        return b

    def _slice_bounds(self, pbin: int) -> Tuple[int, int]:
        return self._slice_bounds_cache[pbin]

    def _micro_pos(self, i_end: int, L: int) -> int:
        """
        RC-invariant fine position: micro = floor(min(i, L-1-i) * micro_buckets / ceil(L/2)).
        Ensures micro(i) == micro(L-1-i).
        """
        if L <= 1:
            return 0
        half_span = (L + 1) // 2  # ceil(L/2)
        mirrored = i_end if i_end <= (L - 1 - i_end) else (L - 1 - i_end)
        micro = (mirrored * self.micro_buckets) // half_span
        if micro >= self.micro_buckets:
            micro = self.micro_buckets - 1
        return micro

    def _bucket_and_sign_in_slice(
        self, key: int, j: int, start: int, end: int
    ) -> Tuple[int, float]:
        h = self._splitmix64((key ^ self._seeds[j % len(self._seeds)]) & self._MASK64)
        span = max(1, end - start)
        bucket = int(start + (h % span))
        sign = 1.0 if (h & 0x8000_0000_0000_0000) == 0 else -1.0
        return bucket, sign

    @staticmethod
    def _orient_bit(seq: str) -> int:
        rc = revcomp(seq)
        return 1 if seq > rc else 0  # ties -> 0

    # ---- public API ----------------------------------------------------------
    def embed_one(self, seq: str) -> torch.Tensor:
        """Embed one sequence -> CUDA float32 vector [D]."""
        if len(seq) > self.max_length:
            seq = seq[: self.max_length]

        vec = torch.zeros(self.dim, dtype=torch.float32, device=self.device)
        L = len(seq)
        if L < self.k:
            return vec

        # per-seq strand bit (when strand should matter)
        orient = self._orient_bit(seq) if not self.rc_merge else 0

        for code_fwd, i_end in self._roll_codes(seq):
            # canonicalize if requested
            code = (
                min(code_fwd, self._revcomp_code(code_fwd))
                if self.rc_merge
                else code_fwd
            )

            pbin = self._pos_bucket(i_end, L)
            start, end = self._slice_bounds(pbin)
            slice_len = max(1, end - start)

            # within-slice half-rotation for strand (rc_merge=False)
            strand_shift = (slice_len // 2) if (not self.rc_merge and orient) else 0

            # RC-invariant micro position
            micro = self._micro_pos(i_end, L)

            # deterministic composite key
            key = (
                code
                ^ ((pbin * self._GOLDEN) & self._MASK64)
                ^ ((micro * self._MICRO_SALT) & self._MASK64)
            ) & self._MASK64

            for j in range(self.num_hashes):
                b, sgn = self._bucket_and_sign_in_slice(key, j, start, end)
                if strand_shift:
                    b = start + ((b - start + strand_shift) % slice_len)
                vec[b] += sgn

        # --- per-slice shaping & normalization ---
        for p in range(self.pos_buckets):
            s, e = self._slice_bounds(p)
            if e <= s:
                continue
            seg = vec[s:e]

            # center slice
            if self.center_slices:
                seg = seg - seg.mean()

            # power-law companding
            if self.power_gamma is not None:
                seg = torch.sign(seg) * torch.pow(seg.abs(), self.power_gamma)

            # L2 normalize slice
            n = torch.linalg.vector_norm(seg)
            if n > 0:
                vec[s:e] = seg / n
            else:
                vec[s:e] = seg

        # global normalization
        if self.normalize:
            vec = self._l2_normalize(vec)

        return vec

    def embed_list(
        self,
        seqs: List[str],
        normalize: Optional[bool] = None,
        rc_invariant: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Batch embed -> CUDA float32 tensor [B, D].
        If rc_invariant is not None, temporarily override self.rc_merge for this call.
        """
        old_norm = self.normalize
        old_rc_merge = self.rc_merge
        if normalize is not None:
            self.normalize = bool(normalize)
        if rc_invariant is not None:
            self.rc_merge = bool(rc_invariant)
        try:
            out = torch.stack([self.embed_one(s) for s in seqs], dim=0)
        finally:
            # restore flags
            self.normalize = old_norm
            self.rc_merge = old_rc_merge
        # ensure CUDA (HELIndexer requires x.is_cuda)
        if not out.is_cuda:
            out = out.to(self.device, dtype=torch.float32, non_blocking=True)
        return out

    def embed_best(
        self,
        seqs: List[str],
        rc_invariant: Optional[bool] = None,
        **_: object,  # ignore future kwargs from callers
    ) -> torch.Tensor:
        """
        Preferred entrypoint for HELIndexer. Must return CUDA [B, D].
        rc_invariant overrides strand handling for this call:
          - True  -> reverse-complement invariant (canonicalized)
          - False -> strand-aware (no canonicalization)
          - None  -> use self.rc_merge
        """
        x = self.embed_list(seqs, rc_invariant=rc_invariant)
        if not (isinstance(x, torch.Tensor) and x.is_cuda and x.ndim == 2):
            raise TypeError("DeterministicBackend.embed_best must return CUDA tensor [B, D]")
        return x

    @property
    def tokenizer(self):
        return None
