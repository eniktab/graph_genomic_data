from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import numpy as np
import torch



@dataclass
class DNALUTEncoder:
    """
    Ultra-fast ASCII->token-id encoder with GPU reverse-complement support.

    Usage:
        backend = HyenaBackend()
        inv = backend.full_token_inventory()
        enc = DNALUTEncoder.from_backend(backend)
        ids = enc.encode_to_cuda_ids("ACGTNxxxacgtn")
        rc  = enc.reverse_complement_ids_cuda(ids, inv)
    """
    ascii_to_id_cpu: np.ndarray                  # (256,) int64
    _ascii_to_id_cuda: Optional[torch.Tensor]    = None
    _rc_id_lut_cuda:   Optional[torch.Tensor]    = None
    unk_id: int                                   = 0
    pad_id: Optional[int]                         = None
    vocab_size_hint: Optional[int]                = None

    # ------------------------- inventory handling -------------------------
    @staticmethod
    def _normalize_inventory(inv_any: Any) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int], int, Optional[int], int]:
        """
        Accepts either:
          - flat mapping: {'A':7, 'C':8, ...}
          - structured: {
                'token_to_id': {...},
                'id_to_token': {...},
                'specials': {'unk_id':6, 'pad_id':4, ...},
                'dna_ids': {'A':7,'C':8,'G':9,'T':10,'N':11,'a':6,'c':6,'g':6,'t':6,'n':6}
            }
        Returns:
          token_to_id, id_to_token, dna_ids, unk_id, pad_id, vocab_size_hint
        """
        if not isinstance(inv_any, dict) or not inv_any:
            raise ValueError("Inventory must be a non-empty dict")

        # Case 1: structured dict
        if "token_to_id" in inv_any or "id_to_token" in inv_any:
            token_to_id_raw = inv_any.get("token_to_id", {})
            id_to_token_raw = inv_any.get("id_to_token", {})
            specials = inv_any.get("specials", {})
            dna_ids_raw = inv_any.get("dna_ids", {})

            # Normalize types
            token_to_id = {str(k): int(v) for k, v in token_to_id_raw.items()}
            id_to_token = {int(k): str(v) for k, v in id_to_token_raw.items()}
            dna_ids = {str(k): int(v) for k, v in dna_ids_raw.items()}

            # Special ids
            unk_id = int(specials.get("unk_id")) if "unk_id" in specials else None
            pad_id = int(specials.get("pad_id")) if "pad_id" in specials else None

            # Fallbacks if specials not given
            if unk_id is None:
                for cand in ("[UNK]", "<unk>", "UNK", "unk", "X", "x"):
                    if cand in token_to_id:
                        unk_id = int(token_to_id[cand]); break
            if unk_id is None:
                raise ValueError("Could not infer unk_id from inventory")

            vocab_size_hint = 1 + max(token_to_id.values()) if token_to_id else (1 + max(id_to_token.keys()))

        # Case 2: flat dict token->id
        else:
            token_to_id = {str(k): int(v) for k, v in inv_any.items()}
            id_to_token = {int(v): str(k) for k, v in inv_any.items()}
            dna_ids = {}
            unk_id = None
            pad_id = None
            for cand in ("[UNK]", "<unk>", "UNK", "unk", "X", "x"):
                if cand in token_to_id: unk_id = int(token_to_id[cand]); break
            for cand in ("[PAD]", "<pad>", "PAD", "pad"):
                if cand in token_to_id: pad_id = int(token_to_id[cand]); break
            if unk_id is None:
                raise ValueError("Could not infer unk_id from flat inventory")
            vocab_size_hint = 1 + max(token_to_id.values())

        return token_to_id, id_to_token, dna_ids, unk_id, pad_id, vocab_size_hint

    @classmethod
    def from_backend(cls, backend) -> "DNALUTEncoder":
        inv_any = backend.full_token_inventory()
        token_to_id, id_to_token, dna_ids, unk_id, pad_id, vocab_size_hint = cls._normalize_inventory(inv_any)

        # Start with all unknowns
        lut = np.full(256, unk_id, dtype=np.int64)

        # If dna_ids provided, honor it *exactly* (e.g., lowercase mapping to unk)
        for ch, tid in dna_ids.items():
            if len(ch) == 1:
                lut[ord(ch)] = int(tid)

        # Otherwise, map canonical bases from token_to_id (upper)
        def set_if(sym: str):
            if sym in token_to_id:
                lut[ord(sym)] = int(token_to_id[sym])

        for base in ("A", "C", "G", "T", "N"):
            set_if(base)

        # If lowercase not explicitly set by dna_ids, mirror uppercase
        for upper, lower in (("A","a"), ("C","c"), ("G","g"), ("T","t"), ("N","n")):
            if dna_ids.get(lower) is None and upper in token_to_id:
                lut[ord(lower)] = int(token_to_id[upper])

        # Whitespace → pad (if available), else leave as unk
        if pad_id is not None:
            for ch in ("\n", "\r", "\t", " "):
                lut[ord(ch)] = pad_id

        enc = cls(
            ascii_to_id_cpu=lut,
            _ascii_to_id_cuda=None,
            _rc_id_lut_cuda=None,
            unk_id=unk_id,
            pad_id=pad_id,
            vocab_size_hint=vocab_size_hint,
        )
        # Prebuild RC LUT
        enc._materialize_rc_lut_cuda(token_to_id)
        return enc

    # ------------------------- encoding -------------------------
    def encode_to_cuda_ids(self, seq: str) -> torch.Tensor:
        u8 = np.frombuffer(seq.encode("ascii"), dtype=np.uint8, count=len(seq))
        ids_cpu = self.ascii_to_id_cpu[u8]                     # np.int64 [L]
        t_cpu = torch.from_numpy(ids_cpu).pin_memory()         # pinned host
        return t_cpu.to(device="cuda", non_blocking=True)      # long [L] on CUDA

    def encode_to_cuda_ids_gpu(self, seq: str) -> torch.Tensor:
        if self._ascii_to_id_cuda is None:
            self._ascii_to_id_cuda = torch.from_numpy(self.ascii_to_id_cpu).to("cuda", non_blocking=True)
        u8_cpu = np.frombuffer(seq.encode("ascii"), dtype=np.uint8, count=len(seq))
        u8_t = torch.from_numpy(u8_cpu).pin_memory().to("cuda", non_blocking=True)
        return self._ascii_to_id_cuda[u8_t.long()]

    # ------------------------- reverse complement -------------------------
    def _materialize_rc_lut_cuda(self, token_to_id: Dict[str, int]) -> None:
        vocab_n = self.vocab_size_hint or (1 + max(token_to_id.values()))
        rc = torch.arange(vocab_n, dtype=torch.long)

        def tok_id(name: str) -> Optional[int]:
            return int(token_to_id[name]) if name in token_to_id else None

        A, C, G, T, N = tok_id("A"), tok_id("C"), tok_id("G"), tok_id("T"), tok_id("N")
        if A is not None and T is not None:
            rc[A] = T; rc[T] = A
        if C is not None and G is not None:
            rc[C] = G; rc[G] = C
        if N is not None:
            rc[N] = N
        if self.pad_id is not None:
            rc[self.pad_id] = self.pad_id
        rc[self.unk_id] = self.unk_id

        self._rc_id_lut_cuda = rc.to("cuda", non_blocking=True)

    @torch.inference_mode()
    def reverse_complement_ids_cuda(self, ids: torch.Tensor, inv_any: Any) -> torch.Tensor:
        if ids.device.type != "cuda":
            raise ValueError("expected ids on CUDA")
        if ids.dtype != torch.long:
            ids = ids.long()

        # Ensure RC LUT exists and is large enough
        if self._rc_id_lut_cuda is None:
            token_to_id, *_ = self._normalize_inventory(inv_any)
            self._materialize_rc_lut_cuda(token_to_id)

        max_id = int(ids.max().item()) if ids.numel() else 0
        if max_id >= self._rc_id_lut_cuda.numel():
            token_to_id, id_to_token, dna_ids, unk_id, pad_id, vocab_size_hint = self._normalize_inventory(inv_any)
            self.vocab_size_hint = max_id + 1
            self._materialize_rc_lut_cuda(token_to_id)

        comp = self._rc_id_lut_cuda[ids]
        return comp.flip(0)              # reverse