from typing import List, Optional

# ============================================================
# 1) Nucleotide Transformer backend
# ============================================================
class NTBackend:
    """
    GPU embedding backend for Nucleotide Transformer.

    - Expects each input sequence to be ≤ 5,994 bp (no tiling here).
    - Float32 pooling (mean or GeM) over last hidden states.
    - Returns L2-normalized embeddings by default.
    """

    MAX_TILE_BP = 5994  # strict hard cap

    def __init__(self,
                 model_name: Optional[str] = None,
                 model_dir: Optional[str] = None,
                 max_len_tokens: Optional[int] = None,  # tokenizer budget; capped at 1000
                 pooling: str = "gem",                  # "mean" or "gem"
                 offline: bool = True,
                 local_files_only: bool = True,
                 device_map: Optional[str] = "auto"):
        import os, torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        from torch.nn.attention import sdpa_kernel, SDPBackend

        if offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required for NTBackend")

        torch.backends.cuda.matmul.allow_tf32 = True
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass
        try: sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
        except Exception: pass

        torch.set_default_device("cuda")
        self.torch = torch
        self.pooling = pooling

        use_bf16 = torch.cuda.is_bf16_supported()
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16

        # Load tokenizer/model
        model_ref = model_dir or model_name
        if model_ref is None:
            raise ValueError("Provide model_name or model_dir.")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_ref, local_files_only=local_files_only, trust_remote_code=True
        )
        model = AutoModelForMaskedLM.from_pretrained(
            model_ref,
            local_files_only=local_files_only,
            dtype=self.dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()
        try: model = torch.compile(model, mode="max-autotune")
        except Exception: pass
        self.model = model

        # Token budget for encode(); keep ≤ 1000
        tok_max = getattr(self.tokenizer, "model_max_length", 1000)
        self.max_len_tokens = int(min(max_len_tokens or tok_max, 1000))

    # ---------- helpers ----------
    def _assert_len_ok(self, seqs: List[str]) -> None:
        bad = [i for i, s in enumerate(seqs) if len(s.strip()) > self.MAX_TILE_BP]
        if bad:
            raise ValueError(f"Sequences must be ≤ {self.MAX_TILE_BP} bp. Offending indices (first 5): {bad[:5]}")

    def _pool_masked(self, token_emb, attention_mask, eps=1e-6):
        """
        token_emb: [B, T, H] (bf16/fp16), attention_mask: [B, T]
        returns:   [B, H] float32
        """
        torch = self.torch
        token_emb = token_emb.to(torch.float32)
        maskf = attention_mask.to(torch.float32).unsqueeze(-1)

        if self.pooling == "mean":
            num = (token_emb * maskf).sum(dim=1)
            den = maskf.sum(dim=1).clamp_min(1.0)
            out = num / den
        elif self.pooling == "gem":
            p_val = 3.0
            x = (token_emb.clamp_min(eps) * maskf).pow(p_val).sum(dim=1)
            den = maskf.sum(dim=1).clamp_min(1.0)
            out = (x / den).clamp_min(eps).pow(1.0 / p_val)
        else:
            raise ValueError("pooling must be 'mean' or 'gem'")
        return out

    # ---------- public ----------
    def embed_list(self, seqs: List[str], normalize: bool = False):
        """
        Embed a list of sequences (each ≤ 5,994 bp). No batching inside.
        Returns [N, H] float32 CUDA (L2-normalized if normalize=True).
        """
        torch = self.torch
        self._assert_len_ok(seqs)

        batch = self.tokenizer(
            [s.strip().upper() for s in seqs],
            return_tensors="pt",
            padding="max_length",
            pad_to_multiple_of=8,
            truncation=True,
            max_length=self.max_len_tokens,
        )
        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)

        with torch.inference_mode():
            outs = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )
            token_emb = outs.hidden_states[-1]  # [B, T, H]
            token_emb = token_emb.to(torch.float32)

        emb = self._pool_masked(token_emb, attn_mask)  # [B, H] float32
        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=1)
        return emb
