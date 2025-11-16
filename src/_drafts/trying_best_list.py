import torch
from src.HyenaBackend import HyenaBackend
from src.configs import IndexConfig
cfg = IndexConfig(window=10000, stride=5000)

backend1 = HyenaBackend(
                    model_name=getattr(cfg, "model_name", None),
                    model_dir=getattr(cfg, "model_dir", None),
                    pooling="mean",
                    normalize=False,    # we normalize here if cosine
                    offline=True,
                    prefer_cuda=True,
                )

backend2 = HyenaBackend(
                    model_name=getattr(cfg, "model_name", None),
                    model_dir=getattr(cfg, "model_dir", None),
                    pooling="mean",
                    normalize=True,    # we normalize here if cosine
                    offline=True,
                    prefer_cuda=True,
                )

# Build a test sequence of length exactly equal to the model max (no EOS, left pad if needed)
sequence = "ACTG" * (backend1.max_length // 4)
tok = backend1.tokenizer(sequence, add_special_tokens=False)
ids = torch.tensor(tok["input_ids"], dtype=torch.long).unsqueeze(0).to(backend1.device)  # [1,T]
# Non-streamed pooled (reference)
e_ref = backend1.embed_tokens(ids, pooling="mean", rc_invariant=False)        # [1,D]

# Build a test sequence of length exactly equal to the model max (no EOS, left pad if needed)
sequence = "ACTG" * (backend1.max_length // 4)
tok = backend2.tokenizer(sequence, add_special_tokens=False)
ids = torch.tensor(tok["input_ids"], dtype=torch.long).unsqueeze(0).to(backend2.device)  # [1,T]
# Non-streamed pooled (reference)
e_ref2 = backend2.embed_tokens(ids, pooling="mean", rc_invariant=False)        # [1,D]



# Streamed pooled (identical values)
streamed = backend1.embed_tokens_streaming(
    batch_iter=[ids],        # any Iterable[LongTensor[B,T]] works
    pooling="mean",
    rc_invariant=False,
    out_device=backend1.device,
    cat=True,                # return a single [N,D] instead of list
)
assert torch.allclose(e_ref, streamed, atol=0, rtol=0)
print(streamed.shape)  # [1, D]
