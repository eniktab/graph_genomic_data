from src.HyenaBackend import HyenaDNAHF
import safetensors.torch as st
from pathlib import Path


_ = HyenaDNAHF.from_pretrained(
    path="/g/data/te53/en9803/data/scratch/hf-cache/models",
    model_name="hyenadna-large-1m-seqlen-hf",
    device="cpu",
    strict_backbone=True,   # <- prints misses (first 20) then raises
    verbose=True,
)


wt = st.load_file(Path("/g/data/te53/en9803/data/scratch/hf-cache/models/hyenadna-large-1m-seqlen-hf/model.safetensors"))
keys = list(wt.keys())

# look for the four patterns that were missing
for pat in [
    "mixer.filter_fn.pos_emb.z",
    "mixer.filter_fn.implicit_filter.0.weight",
    "mixer.filter_fn.implicit_filter.3.freq",
    "mixer.filter_fn.implicit_filter.5.freq",
]:
    any_hit = any(pat in k for k in keys)
    print(f"{pat:55} -> {'present' if any_hit else 'absent'}")