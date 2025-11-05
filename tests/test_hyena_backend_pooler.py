import os
import copy
from pathlib import Path
from typing import List

import pytest
import torch
import src.HyenaBackend as m

from src.HyenaBackend import HyenaBackend, HyenaDNAPooler  # <-- adjust module path if needed


# -----------------------
# Config (update paths)
# -----------------------
OFFLINE_PATH = Path("/g/data/te53/en9803/data/scratch/hf-cache/models")
MODEL_NAME = "hyenadna-large-1m-seqlen-hf"

DEVICE_STR = "cuda"
DEVICE = torch.device(DEVICE_STR) if torch.cuda.is_available() else torch.device("cpu")


# -----------------------
# Session-wide guards
# -----------------------
@pytest.fixture(scope="session")
def model_dir():
    d = OFFLINE_PATH / MODEL_NAME
    if not d.exists():
        pytest.skip(f"Model snapshot not found at: {d}")
    return d


@pytest.fixture(scope="session")
def loaded_wrapper_and_cfg(model_dir):
    """
    Load HyenaDNAHF.from_pretrained using your local snapshot,
    but inject a deterministic ctor so we can compare vs an
    identically-initialized scratch model (to prove grafting happened).
    """
    captured = {}

    def ctor_with_seed(**cfg):
        captured["cfg"] = copy.deepcopy(cfg)
        torch.manual_seed(1337)
        return m.HyenaDNAModel(**cfg)

    wrapper = m.HyenaDNAHF.from_pretrained(
        path=str(model_dir.parent),
        model_name=model_dir.name,
        device=DEVICE,
        use_head=False,
        model_ctor=ctor_with_seed,
        allow_download=False,
        strict_backbone=False,
        verbose=False,
    )
    # Ensure hidden states are returned downstream
    if hasattr(wrapper, "config"):
        wrapper.config.output_hidden_states = True
        wrapper.config.use_return_dict = True
    wrapper.eval()
    return wrapper, captured["cfg"]


@pytest.fixture(scope="session")
def backend(loaded_wrapper_and_cfg, model_dir):
    """
    Construct HyenaBackend and graft the deterministically-loaded wrapper into it.
    This ensures the backend uses your local snapshot and never passes attention_mask.
    """
    wrapper, _cfg = loaded_wrapper_and_cfg

    # Build backend (it will try to load, but we'll graft immediately after)
    be = HyenaBackend(
        model_name=model_dir.name,
        model_dir=str(model_dir.parent),
        prefer_cuda=(DEVICE.type == "cuda"),
        offline=True,
        pooling="mean",
        rc_invariant=True,
    )
    # Graft the preloaded wrapper (deterministic init) and ensure eval/hidden states
    be.model = wrapper
    be.model.eval()
    if hasattr(be.model, "config"):
        be.model.config.output_hidden_states = True
        be.model.config.use_return_dict = True

    # Keep tests snappy by capping max_length for tokenization
    be.max_length = min(be.max_length, 1024)

    # Align backend device bookkeeping with the wrapper device
    be.device = DEVICE
    if DEVICE.type == "cuda":
        be.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        be.torch_dtype = torch.float32

    return be


@pytest.fixture
def seqs() -> List[str]:
    # Short sequences to keep forward passes light; include palindromic RC case
    return [
        "ACGT",                          # RC-palindrome → ACGT
        "ACGTAC",                        # 6
        "NNNNACGT",                      # 8 with Ns
        "TTTTTT",                        # 6
        "AC"*10 + "GT"*5 + "N"           # 41
    ]


def _pick_divisor(h: int):
    for k in (64, 32, 16, 8, 4, 2):
        if h % k == 0:
            return k
    return None


def test_pooler_position_basic(backend: HyenaBackend, seqs):
    pooler = backend.build_pooler(direction="exp_left", tau=64.0, pooling_axis="position", layer_spec=-7, rc_average=False)
    X = pooler.embed(seqs)
    assert X.shape[0] == len(seqs)
    # L2 normalized rows
    norms = torch.linalg.vector_norm(X, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_pooler_rc_average_toggles_output(backend: HyenaBackend, seqs):
    pooler = backend.build_pooler(direction="exp_left", tau=64.0, pooling_axis="position", layer_spec=-7, rc_average=False)
    X = pooler.embed(seqs)
    pooler.set_config(rc_average=True)
    X_rc = pooler.embed(seqs)
    # Expect a difference when sequences are not globally RC-palindromic
    assert not torch.allclose(X, X_rc)


def test_pooler_layers_to_position(backend: HyenaBackend, seqs):
    pooler = backend.build_pooler(direction="exp_left", tau=64.0,
                                  pooling_axis="layers→position", layer_spec=("last_k", 2), rc_average=False)
    X = pooler.embed(seqs)
    assert X.ndim == 2 and X.shape[0] == len(seqs)


def test_pooler_position_to_channels_dim_reduction(backend: HyenaBackend, seqs):
    # Pick a divisor K of the hidden size
    # (Probe hidden size from a tiny forward)
    hs, _mask = backend.embed_list([seqs[0]], pooling="none", rc_invariant=False)
    H = hs.shape[-1]
    K = _pick_divisor(H)
    if K is None or K < 2:
        pytest.skip(f"No suitable channel_groups divisor for H={H}")
    pooler = backend.build_pooler(direction="exp_left", tau=64.0,
                                  pooling_axis="position→channels", channel_groups=K, layer_spec=-7, rc_average=False)
    X = pooler.embed(seqs)
    assert X.shape[1] == K
    norms = torch.linalg.vector_norm(X, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_pooler_channel_groups_divisibility_error(backend: HyenaBackend, seqs):
    hs, _mask = backend.embed_list([seqs[0]], pooling="none", rc_invariant=False)
    H = hs.shape[-1]
    # Choose a bad K that doesn't divide H (find first odd K >= 3 that fails)
    bad_K = None
    for k in range(3, 9, 2):
        if H % k != 0:
            bad_K = k
            break
    if bad_K is None:
        pytest.skip("Could not construct a non-divisor K; skipping.")
    pooler = backend.build_pooler(direction="exp_left", tau=64.0,
                                  pooling_axis="position→channels", channel_groups=bad_K, layer_spec=-7, rc_average=False)
    with pytest.raises(ValueError):
        _ = pooler.embed(seqs)


def test_pooler_from_preset(backend: HyenaBackend, seqs):
    pooler = HyenaDNAPooler.from_preset(backend, "cluster_max_sep")
    X = pooler.embed(seqs[:2])
    assert X.shape[0] == 2
    cfg = pooler.get_config()
    assert cfg["direction"] in {"exp_left", "exp_right", "mean"}


def test_pooler_auto_select_small_sample(backend: HyenaBackend, seqs, capsys):
    auto = HyenaDNAPooler(backend, auto_select=True, auto_seqs=seqs[:4], auto_max_seqs=4, auto_verbose=True)
    X = auto.embed(seqs[:3])
    assert X.shape[0] == 3
    cfg = auto.get_config()
    assert isinstance(cfg, dict) and "direction" in cfg
    _ = capsys.readouterr()  # ensure no crashes when printing