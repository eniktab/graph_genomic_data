# tests/test_hyenadna_offline.py
# ------------------------------------------------------------
# Offline-only tests for your HyenaDNA stack.
# Uses your local HF snapshot at HYENA_MODEL_PATH/HYENA_MODEL_NAME.
# ------------------------------------------------------------
import copy
import math
import importlib
from pathlib import Path
import random
import torch
import torch.nn.functional as F
import pytest

from src.HyenaBackend import HyenaBackend
from src.CudaWindower import CudaWindower
from src.DNALUTEncoder import DNALUTEncoder
from src.configs import IndexConfig

cfg = IndexConfig()

# --- Global test settings ---
RTOL = 5e-3  # allow bf16 differences
ATOL = 5e-3

CUDA_REQ = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for LUT+ids path")
# -----------------------
# Environment wiring
# -----------------------

OFFLINE_PATH = Path("/g/data/te53/en9803/data/scratch/hf-cache/models")
MODEL_NAME = "hyenadna-large-1m-seqlen-hf"
DEVICE_STR = ("cuda")
DEVICE = torch.device(DEVICE_STR)


# -----------------------
# Import your module
# -----------------------
m = None
errors = []
if m is None:
    # Fallback: load from file path <repo_root>/src/HyenaBackend.py
    root = Path(__file__).resolve().parents[1]
    candidate = root / "src" / "HyenaBackend.py"
    if candidate.exists():
        spec = importlib.util.spec_from_file_location("hyena_backend_local", candidate)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        MODNAME = str(candidate)
    else:
        tried = ", ".join(n for n, _ in errors if n)
        raise ImportError(
            "Could not import HyenaBackend module. Tried: "
            + (tried or "<none>")
            + f". Also looked for file at {candidate}"
        )

# Required symbols must exist in the loaded module
for sym in ["CharacterTokenizer", "HyenaDNAHF", "HyenaDNAModel", "HyenaDNAPooler"]:
    assert hasattr(m, sym), f"{sym} not found in module {MODNAME}"


# -----------------------
# Local repo sanity
# -----------------------
def _find_files(model_dir: Path):
    cfg = None
    for name in ("config.json", "hf_config.json"):
        p = model_dir / name
        if p.is_file():
            cfg = p
            break

    weights = None
    for name in ("model.safetensors", "pytorch_model.bin", "weights.ckpt"):
        p = model_dir / name
        if p.is_file():
            weights = p
            break
    return cfg, weights


@pytest.fixture(scope="session")
def model_dir():
    d = OFFLINE_PATH / MODEL_NAME
    cfg, w = _find_files(d)
    if not (cfg and w):
        pytest.skip(f"Missing config/weights in {d}. Found: config={cfg}, weights={w}")
    return d


@pytest.fixture(scope="session")
def tokenizer():
    return m.CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],
        model_max_length=4096,
        padding_side='left',
        default_add_special_tokens=False,
    )


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
    return wrapper, captured["cfg"]


@pytest.fixture(scope="session")
def scratch_same_init(loaded_wrapper_and_cfg):
    _, cfg = loaded_wrapper_and_cfg
    torch.manual_seed(1337)  # same seed as ctor_with_seed
    model = m.HyenaDNAModel(**cfg)
    model.to(DEVICE)
    model.eval()
    return model


# -----------------------
# Tests (original + extended)
# -----------------------
def test_offline_files_present(model_dir):
    cfg, w = _find_files(model_dir)
    assert cfg and w, "config.json / weights file must exist for offline loading"


def _load_pretrained_sd(model_dir: Path, m):
    cfg, w = _find_files(model_dir)
    assert w is not None
    return m.HyenaDNAHF._load_pretrained_dict(w, map_location=DEVICE)

ROOT_PREFIXES = ["", "model.", "hyena.", "hyena.model.", "hyenadna.", "hyenadna.model."]

def _candidates_for(scratch_key: str) -> list[str]:
    """
    Generate likely pretrained key variants for a given scratch key.
    Includes hyena./hyenadna. roots and checkpointing .layer variants.
    """
    def inject_layer(k: str) -> str:
        return k.replace(".mixer.", ".mixer.layer.").replace(".mlp.", ".mlp.layer.")

    bases = {scratch_key}

    # Allow snapshots that drop or duplicate 'backbone.'
    if scratch_key.startswith("backbone."):
        tail = scratch_key[len("backbone."):]    # e.g. 'layers.0.mixer.in_proj.weight'
        bases |= {tail, f"backbone.{tail}"}

    # Checkpointing name deltas
    bases |= {inject_layer(b) for b in list(bases)}

    # Embedding / final norm synonyms
    if scratch_key.endswith("embeddings.word_embeddings.weight"):
        bases |= {
            "embed_tokens.weight", "wte.weight",
            "backbone.embed_tokens.weight", "backbone.wte.weight",
        }
    if scratch_key.endswith("ln_f.weight"):
        bases |= {"final_layer_norm.weight", "final_layernorm.weight"}
    if scratch_key.endswith("ln_f.bias"):
        bases |= {"final_layer_norm.bias", "final_layernorm.bias"}

    # Cross with root prefixes
    out = []
    for base in bases:
        for root in ROOT_PREFIXES:
            out.append(f"{root}{base}")
    return out

def test_weights_grafted_vs_scratch(loaded_wrapper_and_cfg, scratch_same_init, model_dir):
    """
    Prove that disk weights were applied, but be robust to different naming schemes.
    - If the on-disk checkpoint exposes *no* plausible matches to our backbone keys,
      SKIP with a diagnostic (mapping update needed).
    - If there *are* plausible matches but the loaded == scratch, FAIL with a clear hint.
    """
    wrapper, _ = loaded_wrapper_and_cfg
    loaded = wrapper.model
    scratch = scratch_same_init

    loaded_sd = dict(loaded.state_dict())
    scratch_sd = dict(scratch.state_dict())

    # 1) Discover plausible matches against the actual checkpoint
    pretrained_sd = _load_pretrained_sd(model_dir, m)
    pt_keys = set(pretrained_sd.keys())

    # only consider backbone tensors that exist in scratch too
    scratch_backbone_keys = [
        k for k in loaded_sd
        if k.startswith("backbone") and k in scratch_sd and loaded_sd[k].shape == scratch_sd[k].shape
    ]
    assert scratch_backbone_keys, "No comparable backbone parameters found."

    matches: list[str] = []
    for k in scratch_backbone_keys:
        cands = _candidates_for(k)
        if any(c in pt_keys for c in cands):
            matches.append(k)

    if not matches:
        # Nothing in your snapshot looks mappable to our backbone → skip with guidance
        sample = sorted(list(pt_keys))[:30]
        pytest.skip(
            "No plausible mapping from scratch backbone keys to your checkpoint keys. "
            "Update HyenaDNAHF._graft_backbone to map your snapshot’s naming.\n"
            f"Example backbone key: {scratch_backbone_keys[0]!r}\n"
            f"First 30 checkpoint keys:\n- " + "\n- ".join(sample)
        )

    # 2) There ARE plausible matches. Now the loaded model should differ from a same-seed scratch.
    diffs = []
    for k in matches:
        a = loaded_sd[k].detach().float().cpu()
        b = scratch_sd[k].detach().float().cpu()
        diffs.append(float(torch.norm(a - b).item()))

    changed_frac = sum(1 for d in diffs if d > 1e-8) / max(1, len(diffs))

    assert changed_frac > 0.10, (
        f"Found {len(matches)} backbone params with plausible checkpoint matches, "
        f"but only {changed_frac:.2%} differ from scratch.\n"
        "Likely cause: HyenaDNAHF._graft_backbone didn’t actually map those keys. "
        "Extend the mapping to include your snapshot’s prefixes/synonyms "
        "(e.g., 'model.', 'embed_tokens'/ 'wte', '.mixer.layer', '.mlp.layer')."
    )

    # Bonus: if embedding key is among matches, require it to change.
    emb_key = "backbone.embeddings.word_embeddings.weight"
    if emb_key in matches:
        delta = torch.norm(
            loaded_sd[emb_key].detach().float().cpu() - scratch_sd[emb_key].detach().float().cpu()
        ).item()
        assert delta > 1e-6, "Token embedding unchanged vs scratch; expected checkpoint to overwrite it."

def test_print_embeddings_loaded_vs_scratch(loaded_wrapper_and_cfg, scratch_same_init, tokenizer):
    """
    Print pooled embeddings for the same sequences using:
      - loaded (grafted) model
      - scratch (random-init) model
    And assert they differ (sanity that weights matter).
    """
    import torch.nn as nn
    from torch.nn import functional as F

    wrapper, _ = loaded_wrapper_and_cfg
    scratch = scratch_same_init

    # Heuristic skip for very large checkpoints on CPU
    d_model = int(wrapper.model.backbone.ln_f.normalized_shape[0])
    n_layer = len(wrapper.model.backbone.layers)
    if DEVICE.type == "cpu" and (d_model * n_layer > 8192):
        pytest.skip(f"Model too large for CPU quick-test (d_model={d_model}, n_layer={n_layer}). "
                    f"Run on GPU or use a smaller model.")

    # Minimal backend that HyenaDNAPooler expects
    class _Backend:
        def __init__(self, model_like, tok, max_len=256):
            self.model = model_like
            # Ensure HF-like return shape (hidden states) for wrappers that support it
            if hasattr(self.model, "config"):
                try:
                    self.model.config.output_hidden_states = True
                    self.model.config.use_return_dict = True
                except Exception:
                    pass
            self.tokenizer = tok
            self.device = DEVICE
            self.max_length = max_len

        @staticmethod
        def _choose_pad_multiple(L: int):
            for mval in (128, 64, 32, 16, 8, 4):
                if L % mval == 0:
                    return mval
            return None

    # Wrap scratch so it always returns a BaseModelOutput-like object
    class _WrapScratch(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, input_ids, **kwargs):
            # Force HF-style output for pooler
            return self.base(input_ids, output_hidden_states=True, return_dict=True)

    backend_loaded = _Backend(wrapper, tokenizer, max_len=256)
    backend_scratch = _Backend(_WrapScratch(scratch), tokenizer, max_len=256)

    pool_cfg = dict(direction="exp_left", tau=56.0,
                    pooling_axis="position", layer_spec=-1, rc_average=False)
    pool_loaded = m.HyenaDNAPooler(backend_loaded, **pool_cfg)
    pool_scratch = m.HyenaDNAPooler(backend_scratch, **pool_cfg)

    # Short sequences to keep compute tiny
    seqs = ["ACGT" * 64, "GGGGAAAACCCC" * 16]  # 256 & 192 bp
    E_loaded = pool_loaded.embed(seqs)   # [N, D]
    E_scratch = pool_scratch.embed(seqs) # [N, D]

    # Print first 8 dims and deltas
    for i, s in enumerate(seqs):
        cos = float(F.cosine_similarity(E_loaded[i], E_scratch[i], dim=0).item())
        l2  = float(torch.linalg.norm(E_loaded[i] - E_scratch[i]).item())
        print(f"[seq{i}] len={len(s)}")
        print("  loaded  :", ", ".join(f"{x:.5f}" for x in E_loaded[i][:8].tolist()))
        print("  scratch :", ", ".join(f"{x:.5f}" for x in E_scratch[i][:8].tolist()))
        print(f"  cos(loaded, scratch)={cos:.6f}  L2={l2:.6f}")

    # They should *not* be identical if weights were applied
    assert torch.all(F.cosine_similarity(E_loaded, E_scratch, dim=1) < 0.9999), \
        "Embeddings are virtually identical; weights may not have been applied."

def test_forward_hidden_states_shapes(loaded_wrapper_and_cfg, tokenizer):
    """
    Ensure forward works and hidden_states length matches n_layer+1.
    Skips (politely) if CPU with a huge model.
    """
    wrapper, _ = loaded_wrapper_and_cfg
    model = wrapper.model

    d_model = int(model.backbone.ln_f.normalized_shape[0])
    n_layer = len(model.backbone.layers)

    # Heuristic skip for large models on CPU
    if DEVICE.type == "cpu" and (d_model * n_layer > 8192):
        pytest.skip(f"Model too large for CPU quick-test (d_model={d_model}, n_layer={n_layer}). "
                    f"Run on GPU or pick a smaller model (e.g., hyenadna-small-32k).")

    batch = ["ACGTACGTAC", "GGGGAAAACCCC"]
    enc = tokenizer(batch, padding="longest", return_tensors="pt")
    ids = enc["input_ids"].to(DEVICE)

    out = wrapper(ids, output_hidden_states=True, return_dict=True)
    assert hasattr(out, "last_hidden_state")
    H = out.last_hidden_state
    assert H.ndim == 3 and H.shape[0] == 2 and H.shape[-1] == d_model

    hs = out.hidden_states
    assert isinstance(hs, tuple) and len(hs) == (n_layer + 1)


def test_pooler_shapes_norms_and_rc(loaded_wrapper_and_cfg, tokenizer):
    """
    HyenaDNAPooler should return unit-norm [N, D] and rc_average=True
    equals normalized(0.5*(ef+er)).
    """
    wrapper, _ = loaded_wrapper_and_cfg
    # Heuristic skip for huge models on CPU
    d_model = int(wrapper.model.backbone.ln_f.normalized_shape[0])
    n_layer = len(wrapper.model.backbone.layers)
    if DEVICE.type == "cpu" and (d_model * n_layer > 8192):
        pytest.skip("Skip pooler test for large model on CPU.")

    # Minimal backend stub the pooler expects
    class _Backend:
        def __init__(self, w, tok, max_len=2048):
            self.model = w
            if hasattr(self.model, "config"):
                self.model.config.output_hidden_states = True
                self.model.config.use_return_dict = True
            self.tokenizer = tok
            self.device = DEVICE
            self.max_length = max_len

        @staticmethod
        def _choose_pad_multiple(L: int):
            for mval in (128, 64, 32, 16, 8, 4):
                if L % mval == 0:
                    return mval
            return None

    backend = _Backend(wrapper, tokenizer)

    pooler = m.HyenaDNAPooler(backend,
                              direction="exp_left", tau=56.0,
                              pooling_axis="position", layer_spec=-1,
                              rc_average=False)

    seqs = ["ACGT" * 20, "GGGGAAAACCCC" * 4]
    E = pooler.embed(seqs)  # [N, D]
    assert E.ndim == 2 and E.shape[0] == len(seqs)
    norms = torch.linalg.vector_norm(E, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    s = "ACGT" * 16
    ef = pooler._embed_forward_or_rc(s, use_rc=False)
    er = pooler._embed_forward_or_rc(s, use_rc=True)
    avg = F.normalize(0.5 * (ef + er), p=2, dim=0)

    pooler_rc = m.HyenaDNAPooler(backend,
                                 direction="exp_left", tau=56.0,
                                 pooling_axis="position", layer_spec=-1,
                                 rc_average=True)
    e = pooler_rc.embed([s]).squeeze(0)
    assert torch.allclose(e, avg, atol=1e-5)


def test_exp_left_weights_monotone(loaded_wrapper_and_cfg, tokenizer):
    """
    exp_left positional weights should strictly decrease with position (ignoring pads).
    """
    wrapper, _ = loaded_wrapper_and_cfg

    class _Backend:
        def __init__(self, w, tok):
            self.model = w
            if hasattr(self.model, "config"):
                self.model.config.output_hidden_states = True
                self.model.config.use_return_dict = True
            self.tokenizer = tok
            self.device = DEVICE
            self.max_length = 256

        @staticmethod
        def _choose_pad_multiple(L: int):
            for mval in (128, 64, 32, 16, 8, 4):
                if L % mval == 0:
                    return mval
            return None

    backend = _Backend(wrapper, tokenizer)
    pooler = m.HyenaDNAPooler(backend, direction="exp_left", tau=16.0,
                              pooling_axis="position", layer_spec=-1, rc_average=False)

    mask = torch.ones(1, 8, dtype=torch.bool)
    w = pooler._make_weights(mask).squeeze(0)
    diffs = (w[:-1] - w[1:]).detach().cpu().numpy()
    assert (diffs > 0).all(), f"exp_left weights must decrease left→right, got {w.tolist()}"


def test_single_snp_changes_embedding(loaded_wrapper_and_cfg, tokenizer):
    """
    Single-point mutation should perturb pooled embedding (cos<1.0).
    """
    wrapper, _ = loaded_wrapper_and_cfg
    # Heuristic skip for huge models on CPU
    d_model = int(wrapper.model.backbone.ln_f.normalized_shape[0])
    n_layer = len(wrapper.model.backbone.layers)
    if DEVICE.type == "cpu" and (d_model * n_layer > 8192):
        pytest.skip("Skip mutation test for large model on CPU.")

    class _Backend:
        def __init__(self, w, tok):
            self.model = w
            if hasattr(self.model, "config"):
                self.model.config.output_hidden_states = True
                self.model.config.use_return_dict = True
            self.tokenizer = tok
            self.device = DEVICE
            self.max_length = 256

        @staticmethod
        def _choose_pad_multiple(L: int):
            for mval in (128, 64, 32, 16, 8, 4):
                if L % mval == 0:
                    return mval
            return None

    backend = _Backend(wrapper, tokenizer)
    pooler = m.HyenaDNAPooler(backend, direction="exp_left", tau=56.0,
                              pooling_axis="position", layer_spec=-1, rc_average=False)

    s = "ACGT" * 20
    s_mut = "ACCT" + ("ACGT" * 19)
    v1 = pooler.embed([s]).squeeze(0)
    v2 = pooler.embed([s_mut]).squeeze(0)
    cos = float(F.cosine_similarity(v1, v2, dim=0).item())
    assert cos < 0.999999, f"Cosine too close to 1.0 ({cos}); mutation should change embedding."


def test_pooler_arg_validation(loaded_wrapper_and_cfg, tokenizer):
    wrapper, _ = loaded_wrapper_and_cfg

    class _Backend:
        def __init__(self, w, tok):
            self.model = w
            if hasattr(self.model, "config"):
                self.model.config.output_hidden_states = True
                self.model.config.use_return_dict = True
            self.tokenizer = tok
            self.device = DEVICE
            self.max_length = 256

        @staticmethod
        def _choose_pad_multiple(L: int):
            return None

    backend = _Backend(wrapper, tokenizer)
    with pytest.raises(ValueError):
        m.HyenaDNAPooler(backend, direction="not_a_mode")

    with pytest.raises(ValueError):
        m.HyenaDNAPooler(backend, pooling_axis="channels→position", channel_groups=None)

    # Valid construction (should not raise)
    m.HyenaDNAPooler(backend, pooling_axis="position", direction="mean", layer_spec=-1)


def _rand_seq(L, alphabet="ACGTN"):
    rng = random.Random(1234 + L)
    return "".join(rng.choice(alphabet) for _ in range(L))


def _left_pad_ids_backend_rule(ids_list, backend):
    """
    Left-pad to the SAME length the string path uses:
      T = ceil(max_len / pad_mult) * pad_mult   if pad_mult exists
        = max_len                                otherwise
    """
    pad_id = backend.tokenizer.pad_token_id
    assert pad_id is not None, "CharacterTokenizer must define a pad_token_id"

    # HyenaBackend pads to a multiple of _choose_pad_multiple(max_length)
    pad_mult = backend._choose_pad_multiple(backend.max_length)
    max_len = max(int(x.numel()) for x in ids_list)
    if pad_mult:
        T = ((max_len + pad_mult - 1) // pad_mult) * pad_mult
    else:
        T = max_len

    B = len(ids_list)
    device = ids_list[0].device
    out = torch.full((B, T), int(pad_id), dtype=torch.long, device=device)
    for i, ids in enumerate(ids_list):
        L = int(ids.numel())
        out[i, T - L : T] = ids  # left pad
    return out


# -----------------------
# New backend-centric tests
# -----------------------
def test_token_inventory_and_specials(tokenizer):
    backend = HyenaBackend(
        model_name=cfg.model_name,
        model_dir=cfg.model_dir,
        offline=True,
        prefer_cuda=torch.cuda.is_available(),
        pooling="mean",
    )
    inv = backend.token_inventory_details()
    t2i = inv["token_to_id"]
    specials = inv["specials"]
    dna = inv["dna_ids"]

    for base in "ACGTN":
        assert base in dna, f"Missing DNA id for {base}"
        assert isinstance(dna[base], int)
        assert base in t2i, "Uppercase base should appear in flat vocab map"

    # If pad/unk exist, ensure they are ints
    for k in ("pad_id", "unk_id"):
        if k in specials and specials[k] is not None:
            assert isinstance(specials[k], int)


@CUDA_REQ
def test_rc_lut_maps_bases_and_keeps_specials():
    backend = HyenaBackend(
        model_name=cfg.model_name,
        model_dir=cfg.model_dir,
        offline=True,
        prefer_cuda=True,
    )
    lut = backend._ensure_rc_lut_cuda()
    inv = backend.full_token_inventory()
    specials = backend.token_special_ids()

    def gid(tok):
        return inv.get(tok, inv.get(tok.lower(), inv.get(tok.upper())))

    A, C, G, T, N = map(gid, list("ACGTN"))
    if A is not None and T is not None:
        assert int(lut[A]) == T and int(lut[T]) == A
    if C is not None and G is not None:
        assert int(lut[C]) == G and int(lut[G]) == C
    if N is not None:
        assert int(lut[N]) == N
    for sp in ("pad_id", "unk_id"):
        sid = specials.get(sp, None)
        if sid is not None:
            assert int(lut[int(sid)]) == int(sid), f"{sp} should map to itself"


@CUDA_REQ
def test_attention_mask_is_ignored_in_embed_tokens(monkeypatch):
    """Passing attention_mask should NOT be forwarded to the model."""
    backend = HyenaBackend(
        model_name=cfg.model_name,
        model_dir=cfg.model_dir,
        offline=True,
        prefer_cuda=True,
    )

    # Small batch
    seqs = ["ACGT" * 8, "GGGGAAAA"]
    tok = backend.tokenizer(seqs, padding="longest", return_tensors="pt")
    ids = tok["input_ids"].to(backend.device)
    attn = torch.ones_like(ids)

    forwarded_kwargs = {"saw_attention_mask": False}

    orig_forward = backend.model.forward

    def spy_forward(*args, **kwargs):
        assert "attention_mask" not in kwargs, "attention_mask leaked into model.forward"
        forwarded_kwargs["saw_attention_mask"] = "attention_mask" in kwargs
        return orig_forward(*args, **kwargs)

    monkeypatch.setattr(backend.model, "forward", spy_forward, raising=True)

    _ = backend.embed_tokens(ids, pooling="mean", rc_invariant=False, attention_mask=attn)
    assert forwarded_kwargs["saw_attention_mask"] is False


@CUDA_REQ
def test_truncation_max_length_in_embed_tokens():
    """embed_tokens(max_length=X) must truncate to X."""
    backend = HyenaBackend(
        model_name=cfg.model_name,
        model_dir=cfg.model_dir,
        offline=True,
        prefer_cuda=True,
        pooling="none",
    )
    enc = DNALUTEncoder.from_backend(backend)
    seq = _rand_seq(300)
    ids = enc.encode_to_cuda_ids(seq).unsqueeze(0)  # [1,300]
    hs, mask = backend.embed_tokens(ids, pooling="none", rc_invariant=False, max_length=128)
    assert hs.shape[1] == 128 and mask.shape[1] == 128, "Token length must be truncated to 128"


@CUDA_REQ
def test_normalization_flag_behavior_mean_and_gem():
    backend = HyenaBackend(
        model_name=cfg.model_name,
        model_dir=cfg.model_dir,
        offline=True,
        prefer_cuda=True,
        pooling="mean",
        normalize=False,
    )
    enc = DNALUTEncoder.from_backend(backend)
    ids = enc.encode_to_cuda_ids(_rand_seq(512)).unsqueeze(0)

    # mean, normalize=False
    v = backend.embed_tokens(ids, pooling="mean", rc_invariant=False)
    n = float(torch.linalg.norm(v, dim=1).item())
    assert not math.isclose(n, 1.0, rel_tol=1e-5, abs_tol=1e-5)

    # mean, normalize=True
    backend.normalize = True
    v2 = backend.embed_tokens(ids, pooling="mean", rc_invariant=False)
    n2 = torch.linalg.vector_norm(v2, dim=1)
    assert torch.allclose(n2, torch.ones_like(n2), atol=1e-5)

    # gem should run and match shape + be finite
    v3 = backend.embed_tokens(ids, pooling="gem", rc_invariant=False)
    assert v3.shape == v2.shape and torch.isfinite(v3).all()


@CUDA_REQ
def test_streaming_variants_and_empty_iter():
    backend = HyenaBackend(
        model_name=cfg.model_name,
        model_dir=cfg.model_dir,
        offline=True,
        prefer_cuda=True,
        pooling="mean",
        normalize=False,
    )
    enc = DNALUTEncoder.from_backend(backend)

    seq = _rand_seq(2048)
    ids = enc.encode_to_cuda_ids(seq)  # [L]
    wind = CudaWindower(device=ids.device)
    W = wind.as_windows(ids, 256, 128)  # [N,256]

    # cat=False should return a list
    outs = backend.embed_tokens_streaming((W[i:i+32] for i in range(0, W.size(0), 32)),
                                          pooling="mean", rc_invariant=False, cat=False)
    assert isinstance(outs, list) and len(outs) > 0 and outs[0].ndim == 2

    # empty iterator → [0,0] tensor
    empty = backend.embed_tokens_streaming((x for x in []), pooling="mean", rc_invariant=False, cat=True)
    assert empty.ndim == 2 and empty.numel() == 0 and empty.shape[0] == 0 and empty.shape[1] == 0


def test_error_paths_and_validation_cpu_ok():
    """Errors should be informative and consistent without needing GPU."""
    backend = HyenaBackend(
        model_name=cfg.model_name,
        model_dir=cfg.model_dir,
        offline=True,
        prefer_cuda=False,   # allow CPU for quick validation
        pooling="mean",
    )

    with pytest.raises(ValueError):
        backend.embed_tokens(torch.randn(2, 3), pooling="mean")  # wrong dtype

    with pytest.raises(ValueError):
        backend.embed_tokens(torch.randint(0, 4, (2, 3, 4), dtype=torch.long), pooling="mean")  # wrong shape

    with pytest.raises(ValueError):
        backend.embed_tokens(torch.randint(0, 4, (2, 3), dtype=torch.long), pooling="bad_mode")

    with pytest.raises(ValueError):
        backend.embed_tokens_streaming((torch.randint(0, 4, (2, 3), dtype=torch.long) for _ in range(2)),
                                       pooling="none")  # streaming forbids 'none'

    with pytest.raises(ValueError):
        backend.embed_best(object())  # unsupported input type


@CUDA_REQ
def test_build_pooler_factory_returns_pooler():
    backend = m.HyenaBackend(
        model_name=cfg.model_name,
        model_dir=cfg.model_dir,
        offline=True,
        prefer_cuda=True,
    )
    pooler = backend.build_pooler()
    assert isinstance(pooler, m.HyenaDNAPooler)


@CUDA_REQ
def test_rc_indexing_preserves_left_pads():
    """RC ids produced internally should keep pads on the left."""
    backend = HyenaBackend(
        model_name=cfg.model_name,
        model_dir=cfg.model_dir,
        offline=True,
        prefer_cuda=True,
    )
    enc = DNALUTEncoder.from_backend(backend)
    pad_id = int(backend.tokenizer.pad_token_id)

    seqs = [_rand_seq(64), _rand_seq(40), _rand_seq(25)]
    ids_list = [enc.encode_to_cuda_ids(s) for s in seqs]
    batch = _left_pad_ids_backend_rule(ids_list, backend)  # [B,T]

    # Re-run the same RC indexing logic from backend for validation
    lut = backend._ensure_rc_lut_cuda()
    mask = (batch != pad_id)
    B, T = batch.shape
    comp = lut[batch]
    lengths = mask.sum(dim=1)
    starts = (T - lengths).unsqueeze(1)
    J = torch.arange(T, device=batch.device).view(1, T)
    idx = torch.where(J >= starts, starts + (T - 1 - J), J)
    ids_rc = comp.gather(1, idx)

    # All positions before 'start' must be pad_id
    for b in range(B):
        s = int(starts[b].item())
        if s > 0:
            assert torch.all(ids_rc[b, :s] == pad_id), "Pads must remain on the left after RC gather"


# -----------------------
# Ids vs strings & streaming (original GPU suite)
# -----------------------
@CUDA_REQ
def test_ids_vs_strings_pooled_mean_rc_match():
    torch.manual_seed(0)
    backend = HyenaBackend(
        model_name=cfg.model_name,
        model_dir=cfg.model_dir,
        pooling="mean",
        normalize=False,     # keep consistent across paths
        offline=True,
        prefer_cuda=True,
    )
    enc = DNALUTEncoder.from_backend(backend)

    # ACGTN-only to avoid tokenizer surprises
    seqs = [
        _rand_seq(173),
        _rand_seq(512),
        _rand_seq(999),
        _rand_seq(31),
    ]

    # String path (reference)
    ref = backend.embed_list(seqs, pooling="mean", rc_invariant=True)  # [B,D], float32

    # Ids path: build ids_list, then pad using the same rule as the string path
    ids_list = [enc.encode_to_cuda_ids(s) for s in seqs]               # list of [L_i] (CUDA long)
    ids_batch = _left_pad_ids_backend_rule(ids_list, backend)          # [B,T] (CUDA long)

    out = backend.embed_tokens(ids_batch, pooling="mean", rc_invariant=True)  # [B,D]

    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


@CUDA_REQ
def test_ids_vs_strings_token_level_none_rc_match_on_short_sequences():
    """
    Token-level ('none') with RC-invariant=True should match between string and ids paths
    on small sequences (to keep memory bounded).
    """
    torch.manual_seed(1)
    backend = HyenaBackend(
        model_name=cfg.model_name,
        model_dir=cfg.model_dir,
        pooling="mean",      # irrelevant for 'none' but keep consistent
        normalize=False,
        offline=True,
        prefer_cuda=True,
    )
    enc = DNALUTEncoder.from_backend(backend)

    # Short ACGTN-only sequences
    seqs = [_rand_seq(64), _rand_seq(80)]

    # String path (reference)
    hs_ref, mask_ref = backend.embed_list(seqs, pooling="none", rc_invariant=True)  # [B,T,H], [B,T]

    # Ids path: encode and pad using the SAME rule as the string path (left pads; multiple-of)
    ids_list = [enc.encode_to_cuda_ids(s) for s in seqs]           # list of [L_i] (CUDA long)
    ids_batch = _left_pad_ids_backend_rule(ids_list, backend)      # [B,T] (CUDA long)

    # Sanity: ensure both paths see the same T
    tok = backend.tokenizer(
        seqs,
        padding="longest",
        truncation=True,
        max_length=backend.max_length,
        pad_to_multiple_of=backend._choose_pad_multiple(backend.max_length),
        return_tensors="pt",
    )
    tok_T = tok["input_ids"].shape[1]
    assert ids_batch.shape[1] == tok_T

    # Ids path (token-level, RC-invariant=True). Requires backend RC path that preserves left pads.
    hs_ids, mask_ids = backend.embed_tokens(ids_batch, pooling="none", rc_invariant=True)

    # Exact match (no cropping needed now that T is aligned)
    torch.testing.assert_close(hs_ids,  hs_ref,  rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(mask_ids.to(mask_ref.dtype), mask_ref)


@CUDA_REQ
def test_rc_invariance_identity_on_ids_batch():
    """
    For a fixed ids batch with no pads, pooled(rc_avg=True) should equal
    0.5*(pooled(forward) + pooled(reverse-complement-per-window)).
    """
    torch.manual_seed(2)
    backend = HyenaBackend(
                model_name=cfg.model_name,
                model_dir=cfg.model_dir,
                pooling="mean",
                normalize=False,
                offline=True,
                prefer_cuda=True,
            )
    enc = DNALUTEncoder.from_backend(backend)

    # One fixed-length batch (no padding)
    seq = _rand_seq(512)
    ids = enc.encode_to_cuda_ids(seq).unsqueeze(0)  # [1, T]
    rc_lut = backend._ensure_rc_lut_cuda()
    ids_rc = rc_lut[ids].flip(1)                   # [1, T]

    # rc-averaged in one call:
    e_rc = backend.embed_tokens(ids, pooling="mean", rc_invariant=True)         # [1,D]

    # explicit average:
    e_f  = backend.embed_tokens(ids, pooling="mean", rc_invariant=False)        # [1,D]
    e_r  = backend.embed_tokens(ids_rc, pooling="mean", rc_invariant=False)     # [1,D]
    e_avg = 0.5 * (e_f + e_r)

    torch.testing.assert_close(e_rc, e_avg, rtol=RTOL, atol=ATOL)


@CUDA_REQ
def test_streaming_vs_monolithic_on_windows():
    """
    Streaming pooled embeddings over window batches must match a single monolithic call.
    """
    torch.manual_seed(3)
    backend = HyenaBackend(
                model_name=cfg.model_name,
                model_dir=cfg.model_dir,
                pooling="mean",
                normalize=False,
                offline=True,
                prefer_cuda=True,
            )
    enc = DNALUTEncoder.from_backend(backend)

    # Create a long sequence, make windows on GPU
    seq = _rand_seq(8192)
    ids = enc.encode_to_cuda_ids(seq)                   # [L]
    win, stride, B = 256, 64, 1024
    wind = CudaWindower(device=ids.device)
    w2, _, _ = wind.as_windows(ids, win, stride), None, None  # [N, win]

    # Monolithic call
    mono = backend.embed_tokens(w2, pooling="mean", rc_invariant=False)  # [N,D]

    # Streaming call
    stream = backend.embed_token_batches_pooled(
        wind.iter_batches(w2, B),
        rc_invariant=False,
        pooling="mean",
        out_device="cuda",
    )
    torch.testing.assert_close(stream, mono, rtol=RTOL, atol=ATOL)


@CUDA_REQ
def test_embed_best_accepts_all_three_inputs_and_matches():
    """
    embed_best should accept:
      - List[str]
      - LongTensor[B,T]
      - Iterable[LongTensor[B,T]]
    and produce consistent results for equivalent inputs.
    """
    torch.manual_seed(4)
    backend = HyenaBackend(
                model_name=cfg.model_name,
                model_dir=cfg.model_dir,
                pooling="mean",
                normalize=False,
                offline=True,
                prefer_cuda=True,
            )
    enc = DNALUTEncoder.from_backend(backend)

    seqs = [_rand_seq(512), _rand_seq(512)]
    # strings
    v_str = backend.embed_best(seqs)  # Pooler path; defaults

    # ids batch (left-pad to exact same length is unnecessary since lengths are equal)
    ids_b = torch.stack([enc.encode_to_cuda_ids(s) for s in seqs], dim=0)  # [B,T]
    v_ids = backend.embed_best(ids_b, pooling="mean", rc_invariant=True)

    # iterator of batches (split the same batch in two)
    it = (ids_b[i:i+1] for i in range(ids_b.size(0)))
    v_it = backend.embed_best(it, pooling="mean", rc_invariant=True, out_device="cuda")

    assert v_str.shape == v_ids.shape == v_it.shape
    # String path uses HyenaDNAPooler defaults; allow small numerical drift across paths.
    torch.testing.assert_close(v_ids, v_it, rtol=RTOL, atol=ATOL)


@CUDA_REQ
def test_two_strands_vs_rcavg_equivalence_per_window():
    """
    If we only pass forward windows and request rc_invariant=True,
    result should equal averaging the embeddings of forward windows and their RC counterparts.
    """
    torch.manual_seed(5)
    backend = HyenaBackend(
                model_name=cfg.model_name,
                model_dir=cfg.model_dir,
                pooling="mean",
                normalize=False,
                offline=True,
                prefer_cuda=True,
            )
    enc = DNALUTEncoder.from_backend(backend)
    rc_lut = backend._ensure_rc_lut_cuda()

    seq = _rand_seq(2048)
    ids = enc.encode_to_cuda_ids(seq)  # [L]
    win, stride = 256, 128
    wind = CudaWindower(device=ids.device)
    Fw = wind.as_windows(ids, win, stride)                   # [N, win]
    Rw = rc_lut[Fw].flip(1)                                   # [N, win]  reverse-complement per window

    e_rcavg = backend.embed_tokens(Fw, pooling="mean", rc_invariant=True)      # [N,D]
    e_fwd   = backend.embed_tokens(Fw, pooling="mean", rc_invariant=False)
    e_rev   = backend.embed_tokens(Rw, pooling="mean", rc_invariant=False)
    e_pair  = 0.5 * (e_fwd + e_rev)

    torch.testing.assert_close(e_rcavg, e_pair, rtol=RTOL, atol=ATOL)
