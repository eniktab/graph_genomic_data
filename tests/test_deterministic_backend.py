import torch
import pytest

from src.DeterministicBackend import (
    DeterministicBackend,
    revcomp,
    _non_rc_symmetric_seq,
    _mutate,
)

# ---------------------------------------------------------------------------
# Module-wide CUDA gate: the backend (and HELIndexer) require CUDA tensors
# ---------------------------------------------------------------------------
try:
    CUDA_OK = torch.cuda.is_available()
except Exception:
    CUDA_OK = False

pytestmark = pytest.mark.skipif(
    not CUDA_OK, reason="CUDA required for DeterministicBackend tests"
)

# ---------------------------------------------------------------------------
# Shared config so tests are stable/reproducible
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def backend_args():
    """
    Args chosen to:
    - keep dimension small-ish for quick tests
    - enable rc_merge so forward/revcomp collapse to same embedding by default
    - enable per-slice centering + power companding (mutation sensitivity)
    """
    return dict(
        dim=256,
        k=7,
        rc_merge=True,
        normalize=True,
        pos_buckets=8,
        num_hashes=2,
        micro_buckets=64,
        power_gamma=0.5,
        center_slices=True,
        device="cuda",
    )


@pytest.fixture(scope="module")
def backend(backend_args):
    return DeterministicBackend(**backend_args)


@pytest.fixture(scope="module")
def backend_rc_off(backend_args):
    """Same as `backend` but with rc_merge disabled (strand should matter)."""
    args = dict(backend_args)
    args["rc_merge"] = False
    return DeterministicBackend(**args)


@pytest.fixture(scope="module")
def base_seq():
    """Non-RC-symmetric ~10kb synthetic DNA with only A/C/G/T."""
    s = _non_rc_symmetric_seq(10_000)
    assert revcomp(s) != s, "fixture sequence is accidentally RC-symmetric"
    return s


# =============================== Test Classes ================================

class TestDeviceAndDeterminism:
    def test_embed_one_cuda_float32_and_deterministic(self, backend, base_seq):
        """embed_one() should be CUDA, float32, right shape, deterministic."""
        v1 = backend.embed_one(base_seq)
        v2 = backend.embed_one(base_seq)

        assert v1.shape == (backend.dim,)
        assert v1.dtype == torch.float32
        assert v1.is_cuda, "embed_one() must return CUDA tensor"
        # bit-for-bit identical for same input
        assert torch.allclose(v1, v2, atol=0.0, rtol=0.0)

    def test_global_l2_normalization(self, backend, base_seq):
        """If normalize=True, the whole vector should end up ~unit length."""
        v = backend.embed_one(base_seq)
        norm = torch.linalg.vector_norm(v).item()
        assert abs(norm - 1.0) < 1e-5, f"expected ~unit norm, got {norm}"


class TestStrandBehavior:
    def test_rc_merge_on_identical(self, backend, base_seq):
        """
        With rc_merge=True, forward and reverse-complement embeddings
        should be identical (canonical k-mers, strand-invariant bucketing).
        """
        v_fwd = backend.embed_one(base_seq)
        v_rc = backend.embed_one(revcomp(base_seq))
        assert torch.allclose(v_fwd, v_rc, atol=0.0, rtol=0.0)

    def test_rc_merge_off_differs(self, backend_rc_off, base_seq):
        """
        With rc_merge=False, strand should matter (half-slice rotation).
        """
        v_fwd = backend_rc_off.embed_one(base_seq)
        v_rc = backend_rc_off.embed_one(revcomp(base_seq))
        cos = torch.nn.functional.cosine_similarity(v_fwd, v_rc, dim=0).item()
        assert cos < 0.9999, f"rc_merge=False but forward≈RC (cos={cos:.6f})"

    def test_rc_invariant_kwarg_overrides(self, backend, backend_rc_off, base_seq):
        """
        rc_invariant toggles strand behavior per call:
          - On rc_merge=False backend, setting rc_invariant=True collapses strands.
          - On rc_merge=True backend, setting rc_invariant=False makes strands differ.
        """
        s_rc = revcomp(base_seq)

        # 1) Backend default rc_merge=False, but force rc_invariant=True -> equal
        xb = backend_rc_off.embed_best([base_seq, s_rc], rc_invariant=True)
        assert xb.is_cuda and xb.shape == (2, backend_rc_off.dim)
        assert torch.allclose(xb[0], xb[1], atol=0.0, rtol=0.0)

        # 2) Backend default rc_merge=True, but force rc_invariant=False -> differ
        yb = backend.embed_best([base_seq, s_rc], rc_invariant=False)
        cos = torch.nn.functional.cosine_similarity(yb[0], yb[1], dim=0).item()
        assert cos < 0.9999, f"rc_invariant=False but forward≈RC (cos={cos:.6f})"


class TestRobustnessAndMutations:
    def test_single_point_mutation_changes_embedding(self, backend, base_seq):
        """
        A single internal mutation should measurably perturb the embedding.
        """
        v_ref = backend.embed_one(base_seq)
        v_mut = backend.embed_one(_mutate(base_seq, i=1000))
        cos = torch.nn.functional.cosine_similarity(v_ref, v_mut, dim=0).item()
        assert cos < 0.9999, f"single mutation caused almost no change (cos={cos:.6f})"

    def test_short_sequence_is_all_zero_and_not_normalized(self, backend_args):
        """
        Sequences shorter than k should early-return a zero vector
        without per-slice shaping or global normalization.
        """
        B = DeterministicBackend(**backend_args)
        short = "A" * (B.k - 1)  # strictly shorter than k
        v_short = B.embed_one(short)
        assert v_short.is_cuda
        assert torch.count_nonzero(v_short) == 0
        assert torch.linalg.vector_norm(v_short).item() == 0.0


class TestBatchAPIAndAutocast:
    def test_batch_api_consistency_cuda(self, backend, base_seq):
        """
        embed_list() stacks into [B, D] on CUDA, preserves normalize flag,
        and handles mixed-length inputs (normal, mutated, too-short).
        """
        before_norm_flag = backend.normalize

        seq_mut = _mutate(base_seq, i=1000)
        seq_short = "A" * (backend.k - 1)

        batch = backend.embed_list([base_seq, seq_mut, seq_short])

        # shape / dtype / contiguity / device
        assert batch.shape == (3, backend.dim)
        assert batch.dtype == torch.float32
        assert batch.is_contiguous()
        assert batch.is_cuda

        # row[2] should correspond to the all-zero short embedding
        assert torch.count_nonzero(batch[2]).item() == 0

        # normalize flag must be restored
        assert backend.normalize is before_norm_flag

    def test_embed_list_normalize_override_restored(self, backend, base_seq):
        """
        normalize kwarg should temporarily override, then restore.
        """
        before = backend.normalize
        out = backend.embed_list([base_seq, _mutate(base_seq, i=500)], normalize=False)
        assert out.is_cuda
        # Expect at least one row not ~unit norm (global normalization off)
        norms = torch.linalg.vector_norm(out, dim=1)
        assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
        assert backend.normalize is before

    def test_embed_best_autocast_fp16(self, backend, base_seq):
        """
        Under AMP autocast, embed_best should still return a CUDA floating tensor
        with correct shape. We allow fp16 or fp32 depending on op promotion.
        """
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
            xb = backend.embed_best([base_seq, _mutate(base_seq, i=777)], rc_invariant=False)
        assert xb.is_cuda and xb.shape == (2, backend.dim)
        assert xb.dtype in (torch.float16, torch.float32)

    def test_embed_best_accepts_unknown_kwargs(self, backend, base_seq):
        """
        Future-proofing: backend should ignore unknown kwargs.
        """
        out = backend.embed_best([base_seq], rc_invariant=True, unknown_kwarg=123)
        assert out.is_cuda and out.shape == (1, backend.dim)
