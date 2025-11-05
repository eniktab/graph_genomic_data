import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import pytest

from src.configs import IndexConfig
from src.HELIndexer import HELIndexer

# --- GPU requirement (HELIndexer builds an ANN on CUDA) ----------------------
try:
    import torch
    CUDA_OK = torch.cuda.is_available()
except Exception:
    CUDA_OK = False

pytestmark = pytest.mark.skipif(
    not CUDA_OK, reason="CUDA required for HELIndexer/RaftGPU tests"
)

# ------------------------------- Fixtures ------------------------------------

@pytest.fixture
def toy_fasta(tmp_path: Path) -> Path:
    """
    Tiny 2-chrom FASTA so builds are fast but still exercise
    sliding windows across multiple contigs.
    chr1 length = 1600, chr2 length = 1200
    """
    p = tmp_path / "toy.fa"
    with p.open("w") as f:
        f.write(">chr1\n" + ("ACGT" * 400) + "\n")
        f.write(">chr2\n" + ("TGCA" * 300) + "\n")
    return p


# ---------------------------- Helper functions -------------------------------

def expected_windows_no_tail(L: int, T: int, S: int) -> int:
    """
    Mirrors HELIndexer._iter_windows_fast/_count_windows logic:
    - do NOT add an extra 'tail' window
    - require L >= T, else 0
    - n = 1 + (L - T) // S, for S > 0
    """
    if L < T:
        return 0
    if S <= 0:
        return 1
    return 1 + (L - T) // S


def expected_total_windows(window: int, stride: int, rc: bool) -> int:
    """
    For the toy_fasta fixture specifically (chr1=1600, chr2=1200).
    """
    chr1 = expected_windows_no_tail(1600, window, stride)
    chr2 = expected_windows_no_tail(1200, window, stride)
    base = chr1 + chr2
    return base * (2 if rc else 1)


# =============================== Test Classes ================================

class TestBuildAndLoad:
    def test_build_include_dataset_true(self, tmp_path: Path, toy_fasta: Path):
        cfg = IndexConfig(window=128, stride=64, rc_index=False)
        out_dir = tmp_path / "idx_in_dataset"

        idx = HELIndexer(toy_fasta, cfg, embedder="deterministic", emb_batch=128)
        idx.build_or_load(out_dir, reuse_existing=False, include_dataset=True, verbose=False)

        # Artifacts controlled by HELIndexer (do not rely on RAFT internals)
        assert (out_dir / HELIndexer.MANIFEST_NAME).exists()
        assert (out_dir / HELIndexer.HEL_META_NAME).exists()

        # Old fallback file must not be produced in the new implementation
        assert not (out_dir / "embeddings_f32.npy").exists()

        # Check metadata is coherent
        meta = idx.info()
        assert meta["schema"] == "hel-indexer-v1"
        assert meta["reference"].endswith("toy.fa")
        assert meta["window"] == 128
        assert meta["stride"] == 64
        assert meta["rc_index"] is False
        assert isinstance(meta["embedding_dim"], int) and meta["embedding_dim"] > 0
        assert isinstance(meta["n_vectors"], int) and meta["n_vectors"] > 0

        # RaftGPU __len__ should reflect vector count if implemented
        # (keep soft: only assert equality if len(...) is available)
        try:
            assert len(idx.index) == meta["n_vectors"]
        except TypeError:
            # No __len__ exposed; at least ensure index object exists
            assert idx.index is not None

        # Contigs summary should match our toy reference
        contigs = {c["name"]: c["length"] for c in meta["contigs"]}
        assert contigs["chr1"] == 1600
        assert contigs["chr2"] == 1200

    def test_build_without_dataset_requires_fallback(self, tmp_path: Path, toy_fasta: Path):
        """
        Current RaftGPU.save() requires fallback embeddings if include_dataset=False.
        Until HELIndexer passes them through, this should raise ValueError.
        """
        cfg = IndexConfig(window=128, stride=64, rc_index=False)
        out_dir = tmp_path / "idx_no_dataset"

        idx = HELIndexer(toy_fasta, cfg, embedder="deterministic", emb_batch=96)
        with pytest.raises(ValueError, match="requires fallback_embeddings"):
            idx.build_or_load(out_dir, reuse_existing=False, include_dataset=False, verbose=False)


class TestRCIndexingAndWindowMath:
    def test_rc_index_doubles_vectors(self, tmp_path: Path, toy_fasta: Path):
        window, stride = 128, 64

        cfg_no = IndexConfig(window=window, stride=stride, rc_index=False)
        cfg_rc = IndexConfig(window=window, stride=stride, rc_index=True)

        out_a = tmp_path / "idx_no_rc"
        idx_a = HELIndexer(toy_fasta, cfg_no, embedder="deterministic", emb_batch=64)
        idx_a.build_or_load(out_a, reuse_existing=False, include_dataset=True, verbose=False)

        out_b = tmp_path / "idx_rc"
        idx_b = HELIndexer(toy_fasta, cfg_rc, embedder="deterministic", emb_batch=64)
        idx_b.build_or_load(out_b, reuse_existing=False, include_dataset=True, verbose=False)

        na = idx_a.info()["n_vectors"]
        nb = idx_b.info()["n_vectors"]

        assert nb == 2 * na, "rc_index=True should double total windows (fwd + rev)"

        # Also verify our expected math against what's built (no tail window)
        exp_no = expected_total_windows(window, stride, rc=False)
        exp_rc = expected_total_windows(window, stride, rc=True)
        assert na == exp_no
        assert nb == exp_rc

    def test_window_math_changes_with_params(self, tmp_path: Path, toy_fasta: Path):
        # Use parameters that change floor division outcome (no tail coverage)
        cfg1 = IndexConfig(window=200, stride=100, rc_index=False)
        cfg2 = IndexConfig(window=200, stride=128, rc_index=False)

        idx1 = HELIndexer(toy_fasta, cfg1, embedder="deterministic", emb_batch=64)
        idx2 = HELIndexer(toy_fasta, cfg2, embedder="deterministic", emb_batch=64)

        out1 = tmp_path / "idx_w200_s100"
        out2 = tmp_path / "idx_w200_s128"

        idx1.build_or_load(out1, reuse_existing=False, include_dataset=True, verbose=False)
        idx2.build_or_load(out2, reuse_existing=False, include_dataset=True, verbose=False)

        n1 = idx1.info()["n_vectors"]
        n2 = idx2.info()["n_vectors"]

        exp1 = expected_total_windows(200, 100, rc=False)
        exp2 = expected_total_windows(200, 128, rc=False)
        assert n1 == exp1
        assert n2 == exp2
        assert n2 <= n1, "Larger stride should not increase window count"


class TestReuseExisting:
    def test_build_or_load_reuse_existing(self, tmp_path: Path, toy_fasta: Path):
        cfg = IndexConfig(window=128, stride=64, rc_index=False)
        out_dir = tmp_path / "idx_reuse"

        idx1 = HELIndexer(toy_fasta, cfg, embedder="deterministic", emb_batch=64)
        idx1.build_or_load(out_dir, reuse_existing=True, include_dataset=True, verbose=False)

        meta_path = out_dir / HELIndexer.HEL_META_NAME
        assert meta_path.exists()
        before_mtime = meta_path.stat().st_mtime

        # Reusing should not rewrite HEL metadata
        idx2 = HELIndexer(toy_fasta, cfg, embedder="deterministic")
        idx2.build_or_load(out_dir, reuse_existing=True, include_dataset=True, verbose=False)

        after_mtime = meta_path.stat().st_mtime

        assert idx2.info()["n_vectors"] == idx1.info()["n_vectors"]
        assert after_mtime == before_mtime, "Reusing should not rewrite HEL meta"


class TestConfigAndMetadata:
    def test_metric_l2_not_normalized(self, tmp_path: Path, toy_fasta: Path):
        # IndexConfig does not accept 'metric' in this codebase; HELIndexer picks it internally.
        cfg = IndexConfig(window=128, stride=64, rc_index=False)
        out_dir = tmp_path / "idx_metric_default"

        idx = HELIndexer(toy_fasta, cfg, embedder="deterministic", emb_batch=64)
        idx.build_or_load(out_dir, reuse_existing=False, include_dataset=True, verbose=False)

        meta = idx.info()

        # We can't assert a specific metric here (API doesn't allow choosing it),
        # but we CAN assert that it's present and sensible, and that the related flags exist.
        assert "metric" in meta and isinstance(meta["metric"], str) and len(meta["metric"]) > 0
        assert "scores_higher_better" in meta and isinstance(meta["scores_higher_better"], bool)
        assert "normalized" in meta and isinstance(meta["normalized"], bool)

    def test_metadata_contains_core_fields(self, tmp_path: Path, toy_fasta: Path):
        cfg = IndexConfig(window=128, stride=64, rc_index=False)
        out_dir = tmp_path / "idx_meta_core"

        idx = HELIndexer(toy_fasta, cfg, embedder="deterministic")
        idx.build_or_load(out_dir, reuse_existing=False, include_dataset=True, verbose=False)

        meta = idx.info()
        # Presence + basic types
        for k in (
            "schema",
            "created_by",
            "reference",
            "window",
            "stride",
            "rc_index",
            "skip_N_frac",
            "metric",
            "scores_higher_better",
            "normalized",
            "embedder_name",
            "embedding_dim",
            "n_vectors",
            "ids_path",
            "token_len_fixed",
            "contigs",
            "raft_params",
            "raft_manifest_file",
            "raft_manifest",
        ):
            assert k in meta, f"missing metadata key: {k}"

        assert isinstance(meta["ids_path"], bool)
        assert isinstance(meta["contigs"], list) and len(meta["contigs"]) >= 2

    def test_window_larger_than_contigs_raises(self, tmp_path: Path, toy_fasta: Path):
        # Window larger than both contigs -> nothing to index -> error
        cfg = IndexConfig(window=5000, stride=64, rc_index=False)
        out_dir = tmp_path / "idx_too_big"

        idx = HELIndexer(toy_fasta, cfg, embedder="deterministic")
        with pytest.raises(RuntimeError):
            idx.build_or_load(out_dir, reuse_existing=False, include_dataset=True, verbose=False)
