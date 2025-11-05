
from dataclasses import dataclass
import numpy as np
from .rc import canon_kmer


@dataclass(frozen=True)
class EmbeddingSpec:
    """
    Configuration for RC-invariant window embeddings built from a k-mer count sketch
    plus optional scalar features (GC%, Shannon entropy).

    Parameters
    ----------
    k : int, default=7
        k-mer length used to build the spectrum sketch. Larger `k` increases
        uniqueness but reduces hit rate in noisy/short reads.
        Examples:
          - k= 2-5 for toy data or tiny genomes (fast, coarse).
          - k=15–17 for HiFi/ONT assemblies (good balance).
          - k=21+ for large, repeat-rich genomes when reads are long.

    sketch_size : int, default=256
        Dimensionality of the count sketch vector. Prefer a power of two
        (256, 512, 1024, …) for fast bit-masking. Higher values reduce hash
        collisions at the cost of memory/CPU.
        Examples: 256 (demo), 1024 (human-scale), 4096 (very low collision).

    include_gc : bool, default=True
        If True, append a single scalar feature = GC fraction in the window
        (after dropping ‘N’s). Set False to omit GC from the embedding.

    include_entropy : bool, default=True
        If True, append a single scalar feature = base-level Shannon entropy
        (A/C/G/T distribution) scaled to [0, 1]. Useful to distinguish
        low-complexity/repetitive windows.

    rc_invariant : bool, default=True
        If True, use canonical k-mers (min(k-mer, reverse-complement)) so that
        embedding(seq) == embedding(revcomp(seq)). This is robust to inversions
        and mixed strand orientation in reads. Set False only if you need
        strand-aware embeddings.

    window_len : int, default=4000
        Length of the sliding window in bases.
        Examples:
          - 10 for toy data or tiny genomes (fast, coarse).
          - 2000 for compact bacterial genomes or short-read contexts.
          - 4000 (default) for general long-read/assembly use.
          - 8000+ to average over highly repetitive regions.

    stride : int, default=1000
        Step size (in bases) between consecutive windows.
        Examples:
          - stride=window_len for non-overlapping windows.
          - stride=1000 with window_len=4000 gives 75% overlap (denser index).

    Examples
    --------
    >>> spec = EmbeddingSpec(k=17, sketch_size=1024, window_len=4096, stride=1024)
    >>> # RC-invariant embedding suitable for long-read assemblies.
    """
    k:int = 15                # k-mer length for spectrum (small for demo; tune for real data)
    sketch_size:int = 256     # dimensionality of the count sketch (power of 2 preferred)
    include_gc:bool = True
    include_entropy:bool = True
    rc_invariant:bool = True  # use canonical k-mers so embedding matches reverse-complement
    window_len:int = 4000
    stride:int = 1000

class WindowEmbedder:
    """Compute RC-invariant composite embeddings per window using canonical k-mer sketch.
    This is Phase-1 friendly (no TF dependency) but stable and fast.
    """
    def __init__(self, spec: EmbeddingSpec = EmbeddingSpec()):
        self.spec = spec
        self._hash_mask = self.spec.sketch_size - 1
        if self.spec.sketch_size & self._hash_mask != 0:
            raise ValueError("sketch_size must be a power of 2 for fast hashing.")

    def _hash(self, s:str)->int:
        # Simple 64-bit FNV-1a then fold; deterministic and fast.
        h = 1469598103934665603
        for ch in s.encode('ascii', errors='ignore'):
            h ^= ch
            h *= 1099511628211
            h &= (1<<64)-1
        # final mix
        h ^= (h >> 33)
        h *= 0xff51afd7ed558ccd
        h &= (1<<64)-1
        return h

    def _kmer_sketch(self, seq:str)->np.ndarray:
        k = self.spec.k
        m = self.spec.sketch_size
        v = np.zeros(m, dtype=np.float32)
        if len(seq) < k:
            return v
        bad = 0
        for i in range(len(seq)-k+1):
            kmer = seq[i:i+k]
            if 'N' in kmer:
                bad += 1
                continue
            if self.spec.rc_invariant:
                kmer = canon_kmer(kmer)
            idx = self._hash(kmer) & (m-1)
            v[idx] += 1.0
        if v.sum() > 0:
            v /= (v.sum())  # L1-normalize for length invariance
        return v

    def _gc_entropy_feats(self, seq:str)->np.ndarray:
        if not (self.spec.include_gc or self.spec.include_entropy):
            return np.zeros((0,), np.float32)
        seq = seq.replace('N','')
        if not seq:
            gc = 0.0; ent = 0.0
        else:
            g = seq.count('G'); c = seq.count('C')
            gc = (g+c) / len(seq)
            # Shannon entropy over A/C/G/T
            counts = [seq.count(b) for b in 'ACGT']
            total = sum(counts) or 1
            p = [c/total for c in counts if c>0]
            ent = -sum([pi*np.log2(pi) for pi in p])
        feats = []
        if self.spec.include_gc: feats.append(gc)
        if self.spec.include_entropy: feats.append(ent/2.0)  # scale entropy (max 2) to 0..1
        return np.asarray(feats, dtype=np.float32)

    def windows(self, name:str, seq:str):
        L = len(seq)
        W, S = self.spec.window_len, self.spec.stride
        for start in range(0, max(1, L-W+1), S):
            end = min(L, start+W)
            yield (name, start, end, seq[start:end])

    def embed_seq(self, name:str, seq:str):
        """Yield (name, start, end, embedding) for windows across seq.
        RC-invariant: embedding(seq) == embedding(revcomp(seq)).
        """
        for (name, start, end, s) in self.windows(name, seq):
            sketch = self._kmer_sketch(s)
            feats = self._gc_entropy_feats(s)
            emb = np.concatenate([sketch, feats], axis=0)
            # L2-normalize
            n = np.linalg.norm(emb) or 1.0
            emb = emb / n
            yield (name, start, end, emb)

    def embed_fasta(self, fasta_path:str):
        from src.old.quilt import read_fasta
        for name, seq in read_fasta(fasta_path):
            for rec in self.embed_seq(name, seq):
                yield rec
