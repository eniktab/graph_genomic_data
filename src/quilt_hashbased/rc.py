
BASE_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")

def revcomp(seq: str) -> str:
    """Reverse-complement of a DNA string."""
    return seq.translate(BASE_COMP)[::-1]

def canon_kmer(kmer: str) -> str:
    """Return canonical representation: min(kmer, revcomp(kmer))."""
    rc = revcomp(kmer)
    return kmer if kmer <= rc else rc
