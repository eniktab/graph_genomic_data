
from typing import Iterator, Tuple

def read_fasta(path) -> Iterator[Tuple[str, str]]:
    """Yield (header, sequence) from a FASTA file (single or multi-line)."""
    name = None
    seq_chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            if line.startswith('>'):
                if name is not None:
                    yield name, ''.join(seq_chunks).upper()
                name = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if name is not None:
            yield name, ''.join(seq_chunks).upper()

