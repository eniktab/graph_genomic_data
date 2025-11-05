
from dataclasses import dataclass, field
from typing import List, Tuple
import hashlib
import numpy as np

def hash_bytes(b: bytes, nbytes: int=16) -> bytes:
    # Use BLAKE2s (stdlib) for deterministic content IDs
    h = hashlib.blake2s(b, digest_size=nbytes)
    return h.digest()

@dataclass
class WindowRecord:
    genome: str
    start: int
    end: int
    emb: np.ndarray

@dataclass
class Island:
    island_id: bytes
    centroid: np.ndarray
    windows: List[WindowRecord] = field(default_factory=list)

    @staticmethod
    def make_id(centroid_q: bytes, spec: str) -> bytes:
        return hash_bytes(centroid_q + spec.encode('utf-8'), nbytes=16)
