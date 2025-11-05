
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from .island import Island, WindowRecord, hash_bytes

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    da = (a / (np.linalg.norm(a) + 1e-9))
    db = (b / (np.linalg.norm(b) + 1e-9))
    return float(np.dot(da, db))

@dataclass
class IslandClusterer:
    tau: float = 0.88  # cosine threshold for same island (tune!)
    min_windows: int = 1

    def cluster(self, windows: List[WindowRecord]) -> List[Island]:
        """Brute-force agglomerative clustering by cosine threshold.
        For Phase-1 scale and clarity. Replace with HNSW in Phase-2.
        """
        clusters: List[List[int]] = []  # indices of windows
        centroids: List[np.ndarray] = []

        for i, w in enumerate(windows):
            placed = False
            for ci, c_inds in enumerate(clusters):
                c = centroids[ci]
                sim = cosine_sim(c, w.emb)
                if sim >= self.tau:
                    c_inds.append(i)
                    # update centroid
                    centroids[ci] = (c * (len(c_inds)-1) + w.emb) / len(c_inds)
                    placed = True
                    break
            if not placed:
                clusters.append([i])
                centroids.append(w.emb.copy())

        isles: List[Island] = []
        for ci, inds in enumerate(clusters):
            if len(inds) < self.min_windows:
                continue
            centroid = centroids[ci] / (np.linalg.norm(centroids[ci]) + 1e-9)
            # simple uint8 quantization of centroid for ID stability
            c_q = (np.round((centroid*0.5 + 0.5) * 255.0).astype(np.uint8)).tobytes()
            iid = hash_bytes(c_q, nbytes=16)
            island = Island(iid, centroid, [windows[j] for j in inds])
            isles.append(island)
        return isles

def stitch_islands(islands: List[Island], max_gap:int=2000) -> List[Island]:
    """Within each genome, merge adjacent windows from the same cluster if they
    lie within max_gap, creating longer spans. (Phase-1: keep windows list; stitching
    is used later when deriving spans.)
    """
    # Phase-1 keeps islands as clustered window sets; span-level stitching can be
    # done downstream. We return islands unchanged here, but the function is
    # provided for API compatibility.
    return islands
