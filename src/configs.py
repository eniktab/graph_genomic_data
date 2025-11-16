from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple
import logging

# ------------------------------
# IndexConfig & HELIndexer
# ------------------------------
@dataclass
class IndexConfig:
    # --- reference chunking / embeddings ---
    window: int = 10_000
    stride: int = 2_500
    ids_max_tokens_per_call: int = 262_144
    rc_index: bool = True
    model_name: str = "hyenadna-large-1m-seqlen-hf"
    model_dir: str = "/g/data/te53/en9803/data/scratch/hf-cache/models/"

    # --- CAGRA build/search knobs (RaftGPU) ---
    build_algo: Literal["nn_descent", "ivf_pq"] = "nn_descent"
    ann_metric: Literal["cosine", "sqeuclidean"] = "cosine"

    # NN-Descent parameters
    graph_degree: int = 128
    intermediate_graph_degree: int = 192
    nn_descent_niter: int = 20

    # IVF-PQ parameters
    ivf_n_lists: Optional[int] = None
    ivf_n_probes: Optional[int] = None
    ivf_pq_dim: int = 64
    ivf_pq_bits: int = 8
    refinement_rate: float = 2.0

    # Search parameter
    search_itopk_size: int = 128

    def __post_init__(self):
        if self.window <= 0 or self.stride <= 0:
            raise ValueError("window and stride must be > 0")
        if self.stride > self.window:
            raise ValueError("stride must be ≤ window")
        if self.build_algo not in ("nn_descent", "ivf_pq"):
            raise ValueError("build_algo must be 'nn_descent' or 'ivf_pq'")
        if self.ann_metric not in ("cosine", "sqeuclidean"):
            raise ValueError("ann_metric must be 'cosine' or 'sqeuclidean'")
        if self.build_algo == "ivf_pq":
            if self.ivf_n_lists is not None and self.ivf_n_lists <= 0:
                raise ValueError("ivf_n_lists must be positive")
            if self.ivf_n_probes is not None and self.ivf_n_probes <= 0:
                raise ValueError("ivf_n_probes must be positive")

