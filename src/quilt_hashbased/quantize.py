
from dataclasses import dataclass
import numpy as np

@dataclass
class QuantParams:
    scale: float
    zero: float

class Quantizer:
    """Simple symmetric int8 quantizer with stored scale/zero."""
    def __init__(self):
        self.params = None

    def fit(self, X: np.ndarray):
        # X: [N,D] float32
        xmax = np.max(np.abs(X), axis=0) + 1e-8
        self.params = [QuantParams(scale=float(xm/127.0), zero=0.0) for xm in xmax]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.params is not None
        scales = np.array([p.scale for p in self.params], dtype=np.float32)
        Q = np.clip(np.round(X / scales), -127, 127).astype(np.int8)
        return Q

    def inverse_transform(self, Q: np.ndarray) -> np.ndarray:
        scales = np.array([p.scale for p in self.params], dtype=np.float32)
        return (Q.astype(np.float32) * scales)
