# conftest.py
# Conditional test backend:
# - If real CuPy + cuVS are available (and a CUDA device exists), use them.
# - Otherwise, install CPU-only fakes for `cupy` and `cuvs.neighbors.{cagra,ivf_pq}`.
# - Adds /mnt/data to sys.path so we can import your RaftGPU.py.

import sys, types, json, os

def _try_real_gpu_stack():
    try:
        import cupy as _cp
        from cuvs.neighbors import cagra as _cagra
        # Check for at least one CUDA device
        try:
            ndev = _cp.cuda.runtime.getDeviceCount()
            if ndev <= 0:
                return None
        except Exception:
            return None
        # Heuristic: if the cagra module is importable from cuvs, we assume real
        return {"cp": _cp, "cagra": _cagra}
    except Exception:
        return None

_real = _try_real_gpu_stack()

if _real is None:
    # ----------- Install CPU fakes -----------
    import numpy as _np

    cupy = types.ModuleType("cupy")
    cupy.ndarray = _np.ndarray
    cupy.float32 = _np.float32

    def _asarray(x, dtype=None):
        return _np.asarray(x, dtype=dtype)

    def _ascontiguousarray(x, dtype=None):
        return _np.ascontiguousarray(x, dtype=dtype)

    def _empty(shape, dtype=_np.float32, order="C"):
        return _np.empty(shape, dtype=dtype, order=order)

    def _asnumpy(x):
        return _np.asarray(x)

    class _Linalg(types.SimpleNamespace):
        norm = staticmethod(_np.linalg.norm)

    cupy.asarray = _asarray
    cupy.ascontiguousarray = _ascontiguousarray
    cupy.empty = _empty
    cupy.asnumpy = _asnumpy
    cupy.linalg = _Linalg()

    class _Stream:
        def __init__(self):
            self.ptr = 0
        def synchronize(self):
            pass

    class _Runtime:
        memcpyDeviceToDevice = 0
        def memcpy2DAsync(self, *args, **kwargs):
            return None

    class _Cuda(types.SimpleNamespace):
        def get_current_stream(self):
            return _Stream()
    cupy.cuda = _Cuda()
    sys.modules["cupy"] = cupy

    # Fake cuVS
    def _topk_inner_product(Q, V, k):
        scores = Q @ V.T
        I = _np.argsort(-scores, axis=1)[:, :k]
        S = _np.take_along_axis(scores, I, axis=1)
        return S, I

    def _topk_sqeuclidean(Q, V, k):
        Qa = _np.sum(Q * Q, axis=1, keepdims=True)
        Vb = _np.sum(V * V, axis=1, keepdims=True).T
        d2 = Qa + Vb - 2.0 * (Q @ V.T)
        I = _np.argsort(d2, axis=1)[:, :k]
        S = _np.take_along_axis(d2, I, axis=1)
        return S, I

    cagra = types.ModuleType("cuvs.neighbors.cagra")
    setattr(cagra, "__FAKE__", True)

    class _Index:
        def __init__(self, dataset=None, dim=0, metric="inner_product"):
            self.dataset = dataset
            self.dim = int(dim or (dataset.shape[1] if dataset is not None else 0))
            self.metric = metric
        def update_dataset(self, V):
            self.dataset = V
            self.dim = V.shape[1]

    def build(params, V):
        return _Index(dataset=_np.asarray(V), metric=getattr(params, "metric", "inner_product"))

    def search(params, index, Q, k):
        V = _np.asarray(index.dataset)
        Q = _np.asarray(Q)
        if index.metric == "inner_product":
            return _topk_inner_product(Q, V, int(k))
        else:
            return _topk_sqeuclidean(Q, V, int(k))

    def save(path, index, include_dataset=True):
        info = {"metric": index.metric, "dim": index.dim, "include_dataset": bool(include_dataset)}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f)
        if include_dataset and index.dataset is not None:
            _np.save(path + ".dataset.npy", _np.asarray(index.dataset, dtype=_np.float32))

    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            info = json.load(f)
        dataset = None
        ds_path = path + ".dataset.npy"
        if os.path.exists(ds_path):
            dataset = _np.load(ds_path)
        return _Index(dataset=dataset, dim=info.get("dim", 0), metric=info.get("metric", "inner_product"))

    def update_dataset(index, V):
        index.dataset = _np.asarray(V)
        index.dim = index.dataset.shape[1]

    class IndexParams:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class SearchParams:
        def __init__(self, itopk_size=128, **kwargs):
            self.itopk_size = int(itopk_size)
            for k, v in kwargs.items():
                setattr(self, k, v)

    cagra.build = build
    cagra.search = search
    cagra.save = save
    cagra.load = load
    cagra.update_dataset = update_dataset
    cagra.IndexParams = IndexParams
    cagra.SearchParams = SearchParams

    ivf_pq = types.ModuleType("cuvs.neighbors.ivf_pq")
    setattr(ivf_pq, "__FAKE__", True)

    class _IVFIndexParams:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _IVFSearchParams:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    ivf_pq.IndexParams = _IVFIndexParams
    ivf_pq.SearchParams = _IVFSearchParams

    cuvs = types.ModuleType("cuvs")
    neighbors = types.ModuleType("cuvs.neighbors")
    neighbors.cagra = cagra
    neighbors.ivf_pq = ivf_pq

    sys.modules["cuvs"] = cuvs
    sys.modules["cuvs.neighbors"] = neighbors
    sys.modules["cuvs.neighbors.cagra"] = cagra
    sys.modules["cuvs.neighbors.ivf_pq"] = ivf_pq

# Ensure we can import your module
if "/mnt/data" not in sys.path:
    sys.path.insert(0, "/mnt/data")
