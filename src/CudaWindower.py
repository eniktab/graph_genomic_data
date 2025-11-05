# src/utils/window_cuda.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class CudaWindower:
    """
    Vectorized CUDA window maker for token id sequences.
    Works with DNALUTEncoder's reverse-complement id LUT.

    - as_windows(ids, win, stride): [N, win] view (contiguous) over ids on CUDA
    - two_strands(ids, win, stride, rc_lut): concat + strand/meta
    - iter_batches(...): yield fixed-size batches (for embedding model)
    """
    device: torch.device

    @staticmethod
    def _check(ids: torch.Tensor) -> None:
        if ids.device.type != "cuda":
            raise ValueError("expected CUDA tensor for ids")
        if ids.dtype != torch.long:
            raise ValueError("expected ids dtype=torch.long")
        if ids.ndim != 1:
            raise ValueError("expected ids shape [L]")

    @torch.inference_mode()
    def as_windows(self, ids: torch.Tensor, win: int, stride: int) -> torch.Tensor:
        """
        Return [N, win] windows using a single as_strided view, then materialize.
        """
        self._check(ids)
        L = ids.numel()
        if win <= 0 or stride <= 0:
            raise ValueError("win and stride must be > 0")
        if L < win:
            return ids.new_empty((0, win))  # [0, win]

        # Number of windows with integer stepping
        N = 1 + (L - win) // stride
        # Create a strided view over the 1-D buffer
        base_stride = ids.stride(0)  # usually 1
        windows = torch.as_strided(
            ids,
            size=(N, win),
            stride=(stride * base_stride, base_stride),
        )
        # Materialize to a compact, row-major [N, win] tensor for the model
        return windows.contiguous()

    @torch.inference_mode()
    def two_strands(
        self,
        ids: torch.Tensor,
        win: int,
        stride: int,
        rc_id_lut_cuda: torch.Tensor,
        return_meta: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Make windows for + and - strands on GPU.

        Returns:
            windows_2x:  [2*N, win]
            starts_2x:   [2*N]   (bp starts for each window)
            strands_2x:  [2*N]   (+1 for forward, -1 for reverse)
        """
        self._check(ids)
        if rc_id_lut_cuda.device.type != "cuda":
            raise ValueError("rc_id_lut_cuda must be on CUDA")

        # + strand
        w_fwd = self.as_windows(ids, win, stride)  # [N, win]

        # - strand: complement id-wise, reverse each window
        # Do whole contig RC once, then window like forward
        comp_all = rc_id_lut_cuda[ids]        # [L]
        rc_all   = comp_all.flip(0)           # [L] reverse
        w_rev    = self.as_windows(rc_all, win, stride)  # [N, win]

        # Concatenate in a stable order: all +, then all -
        windows = torch.cat([w_fwd, w_rev], dim=0)  # [2N, win]

        if not return_meta:
            return windows, None, None

        # Starts/strands metadata (int32 to save space)
        N = w_fwd.size(0)
        starts = torch.arange(N, device=ids.device, dtype=torch.int64) * stride
        starts_2x  = torch.cat([starts, starts], dim=0)  # [2N]
        strands_2x = torch.cat([
            torch.ones(N,  device=ids.device, dtype=torch.int8),   # +1
            -torch.ones(N, device=ids.device, dtype=torch.int8),   # -1
        ], dim=0)
        return windows, starts_2x, strands_2x

    @torch.inference_mode()
    def iter_batches(
        self,
        windows: torch.Tensor,
        batch: int,
    ):
        """
        Yield [B, win] CUDA batches. `windows` must already be CUDA and contiguous.
        """
        if windows.device.type != "cuda":
            raise ValueError("windows must be on CUDA")
        if not windows.is_contiguous():
            windows = windows.contiguous()
        n = windows.size(0)
        if n == 0:
            return
        for i in range(0, n, batch):
            yield windows[i:i + batch]
