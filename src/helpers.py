import time
import threading
import logging
import statistics
from typing import Dict, List, Optional

# -- Optional NVML (preferred); falls back to nvidia-smi if unavailable
try:
    import pynvml  # pip install nvidia-ml-py
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

import subprocess

import pysam  # noqa: F401  (kept to match your original imports)
from src.ReadAssembler import ReadAssembler, QueryConfig, revcomp  # noqa: F401

LOG_GPU_EVERY_SAMPLE = True  # log each sample line-by-line

# --------------------------
# GPU Monitor
# --------------------------
def _as_text(x):
    return x.decode("utf-8", errors="ignore") if isinstance(x, (bytes, bytearray)) else str(x)

class GPUMonitor(threading.Thread):
    """
    Samples GPU utilization/memory/power across all visible devices.
    Prefers NVML, otherwise falls back to `nvidia-smi`.
    """
    def __init__(self, interval: float = 1.0, logger: Optional[logging.Logger] = None):
        super().__init__(daemon=True)
        self.interval = interval
        self.logger = logger
        self._stop_evt = threading.Event()
        self.samples: Dict[int, List[dict]] = {}  # dev -> list of dicts

        self._use_nvml = _HAS_NVML
        self._nvml_handles = []
        if self._use_nvml:
            try:
                pynvml.nvmlInit()
                count = pynvml.nvmlDeviceGetCount()
                for i in range(count):
                    self._nvml_handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
                    self.samples[i] = []
                if self.logger:
                    names = []
                    for h in self._nvml_handles:
                        name_raw = pynvml.nvmlDeviceGetName(h)  # bytes on some systems, str on others
                        names.append(_as_text(name_raw))
                    self.logger.info(f"NVML initialized: detected {count} GPU(s): {names}")
            except Exception as e:
                self._use_nvml = False
                if self.logger:
                    self.logger.warning(f"NVML init failed ({e}); will fall back to nvidia-smi.")
        if not self._use_nvml:
            # Try to get device count from nvidia-smi
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                gpu_indices = [int(x.strip()) for x in out.strip().splitlines() if x.strip() != ""]
                for i in gpu_indices:
                    self.samples[i] = []
                if self.logger:
                    self.logger.info(f"nvidia-smi fallback: detected {len(gpu_indices)} GPU(s): {gpu_indices}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"nvidia-smi not available or failed ({e}). GPU sampling disabled.")
                self.samples = {}

    def stop(self):
        self._stop_evt.set()

    def _poll_nvml_once(self) -> Dict[int, dict]:
        now = time.time()
        data = {}
        for i, h in enumerate(self._nvml_handles):
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                power = None
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # W
                except Exception:
                    power = None
                temp = None
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    temp = None
                clocks = {}
                try:
                    sm = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                    memclk = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
                    clocks = {"sm_clock_MHz": sm, "mem_clock_MHz": memclk}
                except Exception:
                    pass

                rec = {
                    "t": now,
                    "util_pct": util.gpu,
                    "mem_used_MB": mem.used / (1024 ** 2),
                    "mem_total_MB": mem.total / (1024 ** 2),
                    "power_W": power,
                    "temp_C": temp,
                    **clocks,
                }
                data[i] = rec
            except Exception:
                # Skip device on error
                continue
        return data

    def _poll_smi_once(self) -> Dict[int, dict]:
        now = time.time()
        # Query utilization, mem used/total, power, temperature, clocks in a single call.
        # Note: some fields may be N/A on certain systems.
        q = [
            "index",
            "utilization.gpu",
            "memory.used",
            "memory.total",
            "power.draw",
            "temperature.gpu",
            "clocks.sm",
            "clocks.mem",
        ]
        cmd = ["nvidia-smi", f"--query-gpu={','.join(q)}", "--format=csv,noheader,nounits"]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        data = {}
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(q):
                continue
            idx = int(parts[0])
            def _to_float(x):
                try:
                    return float(x)
                except Exception:
                    return None
            rec = {
                "t": now,
                "util_pct": _to_float(parts[1]),
                "mem_used_MB": _to_float(parts[2]),
                "mem_total_MB": _to_float(parts[3]),
                "power_W": _to_float(parts[4]),
                "temp_C": _to_float(parts[5]),
                "sm_clock_MHz": _to_float(parts[6]),
                "mem_clock_MHz": _to_float(parts[7]),
            }
            data[idx] = rec
        return data

    def run(self):
        if not self.samples:
            # No devices found or tools unavailable
            if self.logger:
                self.logger.info("GPU sampling disabled (no devices or tools).")
            return
        poll_fn = self._poll_nvml_once if self._use_nvml else self._poll_smi_once
        while not self._stop_evt.is_set():
            try:
                data = poll_fn()
                for idx, rec in data.items():
                    self.samples[idx].append(rec)
                    if LOG_GPU_EVERY_SAMPLE and self.logger:
                        used = rec.get("mem_used_MB")
                        tot = rec.get("mem_total_MB")
                        util = rec.get("util_pct")
                        pwr = rec.get("power_W")
                        tmp = rec.get("temp_C")
                        smc = rec.get("sm_clock_MHz")
                        mmc = rec.get("mem_clock_MHz")
                        self.logger.info(
                            f"[GPU {idx}] util={util:.0f}% | mem={used:.0f}/{tot:.0f} MB | "
                            f"power={pwr if pwr is not None else 'NA'} W | "
                            f"temp={tmp if tmp is not None else 'NA'} C | "
                            f"sm_clk={smc if smc is not None else 'NA'} MHz | "
                            f"mem_clk={mmc if mmc is not None else 'NA'} MHz"
                        )
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"GPU sampling error: {e}")
            finally:
                # Wait with interruptibility
                if self._stop_evt.wait(self.interval):
                    break

    def summarize(self) -> Dict[int, dict]:
        summary = {}
        for idx, rows in self.samples.items():
            if not rows:
                summary[idx] = {}
                continue
            def stats(key):
                vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
                if not vals:
                    return {"min": None, "max": None, "mean": None}
                return {
                    "min": min(vals),
                    "max": max(vals),
                    "mean": statistics.fmean(vals),
                }
            summary[idx] = {
                "util_pct": stats("util_pct"),
                "mem_used_MB": stats("mem_used_MB"),
                "power_W": stats("power_W"),
                "temp_C": stats("temp_C"),
                "sm_clock_MHz": stats("sm_clock_MHz"),
                "mem_clock_MHz": stats("mem_clock_MHz"),
                "samples": len(rows),
                "duration_sec": (rows[-1]["t"] - rows[0]["t"]) if len(rows) >= 2 else 0.0,
            }
        return summary

    def close(self):
        try:
            if self._use_nvml:
                pynvml.nvmlShutdown()
        except Exception:
            pass

# --------------------------
# Simple timer helpers
# --------------------------
class StageTimer:
    def __init__(self, name: str, logger: logging.Logger):
        self.name = name
        self.logger = logger
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.logger.info(f"[TIMER] {self.name} started.")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        self.logger.info(f"[TIMER] {self.name} finished in {dt:.3f} s.")

t_total_start = time.perf_counter()