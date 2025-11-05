# wfa_gpu/wfagpu.py  (only the helpers and _align_cli changed vs previous version)
import ctypes, json, os, re, shlex, shutil, subprocess, tempfile, sys
from pathlib import Path
from typing import Optional, Union, Dict, Any

_LIB_ENV = "WFAGPU_PYSHIM"
_BIN_ENV = "WFAGPU_BIN"
_LIBDIR_ENV = "WFAGPU_LIBDIR"

def _lib_suffixes():
    out = [".so"]
    if sys.platform.startswith("darwin"):
        out = [".dylib", ".so"]
    elif os.name == "nt":
        out = [".dll", ".so"]
    return out

def _search_upwards(start: Path, rels, max_up: int = 4) -> Optional[Path]:
    cur = start
    for _ in range(max_up):
        for rel in rels:
            cand = (cur / rel).resolve()
            if cand.exists():
                return cand
        cur = cur.parent
    return None

def _auto_find_shim() -> Optional[str]:
    env = os.environ.get(_LIB_ENV)
    if env and Path(env).exists():
        return str(Path(env).resolve())
    here = Path(__file__).resolve().parent
    lib_names = [f"libwfagpu_pyshim{ext}" for ext in _lib_suffixes()]
    local = _search_upwards(here, [*lib_names,
                                   *[Path("lib")/n for n in lib_names],
                                   *[Path("build")/n for n in lib_names]], max_up=3)
    if local:
        return str(local)
    for name in lib_names:
        try:
            ctypes.CDLL(name)
            return name
        except OSError:
            pass
    return None

def _auto_find_bin() -> Optional[str]:
    env = os.environ.get(_BIN_ENV)
    if env and Path(env).exists():
        return str(Path(env).resolve())
    here = Path(__file__).resolve().parent
    bin_names = ["wfa.affine.gpu", "wfa.affine.gpu.exe"]
    cand = _search_upwards(here, [Path("bin")/n for n in bin_names] + [Path(n) for n in bin_names], max_up=4)
    if cand:
        return str(cand)
    for n in bin_names:
        which = shutil.which(n)
        if which:
            return which
    return None

# ---------- robust CIGAR parsing & affine scoring ----------
# Accept full SAM op set, case-insensitive
_CIGAR_RE = re.compile(r"\b(?:\d+[MIDNSHP=XB])+\b", re.IGNORECASE)
_CIGAR_TOK_RE = re.compile(r"(\d+)([MIDNSHP=XBmidnshp=xb])")

def _parse_cigar_any(text: str) -> Optional[str]:
    """Find the last CIGAR-looking token in text (stderr+stdout)."""
    m = _CIGAR_RE.findall(text)
    return m[-1] if m else None

def _cigar_tokens(cigar: str):
    """Return normalized upper-case tokens [(n, op)], tolerate lower-case."""
    return [(int(n), op.upper()) for n, op in _CIGAR_TOK_RE.findall(cigar or "")]

def _affine_from_cigar(cigar: str, x: int, o: int, e: int) -> Optional[int]:
    """
    Compute affine distance from a CIGAR:
      - mismatch cost: x per 'X'
      - gaps: o + e*len per contiguous run of 'I' or 'D'
      - matches 'M' or '=' cost 0
    If the CIGAR has only 'M' (no 'X') we can't count mismatches—returns 0 + gap cost.
    """
    if not cigar:
        return None
    toks = _cigar_tokens(cigar)
    # mismatches
    score = sum(n for n, op in toks if op == "X") * int(x)
    # gaps (contiguous runs)
    prev = None
    run = 0
    for n, op in toks:
        if op in ("I", "D"):
            if prev == op:
                run += n
            else:
                if prev in ("I", "D"):
                    score += int(o) + int(e) * run
                prev = op
                run = n
        else:
            if prev in ("I", "D"):
                score += int(o) + int(e) * run
            prev = op
            run = 0
    if prev in ("I", "D"):
        score += int(o) + int(e) * run
    return score

def _collect_lib_dirs(bin_path: Optional[str]) -> list[str]:
    dirs: list[Path] = []
    here = Path(__file__).resolve().parent
    dirs += [here, here / "build", here / "lib", here / "external" / "WFA2-lib" / "lib"]
    if bin_path:
        b = Path(bin_path).resolve()
        dirs += [b.parent, b.parent.parent, b.parent / ".." / "build", b.parent / ".." / "external" / "WFA2-lib" / "lib"]
    if os.environ.get(_LIBDIR_ENV):
        for p in os.environ[_LIBDIR_ENV].split(":"):
            if p.strip():
                dirs.append(Path(p.strip()).resolve())
    seen = set(); out: list[str] = []
    for d in dirs:
        try:
            rp = str(d.resolve())
        except Exception:
            continue
        if os.path.isdir(rp) and rp not in seen:
            seen.add(rp); out.append(rp)
    return out

def _env_with_libdirs(base_env: Optional[dict], libdirs: list[str]) -> dict:
    env = dict(base_env or os.environ)
    if os.name != "nt" and not sys.platform.startswith("darwin"):
        ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join([*libdirs, ld]) if ld else ":".join(libdirs)
    if sys.platform.startswith("darwin"):
        dy = env.get("DYLD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = ":".join([*libdirs, dy]) if dy else ":".join(libdirs)
    if os.name == "nt":
        path = env.get("PATH", "")
        env["PATH"] = os.pathsep.join([*libdirs, path]) if path else os.pathsep.join(libdirs)
    return env
# ----------------------------------------------------------

class WFAGPU:
    def __init__(self, lib_path: Optional[str] = None, bin_path: Optional[str] = None):
        self.lib_path = lib_path or _auto_find_shim()
        self.bin_path = bin_path or _auto_find_bin()
        self._lib = None
        if self.lib_path:
            try:
                self._lib = ctypes.CDLL(self.lib_path)
                self._lib.wfagpu_align_pair_json.restype = ctypes.c_char_p
                self._lib.wfagpu_align_pair_json.argtypes = [
                    ctypes.c_char_p, ctypes.c_char_p,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int
                ]
                self._lib.wfagpu_free_string.restype = None
                self._lib.wfagpu_free_string.argtypes = [ctypes.c_char_p]
            except OSError:
                self._lib = None

    def align(
        self,
        query: str,
        target: str,
        *,
        x: int = 4, o: int = 6, e: int = 2,
        compute_cigar: bool = True,
        batch_size: Optional[int] = None,
        max_distance: Optional[int] = None,
        band: Union[None, str, int] = None,
        threads_per_block: Optional[int] = None,
        workers: Optional[int] = None,
        check: bool = False,
    ) -> Dict[str, Any]:
        wants_cli = any(v is not None for v in (max_distance, threads_per_block, workers)) \
                    or (band is not None) or check
        if wants_cli or self._lib is None:
            return self._align_cli(
                query, target, x, o, e, compute_cigar, batch_size,
                max_distance, band, threads_per_block, workers, check
            )
        return self._align_lib(query, target, x, o, e, compute_cigar, batch_size)

    def _align_lib(self, query, target, x, o, e, compute_cigar, batch_size):
        raw = self._lib.wfagpu_align_pair_json(
            query.encode(), target.encode(),
            int(x), int(o), int(e),
            1 if compute_cigar else 0,
            int(batch_size if batch_size is not None else -1),
        )
        if not raw:
            raise RuntimeError("wfagpu_align_pair_json returned NULL")
        try:
            return json.loads(raw.decode("utf-8"))
        finally:
            self._lib.wfagpu_free_string(raw)

    def _align_cli(self, query, target, x, o, e, compute_cigar, batch_size,
                   max_distance, band, threads_per_block, workers, check):
        bin_path = self.bin_path or shutil.which("wfa.affine.gpu")
        if not bin_path:
            raise FileNotFoundError("WFA-GPU binary not found. Set WFAGPU_BIN or ensure it is on PATH.")

        libdirs = _collect_lib_dirs(bin_path)
        env = _env_with_libdirs(os.environ, libdirs)

        with tempfile.TemporaryDirectory() as td:
            qf, tf = Path(td)/"q.fasta", Path(td)/"t.fasta"
            qf.write_text(">q\n"+query+"\n"); tf.write_text(">t\n"+target+"\n")

            cmd = [bin_path, "-Q", str(qf), "-T", str(tf), "-g", f"{x},{o},{e}", "-p"]
            if compute_cigar: cmd.append("-x")
            if batch_size is not None: cmd += ["-b", str(int(batch_size))]
            if max_distance is not None: cmd += ["-e", str(int(max_distance))]
            if band is not None:
                if isinstance(band, str) and band.lower() == "auto": cmd += ["-B", "auto"]
                else: cmd += ["-B", str(int(band))]
            if threads_per_block is not None: cmd += ["-t", str(int(threads_per_block))]
            if workers is not None: cmd += ["-w", str(int(workers))]
            if check: cmd += ["-c"]

            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")

            # Handle loader errors nicely
            if "error while loading shared libraries" in out or (proc.returncode != 0 and "libwfagpu.so" in out):
                raise RuntimeError(
                    "The WFA-GPU binary could not load its shared libraries.\n"
                    f"Tried library dirs: {libdirs}\n"
                    f"CMD: {shlex.join(cmd)}\nOUTPUT:\n{out}"
                )

            cigar = _parse_cigar_any(out) if compute_cigar else None

            # NEW: compute a consistent affine score from the CIGAR itself.
            score = None
            if cigar:
                score = _affine_from_cigar(cigar, x, o, e)
            # Fallback (if no CIGAR found): try to find a labeled score/error/distance
            if score is None:
                m = re.findall(r"(?i)(?:score|error|distance)\D+(\d+)", out)
                score = int(m[-1]) if m else 0  # last labeled number, else 0

            return {
                "ok": True,
                "score": int(score),
                "error": int(score),
                "cigar": cigar,
                "query_len": len(query),
                "target_len": len(target),
                "backend": "cli"
            }

# Module-level convenience instance & function
wfa_gpu = WFAGPU()

def align(query: str, target: str, **kwargs) -> Dict[str, Any]:
    return wfa_gpu.align(query, target, **kwargs)
