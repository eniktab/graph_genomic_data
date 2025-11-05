# tests/test_wfagpu_wrapper.py
import re
import sys
import traceback
from typing import List, Tuple, Callable, Dict

from wfa_gpu.wfagpu import wfa_gpu

class SkipTest(Exception):
    pass

def _skip(msg: str):
    try:
        import pytest  # type: ignore
        pytest.skip(msg)
    except Exception:
        raise SkipTest(msg)

# Case-insensitive SAM ops
_CIGAR_TOK_RE = re.compile(r"(\d+)([MIDNSHP=XBmidnshp=xb])")

def parse_cigar(cigar: str):
    if cigar is None:
        raise AssertionError("Expected a CIGAR, got None")
    parts = _CIGAR_TOK_RE.findall(cigar)
    assert parts, f"Cannot parse CIGAR: {cigar!r}"
    return [(int(n), op.upper()) for n, op in parts]

def consume_lengths(tokens: List[tuple], *, orientation: str = "SAM") -> Tuple[int, int]:
    """
    Return (q_used, t_used).
    orientation="SAM": I->query, D/N->target
    orientation="SWAP": I->target, D/N->query  (some builds flip roles)
    """
    q = t = 0
    for n, op in tokens:
        if op in ("M", "=", "X"):
            q += n; t += n
        elif op == "I":
            if orientation == "SAM": q += n
            else: t += n
        elif op in ("D", "N"):
            if orientation == "SAM": t += n
            else: q += n
        elif op == "S":
            q += n
        # H/P/B consume neither
    return q, t

def cigar_summary(tokens: List[tuple]) -> Dict[str, int]:
    """Normalize CIGAR into totals; ‘I’ and ‘D’ kept separate (we’ll allow swap in comparisons)."""
    total_M = sum(n for n, op in tokens if op in ("M", "=", "X"))
    total_I = sum(n for n, op in tokens if op == "I")
    total_D = sum(n for n, op in tokens if op == "D")
    return {"M": total_M, "I": total_I, "D": total_D}

def spans_either_orientation(tokens: List[tuple], q_len: int, t_len: int) -> bool:
    q1, t1 = consume_lengths(tokens, orientation="SAM")
    q2, t2 = consume_lengths(tokens, orientation="SWAP")
    return (q1 == q_len and t1 == t_len) or (q2 == q_len and t2 == t_len)

def _ready_or_skip():
    try:
        _ = wfa_gpu.align("A", "A", x=2, o=3, e=1, compute_cigar=True, batch_size=1)
    except FileNotFoundError as e:
        _skip(f"WFA-GPU binary not found / not runnable: {e}")
    except RuntimeError as e:
        if "could not load its shared libraries" in str(e):
            _skip(f"WFA-GPU shared libraries not found at runtime: {e}")
        raise

def _print_case(tag: str, res: Dict, q: str, t: str, bs: int):
    print(f"{tag} | batch={bs} | backend={res.get('backend','lib')} | "
          f"score={res['score']} | cigar={res['cigar']} | q_len={len(q)} t_len={len(t)}")

# -------- tests --------

def test_score_ordering():
    _ready_or_skip()
    x, o, e = 2, 3, 1
    same_q = same_t = "GATTACA"
    single_q, single_t = "GATTACA", "GACTACA"
    diff_q, diff_t     = "AAAAAA", "TTTTTT"

    r_same   = wfa_gpu.align(same_q, same_t, x=x, o=o, e=e, compute_cigar=True, batch_size=1)
    r_single = wfa_gpu.align(single_q, single_t, x=x, o=o, e=e, compute_cigar=True, batch_size=1)
    r_diff   = wfa_gpu.align(diff_q, diff_t, x=x, o=o, e=e, compute_cigar=True, batch_size=1)

    _print_case("same",   r_same,   same_q,   same_t,   1)
    _print_case("single", r_single, single_q, single_t, 1)
    _print_case("diff",   r_diff,   diff_q,   diff_t,   1)

    assert r_same["score"] <= r_single["score"] <= r_diff["score"], (
        f"Expected score ordering same<=single<=diff, got: "
        f"{r_same['score']} !<= {r_single['score']} !<= {r_diff['score']}"
    )

def test_cigar_same_strings_shape():
    _ready_or_skip()
    q = t = "GATTACA"
    res = wfa_gpu.align(q, t, x=2, o=3, e=1, compute_cigar=True, batch_size=1)
    _print_case("cigar-same", res, q, t, 1)
    toks = parse_cigar(res["cigar"])
    assert spans_either_orientation(toks, len(q), len(t))
    assert all(op in ("M", "=") for _, op in toks), f"Expected only matches in CIGAR, got {res['cigar']}"

def test_cigar_single_difference_spans_full_length():
    _ready_or_skip()
    q, t = "GATTACA", "GACTACA"
    res = wfa_gpu.align(q, t, x=2, o=3, e=1, compute_cigar=True, batch_size=1)
    _print_case("cigar-single", res, q, t, 1)
    toks = parse_cigar(res["cigar"])
    assert spans_either_orientation(toks, len(q), len(t)), f"CIGAR doesn't span both sequences: {res['cigar']}"

def test_cigar_all_different_lengths_consumed():
    _ready_or_skip()
    q, t = "AAAAAA", "TTTTTT"
    res = wfa_gpu.align(q, t, x=2, o=3, e=1, compute_cigar=True, batch_size=1)
    _print_case("cigar-all-diff", res, q, t, 1)
    toks = parse_cigar(res["cigar"])
    assert spans_either_orientation(toks, len(q), len(t)), f"CIGAR doesn't span full sequences: {res['cigar']}"

def test_batching_invariance_per_pair():
    _ready_or_skip()
    cases = [
        ("same",   "GATTACA", "GATTACA"),
        ("single", "GATTACA", "GACTACA"),
        ("diff",   "AAAAAA",  "TTTTTT"),
        ("indel",  "ACGTACGT","ACGTTACGT"),
    ]
    for name, q, t in cases:
        r1 = wfa_gpu.align(q, t, x=2, o=3, e=1, compute_cigar=True, batch_size=1)
        r2 = wfa_gpu.align(q, t, x=2, o=3, e=1, compute_cigar=True, batch_size=32)

        _print_case(f"batch-inv:{name}", r1, q, t, 1)
        _print_case(f"batch-inv:{name}", r2, q, t, 32)

        # scores must match exactly
        assert r1["score"] == r2["score"], f"Score changed with batch size for {name}: {r1['score']} vs {r2['score']}"

        # CIGARs fully span sequences (accept either orientation)
        toks1 = parse_cigar(r1["cigar"]); toks2 = parse_cigar(r2["cigar"])
        assert spans_either_orientation(toks1, len(q), len(t)), f"Batch1 CIGAR doesn't span: {r1['cigar']}"
        assert spans_either_orientation(toks2, len(q), len(t)), f"Batch32 CIGAR doesn't span: {r2['cigar']}"

        # CIGAR shape (M/I/D totals) should be the same; allow I<->D swap by comparing (I,D) as unordered
        s1 = cigar_summary(toks1); s2 = cigar_summary(toks2)
        assert s1["M"] == s2["M"], f"M total changed with batch size for {name}: {s1} vs {s2}"
        assert sorted((s1["I"], s1["D"])) == sorted((s2["I"], s2["D"])), (
            f"I/D totals changed with batch size for {name}:\n"
            f"batch=1  -> {r1['cigar']}  => {s1}\n"
            f"batch=32 -> {r2['cigar']}  => {s2}"
        )

# ---- friendly main runner ----
def _run(fn: Callable[[], None]):
    name = fn.__name__
    try:
        fn()
        return ("PASS", name)
    except SkipTest as e:
        return ("SKIP", f"{name} :: {e}")
    except AssertionError:
        return ("FAIL", f"{name}\n{traceback.format_exc()}")
    except Exception:
        return ("ERROR", f"{name}\n{traceback.format_exc()}")

def main() -> int:
    tests: List[Callable[[], None]] = [
        test_score_ordering,
        test_cigar_same_strings_shape,
        test_cigar_single_difference_spans_full_length,
        test_cigar_all_different_lengths_consumed,
        test_batching_invariance_per_pair,
    ]
    results = [_run(t) for t in tests]

    print("\n=== WFA-GPU wrapper tests ===")
    counts = {"PASS":0,"FAIL":0,"SKIP":0,"ERROR":0}
    for status, details in results:
        counts[status] += 1
        tag = {"PASS":"[PASS]","FAIL":"[FAIL]","SKIP":"[SKIP]","ERROR":"[ERROR]"}[status]
        print(f"{tag} {details}")

    print("\n--- Summary ---")
    print(f"PASS: {counts['PASS']}  FAIL: {counts['FAIL']}  SKIP: {counts['SKIP']}  ERROR: {counts['ERROR']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
