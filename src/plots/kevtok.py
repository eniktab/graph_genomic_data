# make_preprint_figures.py
# -----------------------------------------------------------------------------
# Produce preprint-ready figures (PDF only) from enhanced_tokenizer_benchmark.py.
# - No titles; figure meaning conveyed by filename.
# - X-axis tick labels rotated for readability where helpful.
# - Scientific notation for large values on all throughput axes and annotations.
# - Restrained, CS-journal-friendly color palette (colorblind-safe).
# - Violin plot (per-trial distributions) instead of scatter.
# - Can re-summarize from trials when needed for plotting.
# -----------------------------------------------------------------------------

from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Optional Seaborn styling (falls back to Matplotlib if not installed)
_USE_SEABORN = True
try:
    import seaborn as sns  # type: ignore
except Exception:
    _USE_SEABORN = False

# =========================== USER SETTINGS =======================================

RESULTS_DIR = "/g/data/te53/en9803/workspace/sync/ANU/graph_genomic_data/results"
OVERRIDE_SUMMARY_CSV: Optional[str] = None
OVERRIDE_TRIALS_CSV: Optional[str] = None
FIG_OUT = Path(RESULTS_DIR) / "figures_preprint"


# ======================== MATPLOTLIB / STYLE ====================================

def configure_matplotlib_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 600,
            "figure.autolayout": False,
            "figure.figsize": (5.0, 3.2),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
            "axes.titlesize": 10,  # titles unused by design
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "lines.markersize": 4.5,
            "lines.linewidth": 1.6,
        }
    )
    if _USE_SEABORN:
        # Clean, conservative style appropriate for CS venues
        sns.set_theme(context="paper", style="whitegrid", rc=mpl.rcParams)


# ============================ I/O HELPERS ========================================

def _latest_file(pattern: str) -> Optional[Path]:
    files = glob.glob(pattern)
    if not files:
        return None
    files = sorted(files, key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(files[0])


def discover_inputs(results_dir: str) -> Tuple[Path, Path]:
    if OVERRIDE_SUMMARY_CSV:
        summary: Optional[Path] = Path(OVERRIDE_SUMMARY_CSV)
    else:
        summary = _latest_file(str(Path(results_dir) / "benchmark_summary_*.csv"))

    if summary is None:
        raise FileNotFoundError("Could not find benchmark_summary_*.csv in RESULTS_DIR.")

    if OVERRIDE_TRIALS_CSV:
        trials: Optional[Path] = Path(OVERRIDE_TRIALS_CSV)
    else:
        stamp_match = re.findall(
            r"benchmark_summary_(\d{8}-\d{6})\.csv",
            str(summary.name),
        )
        if stamp_match:
            trials_candidate = Path(results_dir) / f"benchmark_trials_{stamp_match[0]}.csv"
            trials = (
                trials_candidate
                if trials_candidate.exists()
                else _latest_file(str(Path(results_dir) / "benchmark_trials_*.csv"))
            )
        else:
            trials = _latest_file(str(Path(results_dir) / "benchmark_trials_*.csv"))

    if trials is None:
        raise FileNotFoundError("Could not find benchmark_trials_*.csv in RESULTS_DIR.")

    return summary, trials


def ensure_out_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ============================ LABELING / MAPPING =================================

METHOD_RENAMES: Dict[str, str] = {
    # Hugging Face → simpler names
    "HF Rust (robust adapter)": "Hugging Face tokenizer",
    "HF Rust (encode_batch)": "Hugging Face tokenizer (batch)",

    # DNATok encode/staging names
    "DNATok (encode i64)": "DNATok encode (int64)",
    "DNATok (staging i64)": "DNATok staging (int64)",
    "DNATok (staging i32)": "DNATok staging (int32)",

    # Explicitly mark streaming modes as DNATok
    "Streaming baseline": "DNATok streaming (baseline)",
    "Streaming pipelined": "DNATok streaming (pipelined)",
}

# Short legend labels (line/sweep figs). Full names still appear in bar figs.
SHORT_METHOD_LABELS: Dict[str, str] = {
    "Hugging Face tokenizer": "Hugging Face",
    "Hugging Face tokenizer (batch)": "Hugging Face (batch)",
    "DNATok encode (int64)": "DNATok encode i64",
    "DNATok staging (int64)": "DNATok staging i64",
    "DNATok staging (int32)": "DNATok staging i32",
    "DNATok streaming (baseline)": "DNATok stream (baseline)",
    "DNATok streaming (pipelined)": "DNATok stream (pipelined)",
}

# CS-journal-friendly palette: restrained, high-contrast, colorblind-safe
def _build_method_colors() -> Dict[str, Tuple[float, float, float]]:
    keys = [
        "Hugging Face tokenizer",
        "Hugging Face tokenizer (batch)",
        "DNATok encode (int64)",
        "DNATok staging (int64)",
        "DNATok staging (int32)",
        "DNATok streaming (baseline)",
        "DNATok streaming (pipelined)",
    ]
    if _USE_SEABORN:
        # Start from seaborn "deep", desaturate slightly for print friendliness
        base = sns.color_palette("deep", len(keys))
        base = [sns.desaturate(c, 0.9) for c in base]
    else:
        # Fallback: muted custom palette (roughly IEEE/ACM-friendly)
        base = [
            (0.18, 0.31, 0.54),  # deep blue
            (0.80, 0.47, 0.13),  # muted orange
            (0.20, 0.49, 0.23),  # deep green
            (0.55, 0.27, 0.37),  # muted maroon
            (0.37, 0.35, 0.60),  # slate/violet
            (0.40, 0.40, 0.40),  # neutral gray
            (0.12, 0.55, 0.55),  # teal accent
        ]
    return dict(zip(keys, base))


METHOD_COLORS = _build_method_colors()

# Explain abbreviations once
ABBREV_EXPLAIN: Dict[str, str] = {
    #"H2D": "Host-to-device (H2D)",
    #"E2E": "End-to-end (E2E)",
    "H2D": "H2D",
    "E2E": "E2E",
}


def pretty_method(name: str) -> str:
    return METHOD_RENAMES.get(name, name)


def short_label(pretty: str) -> str:
    return SHORT_METHOD_LABELS.get(pretty, pretty)


def _save(fig: mpl.figure.Figure, basename: str) -> None:
    ensure_out_dir(FIG_OUT)
    fig.savefig(
        FIG_OUT / f"{basename}.pdf",
        bbox_inches="tight",
        metadata={"Title": basename, "Author": "Benchmark Plotter"},
    )
    plt.close(fig)


# ============================== PLOTTING UTILS ===================================

def _apply_sci(ax: mpl.axes.Axes, axis: str = "y") -> None:
    # Force scientific notation for large numbers
    ax.ticklabel_format(style="sci", axis=axis, scilimits=(0, 0), useMathText=True)


def _fmt_sci(v: float) -> str:
    # Scientific notation for annotations
    return f"{v:.2e}"


def _barh(
    ax: mpl.axes.Axes,
    labels: List[str],
    values: List[float],
    colors: Optional[List[Tuple[float, float, float]]] = None,
) -> None:
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    _apply_sci(ax, axis="x")
    xmax = max(values) if values else 1.0
    for yi, (v, b) in enumerate(zip(values, bars)):
        txt = _fmt_sci(v)
        # Place annotation inside the bar if it is wide enough, otherwise outside
        if v > 0.18 * xmax:
            ax.text(
                b.get_x() + b.get_width() * 0.98,
                yi,
                txt,
                va="center",
                ha="right",
                fontsize=8,
                color="white",
            )
        else:
            ax.text(
                b.get_x() + b.get_width() * 1.02,
                yi,
                txt,
                va="center",
                ha="left",
                fontsize=8,
            )


def _line(
    ax: mpl.axes.Axes,
    x: List[float],
    y: List[float],
    label: str,
    color: Optional[Tuple[float, float, float]],
) -> None:
    ax.plot(x, y, marker="o", linestyle="-", label=label, color=color)
    _apply_sci(ax, axis="y")


def _to_maybe_int(x: List[float]) -> List[float]:
    return [int(v) if float(v).is_integer() else v for v in x]


def _legend_outside(ax: mpl.axes.Axes, ncols: int = 1) -> None:
    ax.legend(
        frameon=False,
        ncols=ncols,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )


def _rotate_xticks(ax: mpl.axes.Axes, angle: int = 30) -> None:
    for label in ax.get_xticklabels():
        label.set_rotation(angle)
        label.set_ha("right")


# ============================== SUMMARY HELPERS ==================================

def summarize_trials_for_metric(
    trials: pd.DataFrame,
    config_name: str,
    metric: str,
) -> pd.DataFrame:
    """
    Compute method-level means from trials for a given (config_name, metric).
    This lets us draw bar/line figures even if the precomputed summary doesn't
    include an entry (or to ensure consistency with violin categories).
    """
    d = trials[(trials["config_name"] == config_name) & (trials["metric"] == metric)]
    if d.empty:
        return pd.DataFrame(columns=["method", "throughput_mean"])
    g = (
        d.groupby("method", as_index=False)["value"]
        .mean()
        .rename(columns={"value": "throughput_mean"})
    )
    return g


# ============================== FIGURE BUILDERS ==================================

def fig_encode_bar(summary: pd.DataFrame, trials: pd.DataFrame, config_name: str = "standard") -> None:
    df = summary[
        (summary["config_name"] == config_name)
        & (summary["metric"] == "encode_tok_s")
    ]
    # If absent in summary, compute from trials on the fly
    if df.empty:
        df = summarize_trials_for_metric(trials, config_name, "encode_tok_s")
        df["config_name"] = config_name
        df["metric"] = "encode_tok_s"
    df = df.copy()
    df["method_pretty"] = df["method"].map(pretty_method)
    df = df.sort_values("throughput_mean", ascending=True)
    colors = [METHOD_COLORS.get(m, "C0") for m in df["method_pretty"]]

    fig, ax = plt.subplots(figsize=(4.8, 3.0), constrained_layout=False)
    _barh(ax, df["method_pretty"].tolist(), df["throughput_mean"].tolist(), colors)
    ax.set_xlabel("Encode throughput (tokens/s)")
    _save(fig, "F1_encode_throughput_standard")


def fig_h2d_bar(summary: pd.DataFrame, trials: pd.DataFrame, config_name: str = "standard") -> None:
    df = summary[
        (summary["config_name"] == config_name)
        & (summary["metric"] == "h2d_tok_s")
    ]
    # If absent in summary, compute from trials
    if df.empty:
        df = summarize_trials_for_metric(trials, config_name, "h2d_tok_s")
        df["config_name"] = config_name
        df["metric"] = "h2d_tok_s"

    df = df.copy()
    df["method_pretty"] = df["method"].map(pretty_method)
    df = df.sort_values("throughput_mean", ascending=True)
    colors = [METHOD_COLORS.get(m, "C0") for m in df["method_pretty"]]

    fig, ax = plt.subplots(figsize=(4.8, 3.0), constrained_layout=False)
    _barh(ax, df["method_pretty"].tolist(), df["throughput_mean"].tolist(), colors)
    ax.set_xlabel(f"{ABBREV_EXPLAIN['H2D']} throughput (tokens/s)")
    _save(fig, "F2_h2d_throughput_standard")


def fig_streaming_speedup(summary: pd.DataFrame, trials: pd.DataFrame, config_name: str = "standard") -> None:
    df = summary[
        (summary["config_name"] == config_name)
        & (summary["metric"] == "end2end_tok_s")
    ]
    if df.empty:
        # build from trials if needed
        df = summarize_trials_for_metric(trials, config_name, "end2end_tok_s")
        df["config_name"] = config_name
        df["metric"] = "end2end_tok_s"

    pivot = (
        df.assign(method_pretty=df["method"].map(pretty_method))
        .set_index("method_pretty")["throughput_mean"]
    )

    needed = {"DNATok streaming (baseline)", "DNATok streaming (pipelined)"}
    if not needed.issubset(set(pivot.index)):
        return

    baseline = pivot["DNATok streaming (baseline)"]
    piped = pivot["DNATok streaming (pipelined)"]

    fig, ax = plt.subplots(figsize=(3.9, 2.9), constrained_layout=False)
    methods = ["DNATok streaming (baseline)", "DNATok streaming (pipelined)"]
    vals = [baseline, piped]
    cols = [METHOD_COLORS[m] for m in methods]
    x = np.arange(len(methods))
    bars = ax.bar(x, vals, color=cols)
    for xi, (v, b) in enumerate(zip(vals, bars)):
        ax.text(xi, v, _fmt_sci(v), ha="center", va="bottom", fontsize=8)

    ax.set_ylabel(f"{ABBREV_EXPLAIN['E2E']} throughput (tokens/s)")
    _apply_sci(ax, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [SHORT_METHOD_LABELS[m] for m in methods],
        rotation=30,
        ha="right",
    )
    _save(fig, "F3_streaming_e2e_speedup_standard")


def _extract_sweep(
    summary: pd.DataFrame,
    sweep_prefix: str,
    metric: str,
) -> pd.DataFrame:
    pat = re.compile(rf"{sweep_prefix}_(\d+)$")
    rows = []
    for _, r in summary.iterrows():
        m = pat.match(str(r["config_name"]))
        if m and r["metric"] == metric:
            size = int(m.group(1))
            pretty = pretty_method(r["method"])
            rows.append(
                {
                    "size": size,
                    "method_pretty": pretty,
                    "short": short_label(pretty),
                    "throughput_mean": r["throughput_mean"],
                }
            )
    return pd.DataFrame(rows)


def fig_batch_encode_sweep(summary: pd.DataFrame) -> None:
    df = _extract_sweep(summary, "batch_sweep", "encode_tok_s")
    if df.empty:
        return
    keep = {
        "Hugging Face tokenizer",
        "Hugging Face tokenizer (batch)",
        "DNATok encode (int64)",
        "DNATok staging (int64)",
        "DNATok staging (int32)",
    }
    df = df[df["method_pretty"].isin(keep)]
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=False)
    for name, g in df.groupby("method_pretty"):
        g = g.sort_values("size")
        _line(
            ax,
            _to_maybe_int(g["size"].tolist()),
            g["throughput_mean"].tolist(),
            short_label(name),
            METHOD_COLORS.get(name, None),
        )
    ax.set_xlabel("Batch size (B)")
    ax.set_ylabel("Encode throughput (tokens/s)")
    _legend_outside(ax, ncols=1)
    _rotate_xticks(ax, 30)
    _save(fig, "F4_batch_sweep_encode")


def fig_batch_e2e_sweep(summary: pd.DataFrame) -> None:
    df = _extract_sweep(summary, "batch_sweep", "end2end_tok_s")
    if df.empty:
        return
    keep = {
        "DNATok streaming (baseline)",
        "DNATok streaming (pipelined)",
    }
    df = df[df["method_pretty"].isin(keep)]
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=False)
    for name, g in df.groupby("method_pretty"):
        g = g.sort_values("size")
        _line(
            ax,
            _to_maybe_int(g["size"].tolist()),
            g["throughput_mean"].tolist(),
            short_label(name),
            METHOD_COLORS.get(name, None),
        )
    ax.set_xlabel("Batch size (B)")
    ax.set_ylabel(f"{ABBREV_EXPLAIN['E2E']} throughput (tokens/s)")
    ax.legend(frameon=False, loc="upper left")
    _rotate_xticks(ax, 30)
    _save(fig, "F5_batch_sweep_e2e")


def fig_length_encode_sweep(summary: pd.DataFrame) -> None:
    df = _extract_sweep(summary, "length_sweep", "encode_tok_s")
    if df.empty:
        return
    keep = {
        "Hugging Face tokenizer",
        "Hugging Face tokenizer (batch)",
        "DNATok encode (int64)",
        "DNATok staging (int64)",
        "DNATok staging (int32)",
    }
    df = df[df["method_pretty"].isin(keep)]
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=False)
    for name, g in df.groupby("method_pretty"):
        g = g.sort_values("size")
        _line(
            ax,
            _to_maybe_int(g["size"].tolist()),
            g["throughput_mean"].tolist(),
            short_label(name),
            METHOD_COLORS.get(name, None),
        )
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Encode throughput (tokens/s)")
    _legend_outside(ax, ncols=1)
    _rotate_xticks(ax, 30)
    _save(fig, "F6_length_sweep_encode")


def fig_length_e2e_sweep(summary: pd.DataFrame) -> None:
    df = _extract_sweep(summary, "length_sweep", "end2end_tok_s")
    if df.empty:
        return
    keep = {
        "DNATok streaming (baseline)",
        "DNATok streaming (pipelined)",
    }
    df = df[df["method_pretty"].isin(keep)]
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=False)
    for name, g in df.groupby("method_pretty"):
        g = g.sort_values("size")
        _line(
            ax,
            _to_maybe_int(g["size"].tolist()),
            g["throughput_mean"].tolist(),
            short_label(name),
            METHOD_COLORS.get(name, None),
        )
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel(f"{ABBREV_EXPLAIN['E2E']} throughput (tokens/s)")
    ax.legend(frameon=False, loc="upper left")
    _rotate_xticks(ax, 30)
    _save(fig, "F7_length_sweep_e2e")


def fig_ablation(summary: pd.DataFrame) -> None:
    is_ablation = summary["config_name"].astype(str).str.startswith("ablation_")
    df = summary[is_ablation & (summary["metric"] == "end2end_tok_s")].copy()
    if df.empty:
        return

    pretty_cfg = {
        "ablation_baseline": "Baseline",
        "ablation_int32_only": "Int32 H2D only",
        "ablation_overlap_only": "Overlap only",
        "ablation_both": "Both (int32 + overlap)",
    }
    df["label"] = df["config_name"].map(pretty_cfg).fillna(df["config_name"])
    df = df.groupby("label", as_index=False)["throughput_mean"].mean()
    df = df.sort_values("throughput_mean", ascending=True)

    # Muted sequential palette for ablation
    if _USE_SEABORN:
        ab_colors = sns.color_palette("muted", len(df))
    else:
        ab_colors = [(0.40, 0.40, 0.40)] * len(df)

    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=False)
    _barh(ax, df["label"].tolist(), df["throughput_mean"].tolist(), ab_colors)
    ax.set_xlabel(f"{ABBREV_EXPLAIN['E2E']} throughput (tokens/s)")
    _save(fig, "F8_ablation_e2e")


def fig_trials_violin(
    trials: pd.DataFrame,
    summary: pd.DataFrame,
    config_name: str = "standard",
) -> None:
    """
    Simple Matplotlib violin plot:
      - Uses ax.violinplot (no seaborn, no overlays)
      - Shows medians (built-in)
      - Keeps your tick rotation and scientific notation
      - Saves to F9_trials_violin_encode_standard.pdf
    """
    # Restrict to the requested config/metric
    d = trials[
        (trials["config_name"] == config_name)
        & (trials["metric"] == "encode_tok_s")
    ].copy()
    if d.empty:
        return

    # Keep the same method naming as the rest of the figures
    d["method_pretty"] = d["method"].map(pretty_method)

    # If the summary lists a subset, stick to that subset to stay consistent
    have = summary[
        (summary["config_name"] == config_name)
        & (summary["metric"] == "encode_tok_s")
    ][["method"]]
    if not have.empty:
        d = d[d["method"].isin(have["method"].unique())]
        if d.empty:
            return

    # Simple ordering: alphabetical by pretty name
    x_order = sorted(d["method_pretty"].unique())

    # Prepare data series per method for violinplot
    data_series = [d.loc[d["method_pretty"] == m, "value"].to_numpy() for m in x_order]

    # Plot
    fig, ax = plt.subplots(figsize=(5.0, 3.2), constrained_layout=False)
    parts = ax.violinplot(
        data_series,
        showmeans=False,
        showmedians=True,   # draw median line
        showextrema=False,  # keep it simple
    )

    # Keep Matplotlib defaults; no color overrides for a plain look
    # (If you want method colors, set each body facecolor from METHOD_COLORS here.)

    # X labels
    ax.set_xticks(np.arange(1, len(x_order) + 1))
    ax.set_xticklabels([short_label(m) for m in x_order])
    _rotate_xticks(ax, 30)

    # Y axis and styling
    ax.set_ylabel("Encode throughput (tokens/s)")
    _apply_sci(ax, axis="y")
    ax.grid(True, axis="y")

    _save(fig, "F9_trials_violin_encode_standard")

# ============================== MAIN =============================================

def main() -> None:
    configure_matplotlib_style()
    summary_csv, trials_csv = discover_inputs(RESULTS_DIR)

    print("Using files:")
    print(f"  Summary CSV: {summary_csv}")
    print(f"  Trials  CSV: {trials_csv}")
    ensure_out_dir(FIG_OUT)

    summary = pd.read_csv(summary_csv)
    trials = pd.read_csv(trials_csv)

    # Figures that can draw from summary; fall back to re-summarized trials if needed
    fig_encode_bar(summary, trials, config_name="standard")
    fig_h2d_bar(summary, trials, config_name="standard")
    fig_streaming_speedup(summary, trials, config_name="standard")

    fig_batch_encode_sweep(summary)
    fig_batch_e2e_sweep(summary)
    fig_length_encode_sweep(summary)
    fig_length_e2e_sweep(summary)

    fig_ablation(summary)

    # Violin (per-trial distribution) instead of scatter
    fig_trials_violin(trials, summary, config_name="standard")

    print(f"\nPDF figures written to: {FIG_OUT.resolve()}")


if __name__ == "__main__":
    main()
