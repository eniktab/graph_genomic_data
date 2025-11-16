#!/usr/bin/env python3
"""
Stride sweep — separate Nature-style panels (A–F)

Outputs (saved next to the CSV, each as PNG + PDF):
  panel_A_accuracy_vs_stride
  panel_B_time_minutes_vs_stride
  panel_C_per_read_ms_vs_stride
  panel_D_memory_size_gb_vs_stride
  panel_E_pct_perfect_random_vs_stride
  panel_F_alignment_composition_vs_stride

What it shows:
  A) Accuracy (bin/refined; @1 and @1+adj) vs stride
  B) Build/Load and Total Query time (minutes) vs stride
  C) Per-read query time (ms) vs stride
  D) Memory/Size series in GB vs stride (index size, GPU/host peaks, etc.)
  E) % perfect and % random reads vs stride
  F) Alignment composition (ok / adjacent / wrong) in % vs stride (stacked bars)
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe on headless servers
import matplotlib.pyplot as plt

# ==========================
# Set your CSV path here
# ==========================
CSV_PATH = "/home/niktabel/workspace/media/gadi_g_te53/en9803/sandpit/graph_genomics/chr22/hel_cache_sweep_best/sweep_results.csv"  # <- change this if needed


# ---------- Styling (Nature-ish minimalist) ----------
def apply_nature_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,     # editable text in Illustrator
        "ps.fonttype": 42,
        "font.size": 9.5,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
        "axes.grid": False,     # Nature figures rarely use grids
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    })


# ---------- Helpers ----------
def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _safe_div(numer, denom):
    numer = np.asarray(numer, dtype=float)
    denom = np.asarray(denom, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, numer / denom, np.nan)


def _compute_missing_accuracies_and_extras(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # per-read query time (ms) if possible
    if "query_time_ms_per_read" not in df.columns:
        if "query_time_sec_total" in df.columns and "n_reads" in df.columns:
            df["query_time_ms_per_read"] = 1000.0 * _safe_div(df["query_time_sec_total"], df["n_reads"])

    # bin-level accuracies
    if {"n_reads", "bin_correct", "bin_adjacent"}.issubset(df.columns):
        if "acc_bin_overall@1" not in df.columns:
            df["acc_bin_overall@1"] = _safe_div(df["bin_correct"], df["n_reads"])
        if "acc_bin_overall@1+adj" not in df.columns:
            df["acc_bin_overall@1+adj"] = _safe_div(df["bin_correct"] + df["bin_adjacent"], df["n_reads"])

    # refined-level accuracies
    if {"n_reads", "refined_correct", "refined_adjacent"}.issubset(df.columns):
        if "acc_refined_overall@1" not in df.columns:
            df["acc_refined_overall@1"] = _safe_div(df["refined_correct"], df["n_reads"])
        if "acc_refined_overall@1+adj" not in df.columns:
            df["acc_refined_overall@1+adj"] = _safe_div(
                df["refined_correct"] + df["refined_adjacent"], df["n_reads"]
            )

    # % perfect / % random
    if {"n_reads", "n_perfect"}.issubset(df.columns):
        df["pct_perfect"] = 100.0 * _safe_div(df["n_perfect"], df["n_reads"])
    if {"n_reads", "n_random"}.issubset(df.columns):
        df["pct_random"] = 100.0 * _safe_div(df["n_random"], df["n_reads"])

    return df


def _add_unit_conversions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # sec → min
    for c in ["index_build_or_load_time_sec", "query_time_sec_total"]:
        if c in df.columns:
            df[c.replace("_sec", "_min")] = df[c] / 60.0

    # bytes → GB
    for c in list(df.columns):
        if c.endswith("_bytes"):
            df[c[:-6] + "_gb"] = pd.to_numeric(df[c], errors="coerce") / 1e9

    # MB → GB fallback for common fields
    if "index_size_gb" not in df.columns and "index_size_mb" in df.columns:
        df["index_size_gb"] = pd.to_numeric(df["index_size_mb"], errors="coerce") / 1024.0

    return df


def _aggregate_by_stride_sum_counts_mean_rest(df: pd.DataFrame) -> pd.DataFrame:
    if "stride" not in df.columns:
        raise ValueError("CSV is missing required 'stride' column.")

    df = df.copy()
    df["stride"] = pd.to_numeric(df["stride"], errors="coerce")
    df = df.dropna(subset=["stride"])
    df["stride"] = df["stride"].astype(int)

    # SUM these count-like columns
    count_candidates = [
        "n_reads", "n_perfect", "n_random",
        "bin_correct", "bin_adjacent", "bin_wrong",
        "refined_correct", "refined_adjacent", "refined_wrong",
        "rescued_by_alignment", "anchor_ok_align_poor", "similar_region_wrong_bin",
    ]
    sum_cols = [c for c in count_candidates if c in df.columns]

    # MEAN the rest (numeric) excluding 'stride' and sum_cols
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    mean_cols = [c for c in numeric_cols if c not in sum_cols and c != "stride"]

    agg_dict: Dict[str, str] = {c: "sum" for c in sum_cols}
    agg_dict.update({c: "mean" for c in mean_cols})

    agg = (
        df.groupby("stride", as_index=False, dropna=False)
          .agg(agg_dict)
          .sort_values("stride")
          .reset_index(drop=True)
    )

    # Recompute % perfect / % random from summed counts
    if {"n_reads", "n_perfect"}.issubset(agg.columns):
        agg["pct_perfect"] = 100.0 * _safe_div(agg["n_perfect"], agg["n_reads"])
    if {"n_reads", "n_random"}.issubset(agg.columns):
        agg["pct_random"] = 100.0 * _safe_div(agg["n_random"], agg["n_reads"])

    # Alignment composition (ok/adj/wrong) — prefer refined_*, else bin_*
    if {"refined_correct", "refined_adjacent"}.issubset(agg.columns):
        ok = agg["refined_correct"]
        adj = agg["refined_adjacent"]
        wrong = agg["refined_wrong"] if "refined_wrong" in agg.columns else agg["n_reads"] - (ok + adj)
    elif {"bin_correct", "bin_adjacent"}.issubset(agg.columns):
        ok = agg["bin_correct"]
        adj = agg["bin_adjacent"]
        wrong = agg["bin_wrong"] if "bin_wrong" in agg.columns else agg["n_reads"] - (ok + adj)
    else:
        ok = adj = wrong = pd.Series(np.nan, index=agg.index)

    agg["align_ok_pct"] = 100.0 * _safe_div(ok, agg.get("n_reads", np.nan))
    agg["align_adj_pct"] = 100.0 * _safe_div(adj, agg.get("n_reads", np.nan))
    agg["align_wrong_pct"] = 100.0 * _safe_div(wrong, agg.get("n_reads", np.nan))

    # Best stride marker (acc@1 primary, tie acc@1+adj)
    a1 = agg.get("acc_bin_overall@1", np.full(len(agg), np.nan))
    a1p = agg.get("acc_bin_overall@1+adj", np.full(len(agg), np.nan))
    if np.isfinite(a1).any():
        order = np.lexsort((np.nan_to_num(a1p, nan=-1.0), np.nan_to_num(a1, nan=-1.0)))
        best_idx = int(order[-1])
        agg["_is_best"] = False
        agg.loc[best_idx, "_is_best"] = True
    else:
        agg["_is_best"] = False

    return agg


def _panel_axes(title: str, xlabel: str, ylabel: str, figsize=(3.2, 2.4)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_title(title, pad=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def _panel_letter(ax, letter: str):
    ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def _save_panel(fig, outbase: Path):
    fig.savefig(outbase.with_suffix(".png"))
    fig.savefig(outbase.with_suffix(".pdf"))
    plt.close(fig)


# ---------- Panels ----------
def panel_A(agg: pd.DataFrame, outdir: Path):
    series = []
    if "acc_bin_overall@1" in agg:         series.append(("acc_bin_overall@1", "Binning acc@1"))
    if "acc_bin_overall@1+adj" in agg:     series.append(("acc_bin_overall@1+adj", "Binning acc@1+adj"))
    if "acc_refined_overall@1" in agg:     series.append(("acc_refined_overall@1", "Refined acc@1"))
    if "acc_refined_overall@1+adj" in agg: series.append(("acc_refined_overall@1+adj", "Refined acc@1+adj"))
    if not series:
        return False

    fig, ax = _panel_axes("Accuracy vs Stride", "Stride (bp)", "Accuracy (%)")
    for c, label in series:
        ax.plot(agg["stride"], 100.0 * agg[c], marker="o", label=label)
    if "_is_best" in agg.columns and agg["_is_best"].any():
        best = agg.loc[agg["_is_best"]].iloc[0]
        ax.axvline(best["stride"], linestyle=":", lw=1.0)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncols=2)
    _panel_letter(ax, "A")
    _save_panel(fig, outdir / "panel_A_accuracy_vs_stride")
    return True


def panel_B(agg: pd.DataFrame, outdir: Path):
    time_present = False
    fig, ax = _panel_axes("Index/Query Time vs Stride", "Stride (bp)", "Time (min)")
    if "index_build_or_load_time_min" in agg and agg["index_build_or_load_time_min"].notna().any():
        ax.plot(agg["stride"], agg["index_build_or_load_time_min"], marker="o", label="Index build/load")
        time_present = True
    if "query_time_min_total" in agg and agg["query_time_min_total"].notna().any():
        ax.plot(agg["stride"], agg["query_time_min_total"], marker="o", label="Query total")
        time_present = True
    if not time_present:
        plt.close(fig)
        return False
    ax.legend(frameon=False)
    _panel_letter(ax, "B")
    _save_panel(fig, outdir / "panel_B_time_minutes_vs_stride")
    return True


def panel_C(agg: pd.DataFrame, outdir: Path):
    if "query_time_ms_per_read" not in agg or not agg["query_time_ms_per_read"].notna().any():
        return False
    fig, ax = _panel_axes("Per-Read Query Time", "Stride (bp)", "Per-read (ms)")
    ax.plot(agg["stride"], agg["query_time_ms_per_read"], marker="o", label="Per-read (ms)")
    ax.legend(frameon=False)
    _panel_letter(ax, "C")
    _save_panel(fig, outdir / "panel_C_per_read_ms_vs_stride")
    return True


def panel_D(agg: pd.DataFrame, outdir: Path):
    gb_cols_pref = ["index_size_gb", "gpu_mem_peak_gb", "host_mem_peak_gb", "dataset_gb", "reads_gb", "ref_gb"]
    gb_cols_all = [c for c in agg.columns if c.endswith("_gb")]
    series = [c for c in gb_cols_pref if c in gb_cols_all] + [c for c in gb_cols_all if c not in gb_cols_pref]
    series = [c for c in series if agg[c].notna().any()]
    if not series:
        return False
    fig, ax = _panel_axes("Memory / Size vs Stride", "Stride (bp)", "GB")
    for c in series:
        ax.plot(agg["stride"], agg[c], marker="o", label=c)
    ax.legend(frameon=False, ncols=2)
    _panel_letter(ax, "D")
    _save_panel(fig, outdir / "panel_D_memory_size_gb_vs_stride")
    return True


def panel_E(agg: pd.DataFrame, outdir: Path):
    present = False
    fig, ax = _panel_axes("% Perfect / % Random vs Stride", "Stride (bp)", "Percent (%)")
    if "pct_perfect" in agg and agg["pct_perfect"].notna().any():
        ax.plot(agg["stride"], agg["pct_perfect"], marker="o", label="% perfect")
        present = True
    if "pct_random" in agg and agg["pct_random"].notna().any():
        ax.plot(agg["stride"], agg["pct_random"], marker="o", label="% random")
        present = True
    if not present:
        plt.close(fig)
        return False
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    _panel_letter(ax, "E")
    _save_panel(fig, outdir / "panel_E_pct_perfect_random_vs_stride")
    return True


def panel_F(agg: pd.DataFrame, outdir: Path):
    needed = {"align_ok_pct", "align_adj_pct", "align_wrong_pct"}
    if not needed.issubset(agg.columns):
        return False
    fig = plt.figure(figsize=(3.4, 2.6))
    ax = fig.add_subplot(111)
    x = np.arange(len(agg)); w = 0.78
    ok = agg["align_ok_pct"].to_numpy()
    adj = agg["align_adj_pct"].to_numpy()
    wrong = agg["align_wrong_pct"].to_numpy()
    ax.bar(x, ok, width=w, label="ok")
    ax.bar(x, adj, width=w, bottom=ok, label="adjacent")
    ax.bar(x, wrong, width=w, bottom=ok + adj, label="wrong")
    ax.set_xticks(x, agg["stride"].astype(int).astype(str))
    ax.set_xlabel("Stride (bp)")
    ax.set_ylabel("Percent (%)")
    ax.set_title("Alignment composition vs Stride")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncols=3)
    _panel_letter(ax, "F")
    _save_panel(fig, outdir / "panel_F_alignment_composition_vs_stride")
    return True


# ---------- Main ----------
def main(csv_path: str):
    apply_nature_style()
    outdir = Path(csv_path).parent
    outdir.mkdir(parents=True, exist_ok=True)

    # load + clean
    df = pd.read_csv(csv_path)
    likely_numeric = [
        "stride",
        "n_reads", "n_perfect", "n_random",
        "bin_correct", "bin_adjacent", "bin_wrong",
        "refined_correct", "refined_adjacent", "refined_wrong",
        "rescued_by_alignment", "anchor_ok_align_poor", "similar_region_wrong_bin",
        "query_time_sec_total", "query_time_ms_per_read",
        "index_build_or_load_time_sec",
        "index_rows",
        "index_size_bytes", "index_size_mb",
        "gpu_mem_peak_bytes", "host_mem_peak_bytes",
        "dataset_bytes", "reads_bytes", "ref_bytes", "output_bytes",
        "acc_bin_overall@1", "acc_bin_overall@1+adj",
        "acc_refined_overall@1", "acc_refined_overall@1+adj",
    ]
    df = _ensure_numeric(df, likely_numeric)
    df = _compute_missing_accuracies_and_extras(df)
    df = _add_unit_conversions(df)

    agg = _aggregate_by_stride_sum_counts_mean_rest(df)
    # Save aggregated CSV for reference (helper cols removed)
    agg.drop(columns=[c for c in agg.columns if c.startswith("_")], errors="ignore") \
       .to_csv(outdir / "sweep_results_aggregated_by_stride.csv", index=False)

    # Render panels separately
    made = {
        "A": panel_A(agg, outdir),
        "B": panel_B(agg, outdir),
        "C": panel_C(agg, outdir),
        "D": panel_D(agg, outdir),
        "E": panel_E(agg, outdir),
        "F": panel_F(agg, outdir),
    }
    for k, v in made.items():
        print(f"Panel {k}: {'created' if v else 'skipped (insufficient data)'}")
    print(f"Saved panels and aggregated CSV in: {outdir}")


if __name__ == "__main__":
    main(CSV_PATH)
