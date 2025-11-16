"""
Matplotlib & seaborn styling for Nature-style figures, plus per-figure
layout engine helpers.

Usage:
    from src.util.plotting_style import setup_nature_style, new_figure, new_subplots, set_layout
    setup_nature_style(default_layout="constrained")
    fig, ax = new_subplots(1, 2)               # uses the module default
    fig2 = new_figure(layout="none")           # manual spacing later
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns

# Module-level default; set by setup_nature_style()
_DEFAULT_LAYOUT_ENGINE = "constrained"            # "constrained" | "tight" | "none"
_SUPPORTED_ENGINES = {"constrained", "tight", "none"}

def _apply_layout_engine(fig: plt.Figure, engine: str | None):
    eng = (engine or "none").lower()
    if eng not in _SUPPORTED_ENGINES:
        raise ValueError(f"Unknown layout engine: {engine!r}")

    # Matplotlib ≥ 3.8
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine(None if eng == "none" else eng)
        return
        return

    # Fallback for older Matplotlib
    # First make sure nothing is globally on
    for attr, val in (("set_constrained_layout", False), ("set_tight_layout", False)):
        try:
            getattr(fig, attr)(val)
        except Exception:
            pass
    if eng == "constrained":
        try:
            fig.set_constrained_layout(True)
        except Exception:
            pass
    elif eng == "tight":
        try:
            fig.set_tight_layout(True)
        except Exception:
            pass
    # "none": nothing more to do

def new_figure(*args, layout: str | None = None, **kwargs) -> plt.Figure:
    """
    Create a figure and apply a layout engine.
    If layout is None, use the module default set by setup_nature_style().
    """
    chosen = _DEFAULT_LAYOUT_ENGINE if (layout is None) else layout
    try:
        fig = plt.figure(*args, layout=None if (chosen or "none") == "none" else chosen, **kwargs)
    except TypeError:  # older Matplotlib (no 'layout' kw)
        fig = plt.figure(*args, **kwargs)
    _apply_layout_engine(fig, chosen)
    return fig

def new_subplots(*args, layout: str | None = None, **kwargs):
    """
    Create subplots and apply a layout engine.
    If layout is None, use the module default set by setup_nature_style().
    """
    chosen = _DEFAULT_LAYOUT_ENGINE if (layout is None) else layout
    try:
        fig, axes = plt.subplots(*args, layout=None if (chosen or "none") == "none" else chosen, **kwargs)
    except TypeError:
        fig, axes = plt.subplots(*args, **kwargs)
    _apply_layout_engine(fig, chosen)
    return fig, axes

def set_layout(fig: plt.Figure, layout: str | None):
    """Set/override the layout engine on an existing figure."""
    _apply_layout_engine(fig, layout)

def setup_nature_style(default_layout: str = "constrained"):
    """
    Configure matplotlib + seaborn for a Nature-style look.
    Keeps global layout managers off; use per-figure helpers above.
    """
    global _DEFAULT_LAYOUT_ENGINE
    dl = (default_layout or "none").lower()
    if dl not in _SUPPORTED_ENGINES:
        raise ValueError(f"default_layout must be one of {_SUPPORTED_ENGINES}, got {default_layout!r}")
    _DEFAULT_LAYOUT_ENGINE = dl

    # Clean seaborn/axes style
    sns.set_style("white")
    sns.despine()

    plt.rcParams.update({
        # Fonts (Nature-like small sizes)
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 12,

        # Lines/ticks
        "axes.linewidth": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,

        # Figure/export
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.format": "pdf",
        "savefig.pad_inches": 0.1,
        "pdf.fonttype": 42,  # editable text in Illustrator
        "ps.fonttype": 42,

        # Colors
        "image.cmap": "icefire",  # house default; override per-plot if needed
        "axes.prop_cycle": plt.cycler('color', [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]),

        # Axes appearance
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,

        # Text/math
        "text.usetex": False,
        "mathtext.default": "regular",

        # IMPORTANT: don’t force global layout engines
        "figure.constrained_layout.use": False,
        "figure.autolayout": False,
    })

# Optional color tokens
NATURE_COLORS = {
    'primary':   '#1f77b4',
    'secondary': '#ff7f0e',
    'accent':    '#2ca02c',
    'warning':   '#d62728',
    'qualitative': [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
}

__all__ = [
    "setup_nature_style",
    "new_figure",
    "new_subplots",
    "set_layout",
    "NATURE_COLORS",
]