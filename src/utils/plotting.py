import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Style-aware helper: compute pixel keep-out radius
# from scatter size `s` (points^2) + a style-based buffer.
# -------------------------------------------------
def keepout_from_marker(fig, marker_size, buffer_px=None, font_size=None):
    """
    Parameters
    ----------
    fig : matplotlib.figure.Figure
    marker_size : float
        The scatter `s` value (in points^2) used when calling plt.scatter(..., s=marker_size).
    buffer_px : float or None
        Extra pixels to keep beyond the marker radius. If None, derived from `font_size` or rcParams['font.size'].
    font_size : float or None
        Font size in points to use for deriving the buffer. If None, uses rcParams['font.size'].

    Returns
    -------
    float
        Recommended keep-out distance in **pixels** from the dot center.
    """
    base_fs = float(plt.rcParams.get("font.size", 7.0)) if font_size is None else float(font_size)
    if buffer_px is None:
        # A readable default gap: ~0.75em, but never < 4 px
        buffer_px = max(4.0, 0.75 * base_fs)

    # Convert scatter size (pt^2) to a radius in points, then to pixels
    radius_pt = np.sqrt(marker_size) / 2.0
    radius_px = radius_pt * (fig.dpi / 72.0)
    return radius_px + buffer_px


# -------------------------------------------------
# Smart labeler: avoids overlapping labels & points
# Style-aware: uses rcParams set by your plotting_style module.
# -------------------------------------------------
def annotate_smart(
    ax,
    xs,
    ys,
    labels,
    fontsize=None,
    max_iter=800,
    k_attract=0.02,
    repulse=0.8,
    point_repulse=5,
    pad_pts=None,
    pad_axes=None,
    min_anchor_dist_px=None,
    marker_size=None,
    connector_kw=None,
):
    """
    Place non-overlapping labels near points without ever covering the dot.

    Style-aware defaults (pulled from rcParams set by your style):
    - `fontsize` : defaults to rcParams['axes.labelsize'] (falls back to rcParams['font.size']).
    - `pad_pts`  : ~0.6em of rcParams['font.size'] (>= 2 px).
    - `pad_axes` : ~0.4em of rcParams['font.size'] (>= 2 px).
    - `min_anchor_dist_px` :
         * If None and `marker_size` is provided, uses keepout_from_marker(...).
         * Else ~1.6em of rcParams['font.size'] (>= 12 px).
    - Connector width defaults to rcParams['axes.linewidth'] (slightly scaled).

    Parameters are otherwise unchanged.
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    labels = list(labels)

    # --- Style-driven defaults from rcParams ---
    base_fs = float(plt.rcParams.get("font.size", 7.0))
    label_fs_default = float(plt.rcParams.get("axes.labelsize", base_fs))
    if fontsize is None:
        fontsize = label_fs_default
    if pad_pts is None:
        pad_pts = max(2.0, 0.6 * base_fs)    # ~0.6em padding around label bboxes
    if pad_axes is None:
        pad_axes = max(2.0, 0.4 * base_fs)   # ~0.4em padding to frame

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    to_disp = ax.transData.transform
    to_data = ax.transData.inverted().transform

    pts_disp = to_disp(np.c_[xs, ys])

    # Default distance of label from marker
    if min_anchor_dist_px is None:
        if marker_size is not None:
            min_anchor_dist_px = keepout_from_marker(fig, marker_size, font_size=fontsize)
        else:
            # A reasonable generic default: ~1.6em (but never less than 12 px)
            min_anchor_dist_px = max(12.0, 1.6 * base_fs)

    # Initial label positions around each point (cycle through cardinal/diagonals)
    base_dirs = np.deg2rad([90, 270, 180, 0, 45, 225, 135, 315])
    xlab, ylab = [], []
    for i, (px, py) in enumerate(pts_disp):
        ang = base_dirs[i % len(base_dirs)]
        dx = min_anchor_dist_px * np.cos(ang)
        dy = min_anchor_dist_px * np.sin(ang)
        lx, ly = px + dx, py + dy
        dlx, dly = to_data([[lx, ly]])[0]
        xlab.append(dlx)
        ylab.append(dly)
    xlab = np.array(xlab)
    ylab = np.array(ylab)

    # Create text artists with style-aware size
    texts = [
        ax.text(lx, ly, lab, fontsize=fontsize, ha='center', va='center')
        for lx, ly, lab in zip(xlab, ylab, labels)
    ]

    def bboxes_in_pixels(texts_):
        bbs = []
        for t in texts_:
            bb = t.get_window_extent(renderer=renderer)
            # Expand bbox by pad_pts pixels on each side
            bb = bb.expanded((bb.width + 2 * pad_pts) / bb.width,
                             (bb.height + 2 * pad_pts) / bb.height)
            bbs.append(bb)
        return bbs

    ax_bb_disp = ax.get_window_extent(renderer=renderer)

    for it in range(max_iter):
        labs_disp = to_disp(np.c_[xlab, ylab])

        # Sync positions before measuring bboxes
        for t, (dx, dy) in zip(texts, labs_disp):
            t.set_position(to_data([[dx, dy]])[0])
        fig.canvas.draw()
        bbs = bboxes_in_pixels(texts)

        forces = np.zeros_like(labs_disp)

        # Label–label overlap repulsion
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if bbs[i].overlaps(bbs[j]):
                    dx = (bbs[i].x0 + bbs[i].x1)/2 - (bbs[j].x0 + bbs[j].x1)/2
                    dy = (bbs[i].y0 + bbs[i].y1)/2 - (bbs[j].y0 + bbs[j].y1)/2
                    if dx == 0 and dy == 0:
                        dx, dy = 1.0, 0.0
                    dist = max(np.hypot(dx, dy), 1e-6)
                    ox = min(bbs[i].x1, bbs[j].x1) - max(bbs[i].x0, bbs[j].x0)
                    oy = min(bbs[i].y1, bbs[j].y1) - max(bbs[i].y0, bbs[j].y0)
                    overlap_mag = max(ox, 0.0) * max(oy, 0.0)
                    push = repulse * (overlap_mag ** 0.5)
                    fx = push * dx / dist
                    fy = push * dy / dist
                    forces[i] += [fx, fy]
                    forces[j] += [-fx, -fy]

        # Keep-out circle around each point + attraction back toward the anchor
        for i in range(len(texts)):
            vec = pts_disp[i] - labs_disp[i]
            dist = np.hypot(*vec)
            if dist < min_anchor_dist_px:
                if dist == 0:
                    vec = np.array([1.0, 0.0]); dist = 1.0
                forces[i] -= (vec / dist) * (min_anchor_dist_px - dist + 1.0)
            else:
                forces[i] += k_attract * vec

        # Repel labels from all points (helps in dense regions)
        if point_repulse > 0:
            for i in range(len(texts)):
                dxy = labs_disp[i] - pts_disp
                dist = np.linalg.norm(dxy, axis=1, keepdims=True).clip(min=1.0)
                rep = point_repulse * (dxy / dist**2).sum(axis=0)
                forces[i] += rep

        # Keep labels within axes bounds
        for i, bb in enumerate(bbs):
            fx = fy = 0.0
            if bb.x0 < ax_bb_disp.x0 + pad_axes: fx += (ax_bb_disp.x0 + pad_axes - bb.x0)
            if bb.x1 > ax_bb_disp.x1 - pad_axes: fx -= (bb.x1 - (ax_bb_disp.x1 - pad_axes))
            if bb.y0 < ax_bb_disp.y0 + pad_axes: fy += (ax_bb_disp.y0 + pad_axes - bb.y0)
            if bb.y1 > ax_bb_disp.y1 - pad_axes: fy -= (bb.y1 - (ax_bb_disp.y1 - pad_axes))
            forces[i] += [fx, fy]

        # Integrate one step
        step = 0.6 if it < 0.5 * max_iter else 0.25
        new_disp = labs_disp + step * forces
        moved = np.mean(np.linalg.norm(new_disp - labs_disp, axis=1))
        new_data = to_data(new_disp)
        xlab, ylab = new_data[:, 0], new_data[:, 1]
        if moved < 0.05:  # convergence threshold in pixels
            break

    # Final draw with connectors
    if connector_kw is None:
        # Use style line width and colors
        axes_lw = float(plt.rcParams.get("axes.linewidth", 0.8))
        edge_color = plt.rcParams.get("axes.edgecolor", "black")
        connector_kw = dict(arrowstyle='-',
                            lw=max(0.5, 0.9 * axes_lw),
                            alpha=0.6,
                            color=edge_color)

    for t, (lx, ly) in zip(texts, zip(xlab, ylab)):
        t.set_position((lx, ly))
        t.set_ha('center')
        t.set_va('center')
        t.set_fontsize(fontsize)

    for (x, y), (lx, ly) in zip(np.c_[xs, ys], np.c_[xlab, ylab]):
        ax.annotate("", xy=(lx, ly), xytext=(x, y), arrowprops=connector_kw)

    ax.autoscale(enable=False)
    ax.margins(x=0.05, y=0.05)
