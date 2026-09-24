"""Shared figure style for the manuscript: fixed categorical palette, two-column ICML sizes, and save/skip helpers."""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#1a1a1a"          # near-black for all text and axis furniture
MUTED = "#8a8a8a"
FULL_WIDTH = 6.75        # inches, ICML two-column page
COL_WIDTH = 3.25         # inches, one ICML column
LW = 1.8

def color(i):
    """Palette colour i, cycling."""
    return PALETTE[i % len(PALETTE)]

def use_style():
    """Apply the manuscript rcParams. Call once at the top of every figure script."""
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "lines.linewidth": LW, "lines.markersize": 3.5, "lines.solid_capstyle": "round",
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False, "legend.handlelength": 1.6,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "figure.facecolor": "white", "axes.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    })

def figures_dir(root=None):
    """paper/figures, created if needed."""
    root = root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    d = os.path.join(root, "paper", "figures"); os.makedirs(d, exist_ok=True); return d

def save(fig, name, root=None):
    """Place the registered panel letters, then write paper/figures/<name>.pdf and .png and report the paths."""
    for problem in place_panel_letters(fig):
        print(f"  [layout] {name}: {problem}")
    d = figures_dir(root); out = []
    for ext in ("pdf", "png"):
        p = os.path.join(d, f"{name}.{ext}"); fig.savefig(p); out.append(p)
    plt.close(fig)
    print("wrote " + " and ".join(os.path.basename(p) for p in out) + f" in {d}")
    return out

def skip(panel, reason):
    """Report a panel that cannot be drawn from the data present."""
    print(f"  [skip] {panel}: {reason}")

def empty(ax, msg):
    """Mark an axis whose data is missing and keep the layout intact."""
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", fontsize=6.5, color=MUTED)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

def label_line(ax, x, y, text, col, dx=0, dy=0, **kw):
    """Direct label at the end of a line, in the line's colour."""
    ax.annotate(text, xy=(x, y), xytext=(x + dx, y + dy), color=col, fontsize=6.8,
                va=kw.pop("va", "center"), ha=kw.pop("ha", "left"), annotation_clip=False, **kw)

LETTER_SIZE, HEADING_SIZE = 8.5, 7.5

def panel_tag(ax, letter):
    """Register a bold panel letter; save() places it above and left of everything the panel draws."""
    ax._cmc_letter, ax._cmc_heading = letter, None

def band(ax, x, mean, sd, col, alpha=0.18):
    """Shaded mean +- sd band under a line."""
    ax.fill_between(x, mean - sd, mean + sd, color=col, alpha=alpha, linewidth=0)

def extend_right(ax, frac=0.16):
    """Leave room at the right edge for direct labels, in log space on a log axis."""
    lo, hi = ax.get_xlim()
    if ax.get_xscale() == "log":
        import math
        llo, lhi = math.log10(lo), math.log10(hi); ax.set_xlim(lo, 10 ** (llo + (lhi - llo) * (1 + frac)))
    else:
        ax.set_xlim(lo, lo + (hi - lo) * (1 + frac))

def rule_legend(fig, entries, ncol=None, y=-0.02):
    """The figure's single legend, centred under the axes. entries: [(label, kwargs for Line2D)]."""
    handles = [plt.Line2D([], [], label=lab, **kw) for lab, kw in entries]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol or len(entries),
               frameon=False, borderaxespad=0)

def label_ends(ax, items, min_sep=0.06, x=1.01):
    """Direct labels at the right edge, pushed apart so that close-lying lines stay readable.
    items: [(y_data, text, colour)]; x is an axes fraction (use extend_right first to label inside the axis).
    Call after the data and the y limits are final."""
    lo, hi = ax.get_ylim(); span = (hi - lo) or 1.0
    rows = sorted(((y - lo) / span, t, c) for y, t, c in items if y == y)
    for i in range(1, len(rows)):
        if rows[i][0] - rows[i - 1][0] < min_sep:
            rows[i] = (rows[i - 1][0] + min_sep, rows[i][1], rows[i][2])
    for f, text, col in rows:
        ax.annotate(text, xy=(x, min(max(f, 0.0), 1.0)), xycoords="axes fraction",
                    ha="left", va="center", fontsize=6.8, color=col, annotation_clip=False)

def panel_title(ax, letter, text, pad=3):
    """Register a panel letter with a heading; save() sets both on one line above the panel, the letter leftmost.
    pad is kept for call compatibility; the header position is computed from the laid-out panel."""
    ax._cmc_letter, ax._cmc_heading = letter, text

def _content_box(fig, ax, r):
    """Everything a panel draws: its tight box, the full extent of its axis labels (matplotlib clips long labels to the
    axis span) and any twin axes sharing its position."""
    from matplotlib.transforms import Bbox
    members = [ax] + [o for o in fig.axes if o is not ax and not getattr(o, "_cmc_letter", None)
                      and o.get_position().bounds == ax.get_position().bounds]
    boxes = []
    for a in members:
        boxes.append(a.get_tightbbox(r))
        for label in (a.xaxis.label, a.yaxis.label):
            if label.get_visible() and label.get_text():
                boxes.append(label.get_window_extent(r))
    return Bbox.union(boxes)

def _groups(axes, key, tol=0.03):
    """Axes whose key (a figure-fraction coordinate) agrees within tol, i.e. one grid row or column."""
    out = []
    for ax in sorted(axes, key=key):
        if out and abs(key(out[-1][0]) - key(ax)) < tol: out[-1].append(ax)
        else: out.append([ax])
    return out

def place_panel_letters(fig, pad_pt=2.0, gap_pt=4.0):
    """Put each registered letter above the highest and left of the leftmost element of its panel, aligned across its
    grid row (common baseline) and column (common left edge); a heading follows the letter on the same line.
    Returns descriptions of any header that overlaps another panel."""
    panels = [ax for ax in fig.axes if getattr(ax, "_cmc_letter", None)]
    if not panels: return []
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    box = {ax: _content_box(fig, ax, r) for ax in panels}
    pos = {ax: ax.get_position() for ax in panels}
    top, left = {}, {}
    for row in _groups(panels, lambda a: -pos[a].y1):
        y = max(box[a].y1 for a in row); top.update({a: y for a in row})
    for col in _groups(panels, lambda a: pos[a].x0):
        x = min(box[a].x0 for a in col); left.update({a: x for a in col})
    px = fig.dpi / 72.0; inv = fig.transFigure.inverted(); headers = []
    for ax in panels:
        base = top[ax] + (pad_pt + 0.3 * LETTER_SIZE) * px          # room for descenders under the baseline
        letter = fig.text(*inv.transform((left[ax] - pad_pt * px, base)), ax._cmc_letter, ha="left", va="baseline",
                          fontsize=LETTER_SIZE, fontweight="bold", color=INK)
        headers.append((ax, letter))
        if ax._cmc_heading:
            hx = max(ax.get_window_extent(r).x0, letter.get_window_extent(r).x1 + gap_pt * px)
            headers.append((ax, fig.text(*inv.transform((hx, base)), ax._cmc_heading, ha="left", va="baseline",
                                         fontsize=HEADING_SIZE, color=INK)))
    problems = []
    extents = [(owner, text, text.get_window_extent(r).padded(gap_pt * px / 2)) for owner, text in headers]
    for owner, text, tb in extents:
        for ax in panels:
            if ax is not owner and tb.overlaps(box[ax]):
                problems.append(f"header '{text.get_text()}' of panel {owner._cmc_letter} overlaps panel {ax._cmc_letter}")
        for other_owner, other, ob in extents:
            if other_owner is not owner and id(text) < id(other) and tb.overlaps(ob):
                problems.append(f"header '{text.get_text()}' of panel {owner._cmc_letter} overlaps header "
                                f"'{other.get_text()}' of panel {other_owner._cmc_letter}")
    return problems
