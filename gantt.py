"""
gantt.py
--------
Table-style Gantt chart generator.

HOW TO ADD TASKS
----------------
Edit the TASKS list below. Each task is a dict with:
    group  : str   — "Proposal" | "First Report" | "Final Report"
    name   : str   — task label shown in the left column
    start  : int   — starting week number (4–14)
    end    : int   — ending week number   (4–14, inclusive)
    owner  : str   — "mustafa" | "furkan" | "together"

COLOR CODES
-----------
    Mustafa  →  #2196F3  (blue)
    Furkan   →  #4CAF50  (green)
    Together →  #FF9800  (orange)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────
#  COLOUR MAP
# ─────────────────────────────────────────────────────────────
COLORS = {
    "mustafa"  : "#2196F3",   # blue
    "furkan"   : "#4CAF50",   # green
    "together" : "#FF9800",   # orange
}

# ─────────────────────────────────────────────────────────────
#  PHASE / GROUP DEFINITIONS
# ─────────────────────────────────────────────────────────────
PHASES = [
    {"label": "Proposal",     "start": 4,  "end": 5},
    {"label": "First Report", "start": 6,  "end": 8},
    {"label": "Final Report", "start": 9,  "end": 14},
]

# ─────────────────────────────────────────────────────────────
#  TASKS  ← add / remove tasks here
# ─────────────────────────────────────────────────────────────
TASKS = [
    # ── Proposal (weeks 4–5) ──────────────────────────────────
    {"group": "Proposal", "name": "Define problem & dataset",  "start": 4, "end": 4, "owner": "together"},
    {"group": "Proposal", "name": "Literature review",         "start": 4, "end": 5, "owner": "mustafa"},
    {"group": "Proposal", "name": "Write proposal document",   "start": 5, "end": 5, "owner": "together"},

    # ── First Report (weeks 6–8) ──────────────────────────────
    {"group": "First Report", "name": "Data preparation pipeline",    "start": 6, "end": 6, "owner": "mustafa"},
    {"group": "First Report", "name": "kNN implementation",           "start": 6, "end": 7, "owner": "mustafa"},
    {"group": "First Report", "name": "Decision Tree implementation",  "start": 6, "end": 7, "owner": "furkan"},
    {"group": "First Report", "name": "Logistic Regression impl.",    "start": 7, "end": 7, "owner": "furkan"},
    {"group": "First Report", "name": "Initial evaluation & plots",   "start": 7, "end": 8, "owner": "together"},
    {"group": "First Report", "name": "Write first report",           "start": 8, "end": 8, "owner": "together"},

    # ── Final Report (weeks 9–14) ─────────────────────────────
    {"group": "Final Report", "name": "Hyperparameter tuning (CV)", "start": 9,  "end": 10, "owner": "mustafa"},
    {"group": "Final Report", "name": "Cross-model comparison",     "start": 10, "end": 11, "owner": "together"},
    {"group": "Final Report", "name": "Error analysis & metrics",   "start": 11, "end": 12, "owner": "furkan"},
    {"group": "Final Report", "name": "Visualisations & plots",     "start": 11, "end": 12, "owner": "mustafa"},
    {"group": "Final Report", "name": "Final report writing",       "start": 12, "end": 13, "owner": "together"},
    {"group": "Final Report", "name": "Review & proofreading",      "start": 13, "end": 14, "owner": "together"},
    {"group": "Final Report", "name": "Presentation prep",          "start": 14, "end": 14, "owner": "together"},
]

# ─────────────────────────────────────────────────────────────
#  RENDERING
# ─────────────────────────────────────────────────────────────

WEEK_START = 4
WEEK_END   = 14
WEEKS      = list(range(WEEK_START, WEEK_END + 1))
N_WEEKS    = len(WEEKS)

# Cell dimensions (in figure data units)
GRP_W   = 0.55   # group label column width
TASK_W  = 3.20   # task name column width
WEEK_W  = 1.00   # each week column width
PHASE_H = 0.60   # phase header row height
LABEL_H = 0.55   # week label row height
TASK_H  = 0.62   # each task row height

# Style
BORDER      = "#aaaaaa"
PHASE_FILLS = {
    "Proposal"     : "#C8D9C9",   # military green tint
    "First Report" : "#C8CDD9",   # steel blue tint
    "Final Report" : "#D9C8C8",   # dark red tint
}
PHASE_HDR = {
    "Proposal"     : "#5D7A61",   # military green
    "First Report" : "#3D5A80",   # steel blue
    "Final Report" : "#8B2E2E",   # dark crimson
}
ROW_EVEN = "#fafafa"
ROW_ODD  = "#f2f2f2"


def _x(week):
    """Left edge x-coordinate of a week column."""
    return GRP_W + TASK_W + (week - WEEK_START) * WEEK_W


def _y(row):
    """Bottom edge y-coordinate of a data row (0 = topmost task row)."""
    total_header = PHASE_H + LABEL_H
    return total_header + row * TASK_H


def draw_rect(ax, x, y, w, h, fc, ec=BORDER, lw=0.7, zorder=1):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="square,pad=0",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        zorder=zorder,
    ))


def draw_gantt(tasks=TASKS, save_path="gantt.png"):
    n_tasks = len(tasks)
    total_w = GRP_W + TASK_W + N_WEEKS * WEEK_W
    total_h = PHASE_H + LABEL_H + n_tasks * TASK_H + 0.3   # +0.3 for legend row

    fig, ax = plt.subplots(figsize=(total_w * 0.95, total_h * 1.1))
    ax.set_xlim(0, total_w)
    ax.set_ylim(total_h, 0)   # y increases downward
    ax.axis("off")

    # ── Title ────────────────────────────────────────────────
    ax.text(
        total_w / 2, -0.05,
        "EEE 485 Term Project",
        ha="center", va="bottom",
        fontsize=20, fontweight="bold",
    )

    # ── Phase header row ─────────────────────────────────────
    # Grey cells for group/task header columns
    draw_rect(ax, 0,       0, GRP_W,  PHASE_H, "#e0e0e0")
    draw_rect(ax, GRP_W,   0, TASK_W, PHASE_H, "#e0e0e0")

    for phase in PHASES:
        px     = _x(phase["start"])
        pw     = (phase["end"] - phase["start"] + 1) * WEEK_W
        draw_rect(ax, px, 0, pw, PHASE_H, PHASE_HDR[phase["label"]], lw=1.2)
        ax.text(
            px + pw / 2, PHASE_H / 2,
            phase["label"],
            ha="center", va="center",
            fontsize=18, fontweight="bold",
        )

    # ── Week label row ───────────────────────────────────────
    y_lbl = PHASE_H
    draw_rect(ax, 0,     y_lbl, GRP_W,  LABEL_H, "#e8e8e8")
    draw_rect(ax, GRP_W, y_lbl, TASK_W, LABEL_H, "#e8e8e8")
    ax.text(GRP_W + TASK_W / 2, y_lbl + LABEL_H / 2,
            "Task Name", ha="center", va="center",
            fontsize=14, fontweight="bold")

    for week in WEEKS:
        # Find which phase this week belongs to for bg colour
        phase_fill = "#e8e8e8"
        for phase in PHASES:
            if phase["start"] <= week <= phase["end"]:
                phase_fill = PHASE_HDR[phase["label"]]
                break
        draw_rect(ax, _x(week), y_lbl, WEEK_W, LABEL_H, phase_fill)
        ax.text(_x(week) + WEEK_W / 2, y_lbl + LABEL_H / 2,
                f"Week {week}", ha="center", va="center", fontsize=12, fontweight="bold")

    # ── Task rows ────────────────────────────────────────────
    # Pre-compute group spans for vertical group labels
    group_spans = {}   # group_name -> (first_row, last_row)
    for i, task in enumerate(tasks):
        g = task["group"]
        if g not in group_spans:
            group_spans[g] = [i, i]
        else:
            group_spans[g][1] = i

    for row, task in enumerate(tasks):
        y_row  = _y(row)
        bg     = ROW_EVEN if row % 2 == 0 else ROW_ODD

        # Group label cell (only draw on first row of group)
        g = task["group"]
        if group_spans[g][0] == row:
            span_rows = group_spans[g][1] - group_spans[g][0] + 1
            span_h    = span_rows * TASK_H
            draw_rect(ax, 0, y_row, GRP_W, span_h, PHASE_FILLS[g], lw=1.2)
            ax.text(
                GRP_W / 2, y_row + span_h / 2,
                g,
                ha="center", va="center",
                fontsize=14, fontweight="bold",
                rotation=90,
            )

        # Task name cell
        draw_rect(ax, GRP_W, y_row, TASK_W, TASK_H, bg)
        ax.text(
            GRP_W + 0.15, y_row + TASK_H / 2,
            task["name"],
            ha="left", va="center",
            fontsize=12,
        )

        # Week cells
        for week in WEEKS:
            # Background tinted by phase
            phase_bg = bg
            for phase in PHASES:
                if phase["start"] <= week <= phase["end"]:
                    base = PHASE_FILLS[phase["label"]]
                    # Blend with row bg slightly
                    phase_bg = base if row % 2 == 0 else base
                    break
            draw_rect(ax, _x(week), y_row, WEEK_W, TASK_H, phase_bg)

        # Coloured bar spanning active weeks
        if task["start"] <= task["end"]:
            bar_x = _x(task["start"]) + 0.04
            bar_w = (task["end"] - task["start"] + 1) * WEEK_W - 0.08
            bar_y = y_row + TASK_H * 0.18
            bar_h = TASK_H * 0.64
            color = COLORS.get(task["owner"].lower(), "#9E9E9E")
            draw_rect(ax, bar_x, bar_y, bar_w, bar_h,
                      fc=color, ec="white", lw=1.0, zorder=3)

    # ── Phase separator lines ─────────────────────────────────
    group_names = list(group_spans.keys())
    for g in group_names[:-1]:   # draw after every group except the last
        sep_y = _y(group_spans[g][1]) + TASK_H
        ax.plot([0, total_w], [sep_y, sep_y],
                color="#444444", linewidth=2.0, zorder=6)

    # ── Outer border ─────────────────────────────────────────
    outer_h = PHASE_H + LABEL_H + n_tasks * TASK_H
    draw_rect(ax, 0, 0, total_w, outer_h, fc="none", ec="#555555", lw=1.5, zorder=5)

    # ── Legend ───────────────────────────────────────────────
    legend_y = PHASE_H + LABEL_H + n_tasks * TASK_H + 0.08
    items = [
        ("Mustafa",  COLORS["mustafa"]),
        ("Furkan",   COLORS["furkan"]),
        ("Together", COLORS["together"]),
    ]
    lx = total_w
    for label, color in reversed(items):
        lx -= 1.4
        draw_rect(ax, lx, legend_y, 0.25, 0.22, fc=color, ec="white", lw=0.8, zorder=3)
        ax.text(lx + 0.32, legend_y + 0.11, label,
                ha="left", va="center", fontsize=10)

    plt.tight_layout(pad=0.3)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    print(f"Gantt chart saved to {save_path}")


if __name__ == "__main__":
    draw_gantt()
