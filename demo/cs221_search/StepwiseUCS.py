import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap
from dataclasses import dataclass
from typing import List, Optional

from MapUtils import (
    CityMap,
    checkValid,
    computeDistance,
    createGridMap,
    getTotalCost,
    makeGridLabel,
    makeTag,
)
from SearchProblemUtils import (
    PriorityQueue,
    Heuristic,
    SearchProblem,
    State,
)
from UniformCostSearch import UniformCostSearch


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_label(label: str):
    """Convert "x,y" grid label to (int, int)."""
    x, y = label.split(",")
    return int(x), int(y)

def _is_grid_map(cityMap) -> bool:
    """Return True if location labels are 'x,y' grid format (vs OSM node IDs)."""
    sample = next(iter(cityMap.geoLocations))
    return "," in sample

def loc_to_xy(loc: str, cityMap) -> tuple:
    """Return (x, y) plot coordinates for a location.
    Grid maps : parse 'x,y' label as integers.
    Geo maps  : use (longitude, latitude) from geoLocations.
    """
    if "," in loc:
        x, y = loc.split(",")
        return int(x), int(y)
    geo = cityMap.geoLocations[loc]
    return geo.longitude, geo.latitude

def _format_loc(loc: str) -> str:
    """Human-readable location string for sidebar tables.
    Grid: '(x,y)'.  Geo: last 8 chars of the OSM node ID, prefixed with '…'.
    """
    if "," in loc:
        x, y = loc.split(",")
        return f"({x},{y})"
    return f"\u2026{loc[-8:]}" if len(loc) > 8 else loc

def reconstruct_path(state, start_state, backpointers):
    """Walk backpointers from `state` back to `start_state`."""
    path = []
    cur = state
    while cur != start_state:
        path.append(cur.location)
        _, cur = backpointers[cur]
    path.append(start_state.location)
    path.reverse()
    return path


# ── search context ────────────────────────────────────────────────────────────

@dataclass
class SearchContext:
    """All state needed by a visualizer for one search step."""
    cityMap:             CityMap
    startLoc:            str
    endTag:              str
    current_loc:         str
    path:                list
    backpointers:        dict
    explored:            list
    pastCost:            float
    neighbors:           set
    discovered_edges:    dict
    frontier_priorities: dict
    step:                int
    # Problem-specific extras (None when not applicable)
    memory:              object = None   # frozenset — WaypointsShortestPath
    waypoint_tags:       object = None   # tuple    — WaypointsShortestPath
    heuristic_values:    object = None   # dict     — A*
    neighbor_costs:      object = None   # dict location→cost as returned by the problem
    updated_frontier:    object = None   # set of states whose frontier cost was lowered this step
    is_final:            bool   = False  # True on the step where the goal is reached


# ── drawing utilities ─────────────────────────────────────────────────────────

_ROW_H_IN   = 0.28   # desired physical row height in inches
_TITLE_H    = 0.04   # axes-fraction gap between a section title and its table
_GAP        = 0.07   # axes-fraction gap between table groups
_WRAP_WIDTH = 38     # approximate characters per line in a 260 px column at fontsize 9

def _row_h(tax):
    """Convert _ROW_H_IN to axes fraction for the given axes."""
    fig_h = tax.figure.get_size_inches()[1]
    ax_h  = tax.get_position().height
    return _ROW_H_IN / (fig_h * ax_h) if ax_h > 0 else 0.08

def _draw_text_box(tax, cursor, text, fc, ec="darkgrey", fontsize=9):
    """Draw a full-width text box with wrapped text. Returns the new cursor position."""
    # Wrap each logical line independently
    lines = []
    for line in text.split('\n'):
        wrapped = textwrap.wrap(line, width=_WRAP_WIDTH)
        lines.extend(wrapped if wrapped else [''])
    wrapped_text = '\n'.join(lines)
    n_lines = len(lines)

    # Estimate box height in axes fraction
    fig_h    = tax.figure.get_size_inches()[1]
    ax_h     = tax.get_position().height
    line_h   = (_ROW_H_IN * fontsize / 12) / (fig_h * ax_h) if ax_h > 0 else 0.05
    pad_frac = 0.015
    box_h    = n_lines * line_h + 2 * pad_frac

    # Full-width background rectangle
    tax.add_patch(mpatches.Rectangle(
        (0.0, cursor - box_h), 1.0, box_h,
        facecolor=fc, edgecolor=ec, linewidth=0.8,
        transform=tax.transAxes, clip_on=False
    ))
    # Text on top (no bbox — background is handled by the patch)
    tax.text(0.03, cursor - pad_frac, wrapped_text,
             fontsize=fontsize, va="top", ha="left",
             transform=tax.transAxes)
    return cursor - box_h - _GAP

def _draw_table(tax, top, rows, col_labels, header_color, row_h):
    """Draw one table in `tax` with its top edge at axes-fraction `top`."""
    if not rows:
        return top
    n   = len(rows) + 1   # +1 for header
    h   = n * row_h
    tbl = tax.table(cellText=rows, colLabels=col_labels,
                    bbox=[0, top - h, 1, h], cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    for col in range(len(col_labels)):
        tbl[0, col].set_facecolor(header_color)
    return top - h


# ── visualizer hierarchy ──────────────────────────────────────────────────────

class GridVisualizer:
    """
    Draws a grid-based search step.  Contains all shared rendering logic.

    Override these hooks in subclasses for problem-specific rendering:
      get_title(ctx)             → plot title string
      _draw_extra_graph(ax, ctx) → extra elements on the main axes
      _get_extra_sections(ctx)   → extra sidebar table sections (appended after the standard three)
    """

    def get_title(self, ctx: SearchContext) -> str:
        return "ShortestPathProblem"

    # ── main entry points (called by StepwiseMixin._on_step) ─────────────────

    def draw_graph(self, ax, ctx: SearchContext):
        """Draw all graph elements on the main axes."""
        ax.clear()
        ax.set_aspect("equal", adjustable="box")

        is_grid      = _is_grid_map(ctx.cityMap)
        L            = lambda loc: loc_to_xy(loc, ctx.cityMap)   # coordinate shorthand
        explored_set = set(ctx.explored)
        all_x = [loc_to_xy(loc, ctx.cityMap)[0] for loc in ctx.cityMap.geoLocations]
        all_y = [loc_to_xy(loc, ctx.cityMap)[1] for loc in ctx.cityMap.geoLocations]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        cx, cy = L(ctx.current_loc)
        DONE = -100000

        # Scale-appropriate offsets for text labels
        rng_x  = max_x - min_x or 1
        rng_y  = max_y - min_y or 1
        _dx    = 0.08  if is_grid else rng_x * 0.012   # small offset (edge cost labels, coord labels)
        _lx    = 0.15  if is_grid else rng_x * 0.022   # larger offset (named badges)
        _ly    = 0.15  if is_grid else rng_y * 0.022

        # ── Priority scheme (highest priority = smallest/thinnest/topmost) ──
        # P1 start/end  : size=8,  no line,        z_node=11
        # P2 current    : size=11, lw=1.5 blue,    z_node=10, z_line=9
        # P3 frontier   : size=14, lw=2.5 orange,  z_node=8,  z_line=7
        # P4 neighbor   : size=17, lw=3.5 yellow,  z_node=6,  z_line=5
        # P5 explored   : size=20, lw=5.0 skyblue, z_node=4,  z_line=3
        # gray grid: z=1,  gray bg nodes: z=2,  text: z=12

        # Gray grid lines (grid maps only — z=1)
        if is_grid:
            for gx in range(int(min_x), int(max_x) + 2):
                ax.plot([gx, gx], [min_y, max_y + 1], color="lightgray", linewidth=0.6, zorder=1)
            for gy in range(int(min_y), int(max_y) + 2):
                ax.plot([min_x, max_x + 1], [gy, gy], color="lightgray", linewidth=0.6, zorder=1)

        # P5 explored edges (z=3, thickest)
        drawn_edges = set()
        for loc in explored_set:
            x1, y1 = L(loc)
            for nbr in ctx.cityMap.distances[loc]:
                if nbr in explored_set:
                    edge = tuple(sorted([loc, nbr]))
                    if edge not in drawn_edges:
                        x2, y2 = L(nbr)
                        ax.plot([x1, x2], [y1, y2], color="skyblue", linewidth=5.0, zorder=3)
                        drawn_edges.add(edge)

        # P4 neighbor lines — current→each neighbor (z=5)
        for nbr in ctx.neighbors:
            nx, ny = L(nbr)
            ax.plot([cx, nx], [cy, ny], color="yellow", linewidth=3.5, zorder=5)

        # P3 frontier lines — each frontier node → its backpointer parent (z=7)
        for state, p in ctx.frontier_priorities.items():
            if p != DONE and state in ctx.backpointers:
                _, parent_state = ctx.backpointers[state]
                fx, fy = L(state.location)
                px, py = L(parent_state.location)
                ax.plot([fx, px], [fy, py], color="orange", linewidth=2.5, zorder=7)

        # P2 current path (z=9, thinnest colored line)
        if len(ctx.path) > 1:
            xs = [L(loc)[0] for loc in ctx.path]
            ys = [L(loc)[1] for loc in ctx.path]
            ax.plot(xs, ys, color="blue", linewidth=1.5, zorder=9)

        # Gray background nodes (z=2)
        for loc in ctx.cityMap.geoLocations:
            x, y = L(loc)
            ax.plot(x, y, "o", color="lightgray", markersize=4, zorder=2)

        # P5 explored nodes (z=4, largest)
        for loc in ctx.explored:
            x, y = L(loc)
            ax.plot(x, y, "o", color="skyblue", markersize=20, zorder=4)

        # P4 neighbor nodes (z=6)
        for nbr in ctx.neighbors:
            nx, ny = L(nbr)
            ax.plot(nx, ny, "o", color="yellow", markersize=17, zorder=6)

        # P3 frontier nodes (z=8)
        for state, p in ctx.frontier_priorities.items():
            if p != DONE:
                fx, fy = L(state.location)
                ax.plot(fx, fy, "o", color="orange", markersize=14, zorder=8)

        # "lower cost found" labels on updated frontier nodes (z=12)
        if ctx.updated_frontier:
            for state in ctx.updated_frontier:
                fx, fy = L(state.location)
                ax.text(fx + _lx, fy - _ly, "lower cost found",
                        fontsize=9, color="black", ha="left", va="top", zorder=12,
                        bbox=dict(boxstyle="round,pad=0.3", fc="orange", ec="darkgrey", alpha=0.95))

        # P2 current node (z=10)
        ax.plot(cx, cy, "o", color="blue", markersize=11, zorder=10)

        # P1 end nodes — line if all share a grid axis, otherwise circles (z=11)
        end_coords = [L(loc) for loc in ctx.cityMap.geoLocations
                      if ctx.endTag in ctx.cityMap.tags[loc]]
        if len(end_coords) > 1:
            ex = [c[0] for c in end_coords]
            ey = [c[1] for c in end_coords]
            if is_grid and len(set(ex)) == 1:
                xv = ex[0]
                ax.plot([xv, xv], [min(ey), max(ey)], color="red", linewidth=3, zorder=11)
                ax.text(xv + _dx, max(ey) - _dx, f"x={xv}",
                        fontsize=9, color="red", ha="left", va="top", zorder=12)
            elif is_grid and len(set(ey)) == 1:
                yv = ey[0]
                ax.plot([min(ex), max(ex)], [yv, yv], color="red", linewidth=3, zorder=11)
                ax.text(max(ex) + _dx, yv - _dx, f"y={yv}",
                        fontsize=9, color="red", ha="left", va="top", zorder=12)
            else:
                for x, y in end_coords:
                    ax.plot(x, y, "o", color="red", markersize=8, zorder=11)
        elif end_coords:
            x, y = end_coords[0]
            ax.plot(x, y, "o", color="red", markersize=8, zorder=11)
            ax.text(x + _dx, y - _dx, _format_loc(ctx.endTag),
                    fontsize=9, color="red", ha="left", va="top", zorder=12)

        # "end" badge — bottom-left of the first end coordinate
        if end_coords:
            ex0, ey0 = end_coords[0]
            ax.text(ex0 - _lx, ey0 - _ly, "end",
                    fontsize=9, color="black", ha="right", va="top", zorder=12,
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightsalmon", ec="darkgrey", alpha=0.95))

        # P1 start node (z=11)
        sx, sy = L(ctx.startLoc)
        ax.plot(sx, sy, "o", color="green", markersize=8, zorder=11)
        ax.text(sx + _dx, sy - _dx, _format_loc(ctx.startLoc),
                fontsize=9, color="green", ha="left", va="top", zorder=12)
        # "start" badge — bottom-left of the start node
        ax.text(sx - _lx, sy - _ly, "start",
                fontsize=9, color="black", ha="right", va="top", zorder=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="darkgrey", alpha=0.95))

        # Current node info box
        info_dx = 1.0 if is_grid else rng_x * 0.15
        info_dy = 0.4 if is_grid else rng_y * 0.08
        ax.text(cx + info_dx, cy + info_dy,
                f"step: {ctx.step}, cost: {ctx.pastCost:.2f}",
                fontsize=11, color="black", ha="center", va="bottom", zorder=20,
                bbox=dict(boxstyle="round,pad=0.6", fc="#e8e8e8", ec="darkgrey", alpha=0.95))

        # Edge cost labels (grid maps only — too cluttered on geo maps)
        if is_grid:
            for (loc, nbr), cost in ctx.discovered_edges.items():
                lx, ly = L(loc)
                nx, ny = L(nbr)
                mx, my = (lx + nx) / 2, (ly + ny) / 2
                if ly == ny:
                    ax.text(mx, my - _dx, f"{cost:.2f}", fontsize=9, color="red",
                            ha="center", va="top", zorder=12)
                else:
                    ax.text(mx + _dx, my, f"{cost:.2f}", fontsize=9, color="red",
                            ha="left", va="center", zorder=12)

        # Axis setup
        if is_grid:
            ax.set_xticks(range(int(min_x), int(max_x) + 2))
            ax.set_yticks(range(int(min_y), int(max_y) + 2))
            ax.set_xlim(min_x - 0.5, max_x + 1.5)
            ax.set_ylim(min_y - 0.5, max_y + 1.5)
        else:
            margin_x = rng_x * 0.05
            margin_y = rng_y * 0.05
            ax.set_xlim(min_x - margin_x, max_x + margin_x)
            ax.set_ylim(min_y - margin_y, max_y + margin_y)
            ax.tick_params(labelsize=7)
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))

        # Legend
        legend = [
            mpatches.Patch(color="green",   label="Start"),
            mpatches.Patch(color="red",     label="End"),
            mpatches.Patch(color="blue",    label="Current"),
            mpatches.Patch(color="orange",  label="Frontier"),
            mpatches.Patch(color="skyblue", label="Explored"),
            mpatches.Patch(color="yellow",  label="Neighbors"),
        ]
        ax.legend(handles=legend, loc="upper right", bbox_to_anchor=(-0.02, 1.0),
                  borderaxespad=0, fontsize=8)
        ax.set_title(self.get_title(ctx), fontsize=12, pad=8)
        ax.set_xlabel("longitude" if not is_grid else "x")
        ax.set_ylabel("latitude"  if not is_grid else "y")

        # Hook: subclasses add extra graph elements here
        self._draw_extra_graph(ax, ctx)

    def draw_tables(self, tax, ctx: SearchContext):
        """Draw all sidebar table sections, truncating any that would overflow."""
        tax.clear()
        tax.axis("off")
        rh     = _row_h(tax)
        cursor = 0.97
        for title, rows, col_labels, header_color in self._get_sections(ctx):
            # Skip section entirely if there isn't room for title + header + 1 row + gap
            if cursor - _TITLE_H - 2 * rh - _GAP < 0:
                break
            tax.text(0.05, cursor, title, fontsize=11, fontweight="bold",
                     transform=tax.transAxes, va="top", ha="left")
            cursor -= _TITLE_H
            # Truncate rows that would overflow below the gap margin
            max_data = max(1, int((cursor - _GAP) / rh) - 1)  # -1 for header row
            if len(rows) > max_data:
                n_hidden = len(rows) - (max_data - 1)
                rows = rows[:max_data - 1] + [[f"({n_hidden} more)"] + [""] * (len(col_labels) - 1)]
            cursor  = _draw_table(tax, cursor, rows, col_labels, header_color, rh)
            cursor -= _GAP

        cursor = self._draw_extra_bottom(tax, cursor, ctx)

        if ctx.is_final:
            done_text = f"Done!\ncost: {ctx.pastCost:.0f}\npath: {str(ctx.path)}"
            _draw_text_box(tax, cursor, done_text, fc="#e8e8e8")

    def draw_frontier_table(self, tax_frontier, ctx: SearchContext):
        """Draw the Frontier table in its own column with full vertical space."""
        tax_frontier.clear()
        tax_frontier.axis("off")
        DONE = -100000
        active = sorted(
            [(state.location, p) for state, p in ctx.frontier_priorities.items() if p != DONE],
            key=lambda x: x[1],
        )
        rows = [
            [_format_loc(loc), f"{cost:.2f}"]
            for loc, cost in active
        ]
        rh     = _row_h(tax_frontier)
        cursor = 0.97
        tax_frontier.text(0.05, cursor, "Frontier", fontsize=11, fontweight="bold",
                          transform=tax_frontier.transAxes, va="top", ha="left")
        cursor -= _TITLE_H
        # Truncate if rows overflow
        max_data = max(1, int((cursor - _GAP) / rh) - 1)
        if len(rows) > max_data:
            n_hidden = len(rows) - (max_data - 1)
            rows = rows[:max_data - 1] + [[f"({n_hidden} more)", ""]]
        _draw_table(tax_frontier, cursor, rows, ["location", "past cost"], "orange", rh)

    # ── hooks ─────────────────────────────────────────────────────────────────

    def _draw_extra_graph(self, ax, ctx: SearchContext):
        """Override to draw extra elements on the main axes (problem-specific)."""
        pass

    def _draw_extra_bottom(self, tax, cursor, ctx: SearchContext) -> float:
        """Override to draw extra content below the main tables. Return updated cursor."""
        return cursor

    def _get_sections(self, ctx: SearchContext) -> list:
        """
        Build the full list of sidebar table sections.
        Calls _get_extra_sections() so subclasses only need to override that.
        """
        # Use problem-level costs (ctx.neighbor_costs) so A* reduced costs are shown
        # correctly; fall back to raw map distances if not available.
        costs_source = ctx.neighbor_costs if ctx.neighbor_costs is not None \
                       else ctx.cityMap.distances[ctx.current_loc]
        neighbor_rows = [
            [_format_loc(nbr), f"{cost:.2f}"]
            for nbr, cost in sorted(costs_source.items(), key=lambda x: x[1])
        ]

        return [
            ("Current Location", [[_format_loc(ctx.current_loc), f"{ctx.pastCost:.2f}"]], ["location", "past cost"], "cornflowerblue"),
            ("Neighbors",        neighbor_rows,                                            ["location", "edge cost"], "yellow"),
        ] + self._get_extra_sections(ctx)

    def _get_extra_sections(self, ctx: SearchContext) -> list:
        """Override to append problem-specific table sections after the standard three."""
        return []


class WaypointsVisualizer(GridVisualizer):
    """
    Extends GridVisualizer for WaypointsShortestPathProblem.
    Adds:
      - Star markers on waypoint nodes (filled gold = covered, hollow = not yet covered)
      - A 'Waypoints' sidebar table showing each waypoint tag and its covered status
    """

    def get_title(self, ctx: SearchContext) -> str:
        return "WaypointsShortestPathProblem"

    def _draw_extra_graph(self, ax, ctx: SearchContext):
        """Mark waypoint nodes with stars; label explored waypoint nodes."""
        if ctx.waypoint_tags is None:
            return
        is_grid      = _is_grid_map(ctx.cityMap)
        all_x        = [loc_to_xy(loc, ctx.cityMap)[0] for loc in ctx.cityMap.geoLocations]
        all_y        = [loc_to_xy(loc, ctx.cityMap)[1] for loc in ctx.cityMap.geoLocations]
        _lx          = 0.15 if is_grid else (max(all_x) - min(all_x)) * 0.022
        _ly          = 0.15 if is_grid else (max(all_y) - min(all_y)) * 0.022
        covered      = ctx.memory or frozenset()
        explored_set = set(ctx.explored)
        for loc in ctx.cityMap.geoLocations:
            loc_tags = ctx.cityMap.tags[loc]
            for tag in ctx.waypoint_tags:
                if tag in loc_tags:
                    x, y = loc_to_xy(loc, ctx.cityMap)
                    # Star marker: filled gold if covered, hollow if not yet
                    if tag in covered:
                        ax.plot(x, y, "*", color="gold", markersize=18,
                                markeredgecolor="darkorange", markeredgewidth=1.0, zorder=13)
                    else:
                        ax.plot(x, y, "*", color="none", markersize=18,
                                markeredgecolor="darkorange", markeredgewidth=1.5, zorder=13)
                    # Label the node once it has been explored
                    if loc in explored_set:
                        ax.text(x - _lx, y - _ly, tag,
                                fontsize=9, color="black", ha="right", va="top", zorder=12,
                                bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="darkgrey", alpha=0.95))

    def _get_extra_sections(self, ctx: SearchContext) -> list:
        """Append a static 'Waypoints' table listing all required waypoint tags in one row."""
        if ctx.waypoint_tags is None:
            return []
        tags_str = str(list(ctx.waypoint_tags))
        lines = textwrap.wrap(tags_str, width=_WRAP_WIDTH)
        rows = [[line] for line in lines] if lines else [[tags_str]]
        return [("Waypoints", rows, ["tags"], "lightgreen")]

    def _draw_extra_bottom(self, tax, cursor, ctx: SearchContext) -> float:
        """Draw the 'Current path' info box below the tables."""
        if ctx.waypoint_tags is None:
            return cursor
        tags_covered = sorted(ctx.memory) if ctx.memory is not None else []
        text = f"Current path:\nsteps: {str(ctx.path)}\ntags: {str(tags_covered)}"
        return _draw_text_box(tax, cursor, text, fc="lightblue")


# ── stepwise search mixin ─────────────────────────────────────────────────────

class StepwiseMixin:
    """
    Mixin that adds step-by-step visualization to any search algorithm.
    Combine with a search class via multiple inheritance:

        class StepwiseUCS(StepwiseMixin, UniformCostSearch): ...

    The search's solve() method builds a SearchContext each step and calls
    self._on_step(ctx), which delegates all drawing to self._visualizer.

    Parameters
    ----------
    cityMap       : CityMap
    startLocation : str
    endTag        : str
    visualizer    : GridVisualizer instance (default: GridVisualizer())
    auto_step     : bool   — advance without Enter if True
    step_delay    : float  — seconds between steps in auto mode
    """

    def __init__(self, cityMap, startLocation, endTag,
                 visualizer=None, auto_step=False, step_delay=0.5, **kwargs):
        super().__init__(**kwargs)
        self.cityMap     = cityMap
        self.startLoc    = startLocation
        self.endTag      = endTag
        self._visualizer = visualizer or GridVisualizer()
        self.auto_step   = auto_step
        self.step_delay  = step_delay
        self._setup_figure()

    def _setup_figure(self):
        all_coords = [loc_to_xy(loc, self.cityMap) for loc in self.cityMap.geoLocations]
        all_x_vals = [c[0] for c in all_coords]
        all_y_vals = [c[1] for c in all_coords]
        if _is_grid_map(self.cityMap):
            grid_w = int(max(all_x_vals) - min(all_x_vals)) + 1
            grid_h = int(max(all_y_vals) - min(all_y_vals)) + 1
        else:
            grid_w = 8   # fixed reasonable size for geo maps
            grid_h = 8
        MAX_GRAPH = 10
        scale = min(MAX_GRAPH / (grid_w * 2), MAX_GRAPH / (grid_h * 2), 1.0)
        fig_w = grid_w * 2 * scale + 2.5
        fig_h = grid_h * 2 * scale
        self._fig = plt.figure(figsize=(fig_w, fig_h))
        self._fig.subplots_adjust(left=0.25, right=0.64, top=0.93, bottom=0.13)
        self._ax  = self._fig.add_subplot(111)
        self._ax.set_aspect("equal", adjustable="box")
        self._tax          = self._fig.add_axes([0.66, 0.08, 0.32, 0.85])
        self._tax.axis("off")
        self._tax_frontier = self._fig.add_axes([0.66, 0.08, 0.32, 0.85])  # placeholder
        self._tax_frontier.axis("off")
        self._fig.canvas.mpl_connect(
            "resize_event",
            lambda e: (self._reposition_tax(), self._fig.canvas.draw_idle()),
        )
        plt.ion()

    def _reposition_tax(self):
        """Snap all three table columns flush against the graph, each 20 px apart."""
        try:
            renderer = self._fig.canvas.get_renderer()
            ax_bb = self._ax.get_tightbbox(renderer)
            if ax_bb is None:
                return
            fw, fh = self._fig.bbox.width, self._fig.bbox.height
            bottom = ax_bb.y0 / fh
            height = (ax_bb.y1 - ax_bb.y0) / fh
            GAP_PX = 20 / fw

            left  = (ax_bb.x1 + 20) / fw
            width = min(260 / fw, 1.0 - left - 0.01)
            self._tax.set_position([left, bottom, width, height])

            left2  = left + width + GAP_PX
            width2 = min(260 / fw, 1.0 - left2 - 0.01)
            self._tax_frontier.set_position([left2, bottom, width2, height])

        except Exception:
            pass

    def _on_step(self, ctx: SearchContext):
        """Draw one search step and pause."""
        self._visualizer.draw_graph(self._ax, ctx)
        plt.draw()                          # render graph — computes actual axes positions
        self._reposition_tax()              # snap table axes flush against graph right edge
        self._visualizer.draw_tables(self._tax, ctx)
        self._visualizer.draw_frontier_table(self._tax_frontier, ctx)
        self._fig.canvas.draw_idle()
        if self.auto_step:
            plt.pause(self.step_delay)
        else:
            plt.pause(0.05)
            input("  [Enter] next step, [Ctrl-C] quit ")


# ── stepwise search algorithms ────────────────────────────────────────────────

class StepwiseUCS(StepwiseMixin, UniformCostSearch):
    """
    UCS that visualizes each step.

    Usage:
        # ShortestPathProblem (default visualizer)
        ucs = StepwiseUCS(cityMap, startLoc, endTag)
        ucs.solve(ShortestPathProblem(startLoc, endTag, cityMap))

        # WaypointsShortestPathProblem (waypoints visualizer)
        ucs = StepwiseUCS(cityMap, startLoc, endTag, visualizer=WaypointsVisualizer())
        ucs.solve(WaypointsShortestPathProblem(startLoc, waypointTags, endTag, cityMap))
    """

    def solve(self, problem):
        self.actions           = None
        self.pathCost          = None
        self.numStatesExplored = 0
        self.pastCosts         = {}

        frontier         = PriorityQueue()
        backpointers     = {}
        explored         = []
        discovered_edges = {}
        updated_frontier = set()

        startState = problem.startState()
        frontier.update(startState, 0.0)

        while True:
            state, pastCost = frontier.removeMin()
            if state is None:
                return

            self.pastCosts[state.location] = pastCost
            self.numStatesExplored += 1
            explored.append(state.location)

            # Compute successors once — used for edge recording AND frontier update.
            # Using problem.successorsAndCosts (not cityMap.distances) ensures the
            # displayed edge costs reflect any problem-level modifications such as
            # the A* reduction: c'(s,s') = c + h(s') - h(s).
            successors = list(problem.successorsAndCosts(state))

            for action, newState, cost in successors:
                edge = tuple(sorted([state.location, newState.location]))
                if edge not in discovered_edges:
                    discovered_edges[edge] = cost

            path          = reconstruct_path(state, startState, backpointers)
            neighbor_locs  = set(self.cityMap.distances[state.location].keys())
            neighbor_costs = {newState.location: cost for _, newState, cost in successors}
            is_final      = problem.isEnd(state)

            ctx = SearchContext(
                cityMap             = self.cityMap,
                startLoc            = self.startLoc,
                endTag              = self.endTag,
                current_loc         = state.location,
                path                = path,
                backpointers        = backpointers,
                explored            = explored,
                pastCost            = pastCost,
                neighbors           = neighbor_locs,
                discovered_edges    = discovered_edges,
                frontier_priorities = frontier.priorities,
                step                = self.numStatesExplored,
                memory              = state.memory,
                waypoint_tags       = getattr(problem, "waypointTags", None),
                neighbor_costs      = neighbor_costs,
                updated_frontier    = updated_frontier,
                is_final            = is_final,
            )
            self._on_step(ctx)

            # Remove explored node from updated_frontier AFTER drawing, so the label
            # remains visible on the step where the node becomes the current node.
            updated_frontier.discard(state)

            if is_final:
                self.actions  = path[1:]
                self.pathCost = pastCost
                print(f"\nDone! cost={self.pathCost}, steps={self.numStatesExplored}")
                plt.ioff()
                plt.show()
                return

            for action, newState, cost in successors:
                old_priority = frontier.priorities.get(newState)
                if frontier.update(newState, pastCost + cost):
                    backpointers[newState] = (action, state)
                    # State was already in the frontier (not new, not DONE) → cost was lowered
                    if old_priority is not None and old_priority != -100000:
                        updated_frontier.add(newState)
