import io
import threading

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from StepwiseUCS import GridVisualizer, SearchContext, loc_to_xy, _is_grid_map


_mpl_lock = threading.Lock()


def _build_figure(cityMap):
    all_coords = [loc_to_xy(loc, cityMap) for loc in cityMap.geoLocations]
    all_x = [c[0] for c in all_coords]
    all_y = [c[1] for c in all_coords]
    if _is_grid_map(cityMap):
        grid_w = int(max(all_x) - min(all_x)) + 1
        grid_h = int(max(all_y) - min(all_y)) + 1
    else:
        grid_w, grid_h = 8, 8
    MAX_GRAPH = 10
    scale = min(MAX_GRAPH / (grid_w * 2), MAX_GRAPH / (grid_h * 2), 1.0)
    # Tables moved to the frontend; the figure now holds only the main graph
    # plus a left margin for the legend box anchored outside the axes.
    graph_w_in = grid_w * 2 * scale
    graph_h_in = grid_h * 2 * scale
    LEFT_PAD_IN = 1.6
    RIGHT_PAD_IN = 0.4
    fig_w = LEFT_PAD_IN + graph_w_in + RIGHT_PAD_IN
    fig_h = max(graph_h_in + 1.5, 6.0)
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.subplots_adjust(
        left=LEFT_PAD_IN / fig_w,
        right=(LEFT_PAD_IN + graph_w_in) / fig_w,
        top=0.93,
        bottom=0.13,
    )
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    return fig, ax


class StepRenderer:
    """Render each SearchContext to PNG bytes. Reuses one Figure per session."""

    def __init__(self, cityMap, visualizer=None):
        self.visualizer = visualizer or GridVisualizer()
        with _mpl_lock:
            self.fig, self.ax = _build_figure(cityMap)

    def render(self, ctx: SearchContext) -> bytes:
        with _mpl_lock:
            self.visualizer.draw_graph(self.ax, ctx)
            buf = io.BytesIO()
            self.fig.savefig(buf, format="png")
        buf.seek(0)
        return buf.getvalue()

    def close(self):
        with _mpl_lock:
            plt.close(self.fig)
