"""Headless renderer for one ValueIterationContext → PNG bytes."""

import io
import threading

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from StepwiseValueIteration import NumberLineVisualizer, ValueIterationContext, _BOX_HALF


_mpl_lock = threading.Lock()


class _HeadlessNumberLineVisualizer(NumberLineVisualizer):
    """Drop-in visualizer: headless, full-width graph, no breakdown sidebar.

    The Q-value breakdown is returned as JSON to the frontend instead of
    drawn into the image, so the figure can use the full width for the graph.
    """

    def _setup_figure(self, ctx: ValueIterationContext):
        n_states = len(self._all_states(ctx))
        fig_w = max(10, min(n_states * 1.2 + 2, 18))
        self._fig = plt.figure(figsize=(fig_w, 5.5))
        self._fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.12)
        self._ax = self._fig.add_subplot(111)

    def draw_step(self, ctx: ValueIterationContext):
        if self._fig is None:
            self._setup_figure(ctx)
        self._draw_graph(ctx)
        self._fig.canvas.draw_idle()

    def _draw_transition_arrow(self, ax, s_from, s_to, prob, color,
                               half=_BOX_HALF, rad=-0.6, label_margin=0.20):
        # Larger label_margin than the base (0.05) so probability labels
        # stay clear of the arrow arcs now that the figure is shorter
        # without the sidebar.
        super()._draw_transition_arrow(
            ax, s_from, s_to, prob, color,
            half=half, rad=rad, label_margin=label_margin,
        )


class MdpStepRenderer:
    """One figure per session; render(ctx) returns PNG bytes."""

    def __init__(self):
        self.visualizer = _HeadlessNumberLineVisualizer()

    def render(self, ctx: ValueIterationContext) -> bytes:
        with _mpl_lock:
            self.visualizer.draw_step(ctx)
            buf = io.BytesIO()
            self.visualizer._fig.savefig(buf, format="png")
        buf.seek(0)
        return buf.getvalue()

    def close(self):
        with _mpl_lock:
            if self.visualizer._fig is not None:
                plt.close(self.visualizer._fig)
                self.visualizer._fig = None
