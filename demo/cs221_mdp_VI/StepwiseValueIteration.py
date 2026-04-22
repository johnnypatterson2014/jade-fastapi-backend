"""Stepwise value iteration with matplotlib visualization.

Design mirrors ``cs221_search/StepwiseUCS.py``:

- ``ValueIterationContext``  : dataclass capturing one step's algorithm snapshot
- ``NumberLineVisualizer``   : renders the main graph + sidebar tables
- ``StepwiseValueIteration`` : driver that runs the algorithm and calls the
                               visualizer after every sweep (or every state
                               update, when ``state_level=True``)

Usage
-----
    from ValueIteration import build_number_line_succAndRewardProb
    from MDP import NumberLineMDP
    from StepwiseValueIteration import StepwiseValueIteration

    mdp = NumberLineMDP()
    succAndRewardProb = build_number_line_succAndRewardProb(mdp)
    svi = StepwiseValueIteration(succAndRewardProb, mdp.discount)
    svi.solve()
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── layout / style constants ──────────────────────────────────────────────────

_A1_COLOR = "#1f6feb"   # blue   — action 1 (80% backward, 20% forward)
_A2_COLOR = "#d97706"   # orange — action 2 (70% backward, 30% forward)

_END_LEFT_FC,  _END_LEFT_EC  = "#dcfce7", "#16a34a"   # green  — left terminal
_END_RIGHT_FC, _END_RIGHT_EC = "#fef3c7", "#d97706"   # gold   — right terminal
_MID_FC,       _MID_EC       = "#ffffff", "#222222"   # white  — non-terminal

_TITLE_H = 0.04
_GAP     = 0.04

# Geometry of each state square.  Reducing _BOX_SIZE below 1.0 widens the
# visible gap between adjacent state boxes (which sit at integer x-positions).
_BOX_SIZE = 0.55                # side length of a state box (was 0.76)
_BOX_HALF = _BOX_SIZE / 2       # half-side, used for box-edge math


def _fmt2(v):
    """Format a number with at most 2 decimal places, stripping trailing zeros.

    Examples
    --------
    _fmt2(0.2)    -> "0.2"
    _fmt2(0.835)  -> "0.84"   (rounded to 2dp)
    _fmt2(8.0)    -> "8"
    _fmt2(-5)     -> "-5"
    _fmt2(11.5)   -> "11.5"
    """
    s = f"{v:.2f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


# ── context dataclass ─────────────────────────────────────────────────────────

@dataclass
class ValueIterationContext:
    """Snapshot of one value-iteration step, consumed by the visualizer."""
    succAndRewardProb: dict
    stateActions:      dict                      # state -> set[action]
    V_old:             dict                      # V before this step
    V_new:             dict                      # V after this step (may be partial in state-level mode)
    Q:                 dict                      # (state, action) -> Q using V_old
    policy:            dict                      # state -> argmax action
    delta_per_state:   dict                      # state -> |V_new[s] - V_old[s]|
    max_delta:         float
    step:              int                       # 1-indexed sweep number
    epsilon:           float
    discount:          float
    converged:         bool = False
    mode:              str  = "sweep"            # "sweep" or "state"
    current_state:     Optional[object] = None   # only meaningful in "state" mode


# ── visualizer ────────────────────────────────────────────────────────────────

class NumberLineVisualizer:
    """Renders one step of value iteration on a NumberLine-shaped MDP."""

    def __init__(self):
        self._fig = None
        self._ax = None
        self._tax_breakdown = None    # single sidebar column (Q-update breakdown for all states)

    @staticmethod
    def _all_states(ctx: ValueIterationContext):
        """All states appearing anywhere (includes terminals as successors)."""
        states = set(ctx.stateActions.keys())
        for trans in ctx.succAndRewardProb.values():
            for nextState, _, _ in trans:
                states.add(nextState)
        return states

    def _setup_figure(self, ctx: ValueIterationContext):
        n_states = len(self._all_states(ctx))
        # Sidebar is narrower (~23 % of figure) so the main diagram gets more room.
        fig_w = max(12, min(n_states * 1.5 + 3, 22))
        self._fig = plt.figure(figsize=(fig_w, 7.0))
        self._fig.subplots_adjust(left=0.04, right=0.73, top=0.90, bottom=0.10)
        self._ax = self._fig.add_subplot(111)

        self._tax_breakdown = self._fig.add_axes([0.75, 0.04, 0.23, 0.92])
        self._tax_breakdown.axis("off")
        plt.ion()

    # ── public entry point ────────────────────────────────────────────────────

    def draw_step(self, ctx: ValueIterationContext):
        if self._fig is None:
            self._setup_figure(ctx)
        self._draw_graph(ctx)
        self._draw_breakdown(ctx)
        self._fig.canvas.draw_idle()

    # ── main graph ────────────────────────────────────────────────────────────

    def _draw_graph(self, ctx: ValueIterationContext):
        ax = self._ax
        ax.clear()

        states    = sorted(self._all_states(ctx))
        terminals = [s for s in states if s not in ctx.stateActions]
        min_s, max_s = min(states), max(states)

        # Derive per-state arrival reward from succAndRewardProb.  For our
        # number-line MDPs (both stochastic and deterministic) the reward
        # depends only on the destination state, so any transition landing
        # on s' is enough to identify that state's reward.
        state_rewards = {}
        for transitions in ctx.succAndRewardProb.values():
            for s_next, _prob, reward in transitions:
                state_rewards.setdefault(s_next, reward)

        # (No baseline line — removed per design tweak.)

        for s in states:
            is_terminal = s not in ctx.stateActions
            if is_terminal and terminals and s == min(terminals):
                fc, ec, lw = _END_LEFT_FC,  _END_LEFT_EC,  3
            elif is_terminal and terminals and s == max(terminals):
                fc, ec, lw = _END_RIGHT_FC, _END_RIGHT_EC, 3
            else:
                fc, ec, lw = _MID_FC, _MID_EC, 2

            # Highlight states whose V changed meaningfully this step
            changed = ctx.delta_per_state.get(s, 0.0) > 1e-9
            if changed:
                ec, lw = "red", 3

            # Square state box, centered on (s, 0).
            ax.add_patch(mpatches.Rectangle(
                (s - _BOX_HALF, -_BOX_HALF), _BOX_SIZE, _BOX_SIZE,
                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3,
            ))

            # State index above the box
            ax.text(s, 0.62, str(s), ha="center", va="bottom",
                    fontsize=11, fontweight="bold", zorder=4)

            # V value inside the box
            V_val = ctx.V_new.get(s, 0.0)
            v_color = ec if not changed else _MID_EC
            ax.text(s, 0, _fmt2(V_val), ha="center", va="center",
                    fontsize=10, fontweight="bold", color=v_color, zorder=4)

            # Arrival reward at the bottom inside the box.
            # Anchored to _BOX_HALF so it stays inside if the box is resized.
            r_val = state_rewards.get(s)
            if r_val is not None:
                ax.text(s, -(_BOX_HALF - 0.075), f"r = {_fmt2(r_val)}",
                        ha="center", va="center",
                        fontsize=7, color="#666", style="italic", zorder=4)

            if not is_terminal:
                # Q values in the top corners (small, color-coded by action).
                # Inset slightly from the corner so the text doesn't kiss the
                # box edge; also anchored to _BOX_HALF.
                q_corner_off = _BOX_HALF - 0.055
                q1 = ctx.Q.get((s, 1))
                q2 = ctx.Q.get((s, 2))
                if q1 is not None:
                    ax.text(s - q_corner_off, q_corner_off, _fmt2(q1),
                            ha="left", va="top", fontsize=7,
                            fontweight="bold", color=_A1_COLOR, zorder=4)
                if q2 is not None:
                    ax.text(s + q_corner_off, q_corner_off, _fmt2(q2),
                            ha="right", va="top", fontsize=7,
                            fontweight="bold", color=_A2_COLOR, zorder=4)

                # Greedy policy badge below non-terminals
                pi_s = ctx.policy.get(s)
                pi_color = _A1_COLOR if pi_s == 1 else _A2_COLOR
                ax.text(s, -0.55, f"π = a{pi_s}",
                        ha="center", va="top", fontsize=10, color=pi_color,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  fc="white", ec=pi_color, linewidth=1.2),
                        zorder=4)
            else:
                # "end" badge below terminals — same position/styling as the
                # policy badge for non-terminals, color-matched to the box.
                ax.text(s, -0.55, "end",
                        ha="center", va="top", fontsize=10, color=ec,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  fc="white", ec=ec, linewidth=1.2),
                        zorder=4)

            # Δ label just below the state box when V changed this sweep
            if changed:
                ax.text(s, -(_BOX_HALF + 0.04),
                        f"Δ={_fmt2(ctx.delta_per_state[s])}",
                        ha="center", va="top", fontsize=9,
                        color="red", zorder=4)

        # Transition probability arrows for each non-terminal state's greedy
        # action.  Color matches the policy action (a1 blue / a2 orange).
        # Each arrow's probability label sits on the same side as the arrow's
        # peak (handled inside _draw_transition_arrow).
        # Probabilities are read from succAndRewardProb rather than hardcoded,
        # so this works for any forward/backward split configured on the MDP.
        for s in sorted(ctx.stateActions.keys()):
            pi_s = ctx.policy.get(s)
            if pi_s is None:
                continue
            fwd_prob = 0.0
            bwd_prob = 0.0
            for next_s, prob, _reward in ctx.succAndRewardProb[(s, pi_s)]:
                if next_s > s:
                    fwd_prob = prob
                elif next_s < s:
                    bwd_prob = prob
            color = _A1_COLOR if pi_s == 1 else _A2_COLOR
            self._draw_transition_arrow(ax, s, s + 1, fwd_prob, color)
            self._draw_transition_arrow(ax, s, s - 1, bwd_prob, color)

        # State-level mode: ring the state being updated this moment
        if ctx.mode == "state" and ctx.current_state is not None:
            ax.plot(ctx.current_state, 0, "o", color="red", markersize=38,
                    markerfacecolor="none", markeredgewidth=2.5, zorder=6)

        # Title — encodes iter / epsilon / gamma in a single line.
        # epsilon prints raw (no formatting) so values like 0.001 / 0.005
        # show with their full precision; gamma still uses _fmt2 since it's
        # often a clean value like 1.0 that benefits from trailing-zero strip.
        title = (
            f"Value Iteration  "
            f"[iter: {ctx.step}, "
            f"epsilon: {ctx.epsilon}, "
            f"gamma: {_fmt2(ctx.discount)}]"
        )
        ax.set_title(title, fontsize=12, pad=10)

        # Converged! badge under the row of policy badges, centered.
        if ctx.converged:
            center_x = (min_s + max_s) / 2
            ax.text(center_x, -0.95, "Converged!",
                    ha="center", va="top", fontsize=12,
                    color="#15803d", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.45",
                              fc="#dcfce7", ec="#16a34a", linewidth=1.8),
                    zorder=5)

        ax.set_xlim(min_s - 1, max_s + 1)
        ax.set_ylim(-1.6, 1.6)
        ax.set_xticks(states)
        ax.set_yticks([])
        ax.set_xlabel("state  s")
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

        # Legend (policy color key) — terminals are now labeled on-graph
        # via the "end" badge, so they're omitted here.
        legend = [
            mpatches.Patch(color=_A1_COLOR, label="π(s) = a1"),
            mpatches.Patch(color=_A2_COLOR, label="π(s) = a2"),
        ]
        ax.legend(handles=legend, loc="upper left", fontsize=8,
                  framealpha=0.9, bbox_to_anchor=(0.0, 1.0))

    # ── transition arrow helper ───────────────────────────────────────────────

    def _draw_transition_arrow(self, ax, s_from, s_to, prob, color,
                               half=_BOX_HALF, rad=-0.6, label_margin=0.05):
        """Draw a curved arrow from s_from to s_to labeled with its probability.

        Each label sits on the SAME side as its arrow's peak — just past the
        peak, away from the chord — so the label is always visually attached
        to the arrow it describes.

        For matplotlib's arc3 connection style, the control point is at
        ``cy = -rad * dx`` (with a horizontal chord), and the Bezier peak is
        at ``cy / 2``.  Computing this signed value tells us exactly which
        side of the chord the curve bulges to, so the label can follow.
        """
        if s_to > s_from:
            start = (s_from + half, 0.0)
            end   = (s_to   - half, 0.0)
        else:
            start = (s_from - half, 0.0)
            end   = (s_to   + half, 0.0)

        arrow = mpatches.FancyArrowPatch(
            start, end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="->,head_length=6,head_width=4",
            color=color,
            linewidth=1.6,
            zorder=2,
        )
        ax.add_patch(arrow)

        # Signed peak y of matplotlib's arc3 Bezier (positive = above chord).
        dx     = end[0] - start[0]
        peak_y = -rad * dx / 2

        # Place the label just past the peak, on the same side as the curve.
        if peak_y >= 0:
            label_y = peak_y + label_margin
        else:
            label_y = peak_y - label_margin

        mx = (start[0] + end[0]) / 2
        ax.text(mx, label_y, _fmt2(prob),
                ha="center", va="center", fontsize=9,
                color=color, fontweight="bold", zorder=4)

    # ── per-state Bellman update breakdown (right sidebar) ────────────────────

    def _draw_breakdown(self, ctx: ValueIterationContext):
        """Render the step-by-step Q/V update arithmetic for ALL non-terminal
        states, stacked vertically in a single column.

        Each state's breakdown contains, in order:
            state X                                         (blue header)
              action 1
                forward:  prob * (reward + γ * V[s']) = ...
                backward: prob * (reward + γ * V[s']) = ...
                sum: Q(X, 1)
              action 2                                      (same shape)
                ...
              new value: V[X] = max(Q1, Q2) = ...           (pale-yellow row)
        """
        tax = self._tax_breakdown
        tax.clear()
        tax.axis("off")

        non_terminals = sorted(ctx.stateActions.keys())
        if not non_terminals:
            return

        discount = ctx.discount
        V_old    = ctx.V_old

        # Build all rows for all non-terminal states + remember section starts
        rows           = []
        section_starts = []

        for target in non_terminals:
            # State header
            section_starts.append(len(rows))
            rows.append([f"state {target}"])

            actions  = sorted(ctx.stateActions[target])
            q_values = []
            for a in actions:
                section_starts.append(len(rows))
                rows.append([f"action {a}"])

                for next_s, prob, reward in ctx.succAndRewardProb[(target, a)]:
                    v_next = V_old.get(next_s, 0.0)
                    term   = prob * (reward + discount * v_next)
                    label  = "forward" if next_s > target else "backward"
                    rows.append([
                        f"{label}: {_fmt2(prob)} * "
                        f"({_fmt2(reward)} + {_fmt2(discount)} * V[{next_s}]) "
                        f"= {_fmt2(term)}"
                    ])

                q = ctx.Q.get((target, a), 0.0)
                rows.append([f"sum: {_fmt2(q)}"])
                q_values.append(q)

            if q_values:
                section_starts.append(len(rows))
                max_q     = max(q_values)
                terms_str = ", ".join(_fmt2(q) for q in q_values)
                rows.append([
                    f"new value: V[{target}] = max({terms_str}) = {_fmt2(max_q)}"
                ])

        # Title bar
        cursor_top = 0.97
        tax.text(0.03, cursor_top, "Q-value breakdown",
                 fontsize=11, fontweight="bold",
                 transform=tax.transAxes, va="top", ha="left")
        table_top = cursor_top - _TITLE_H

        n_rows = len(rows)
        if n_rows == 0:
            return

        # Adaptive row height + font size so all states fit in one column.
        available = table_top - 0.005
        rh        = min(0.040, available / n_rows)
        h_total   = n_rows * rh
        # Shrink the font when rows get really tight.
        fontsize  = 8 if rh >= 0.030 else (7 if rh >= 0.022 else 6)

        tbl = tax.table(
            cellText = rows,
            bbox     = [0, table_top - h_total, 1, h_total],
            cellLoc  = "left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(fontsize)

        # Per-cell styling: borders between sections only; facecolor by row type.
        for i in range(n_rows):
            cell  = tbl[i, 0]
            edges = "LR"                                    # always L+R
            if i in section_starts:
                edges += "T"                                # divider at section start
            if i == n_rows - 1:
                edges += "B"                                # close the table
            cell.visible_edges = edges
            cell.set_edgecolor("black")
            cell.set_linewidth(0.7)

            text = rows[i][0]
            if text.startswith("state "):
                cell.set_facecolor("#cce5ff")               # blue state header
                cell.set_text_props(fontweight="bold")
            elif text.startswith("action "):
                cell.set_facecolor("#f0f0f0")               # grey action sub-header
                cell.set_text_props(fontweight="bold")
            elif text.startswith("sum:"):
                cell.set_facecolor("white")
                cell.set_text_props(fontweight="bold")
            elif text.startswith("new value"):
                cell.set_facecolor("#fffacd")               # pale yellow emphasis
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor("white")


# ── stepwise value iteration driver ───────────────────────────────────────────

class StepwiseValueIteration:
    """
    Value iteration with per-step visualization.

    Parameters
    ----------
    succAndRewardProb : dict ((state, action) -> list of (nextState, prob, reward))
    discount, epsilon : float
    visualizer        : object with .draw_step(ctx) (defaults to NumberLineVisualizer)
    auto_step         : bool   — advance without Enter if True
    step_delay        : float  — seconds between auto steps
    state_level       : bool   — if True, pause after each state update within a sweep
    """

    def __init__(self, succAndRewardProb, discount, epsilon=0.001,
                 visualizer=None, auto_step=False, step_delay=0.8,
                 state_level=False):
        self.succAndRewardProb = succAndRewardProb
        self.discount          = discount
        self.epsilon           = epsilon
        self.visualizer        = visualizer or NumberLineVisualizer()
        self.auto_step         = auto_step
        self.step_delay        = step_delay
        self.state_level       = state_level

        self.stateActions = defaultdict(set)
        for state, action in succAndRewardProb.keys():
            self.stateActions[state].add(action)

        self.V = defaultdict(float)
        self.pi = {}
        self.numIters = 0

    # ── algorithm helpers ─────────────────────────────────────────────────────

    def _computeQ(self, V, state, action):
        return sum(prob * (reward + self.discount * V[nextState])
                   for nextState, prob, reward
                   in self.succAndRewardProb[(state, action)])

    def _computeAllQ(self, V):
        return {(s, a): self._computeQ(V, s, a)
                for s, actions in self.stateActions.items()
                for a in actions}

    def _greedy_policy(self, Q):
        return {s: max(actions, key=lambda a: Q[(s, a)])
                for s, actions in self.stateActions.items()}

    def _pause(self):
        if self.auto_step:
            plt.pause(self.step_delay)
        else:
            plt.pause(0.05)
            input("  [Enter] next step, [Ctrl-C] quit ")

    # ── driver ────────────────────────────────────────────────────────────────

    def solve(self):
        print("Running StepwiseValueIteration...")
        while True:
            self.numIters += 1
            V_old  = dict(self.V)
            Q      = self._computeAllQ(self.V)
            policy = self._greedy_policy(Q)

            if self.state_level:
                # Per-state sub-steps: update one state at a time, redraw each time
                V_working = defaultdict(float, self.V)
                for state in sorted(self.stateActions.keys()):
                    actions = self.stateActions[state]
                    V_working[state] = max(Q[(state, a)] for a in actions)
                    delta = {s: abs(V_working[s] - V_old.get(s, 0.0))
                             for s in self.stateActions}
                    ctx = ValueIterationContext(
                        succAndRewardProb = self.succAndRewardProb,
                        stateActions      = dict(self.stateActions),
                        V_old             = V_old,
                        V_new             = dict(V_working),
                        Q                 = Q,
                        policy            = policy,
                        delta_per_state   = delta,
                        max_delta         = max(delta.values()) if delta else 0.0,
                        step              = self.numIters,
                        epsilon           = self.epsilon,
                        discount          = self.discount,
                        converged         = False,
                        mode              = "state",
                        current_state     = state,
                    )
                    self.visualizer.draw_step(ctx)
                    self._pause()
                newV = defaultdict(float, V_working)
            else:
                # One full synchronous sweep
                newV = defaultdict(float)
                for state, actions in self.stateActions.items():
                    newV[state] = max(Q[(state, a)] for a in actions)

            delta_per_state = {s: abs(newV[s] - V_old.get(s, 0.0))
                               for s in self.stateActions}
            max_delta = max(delta_per_state.values()) if delta_per_state else 0.0
            converged = max_delta < self.epsilon

            ctx = ValueIterationContext(
                succAndRewardProb = self.succAndRewardProb,
                stateActions      = dict(self.stateActions),
                V_old             = V_old,
                V_new             = dict(newV),
                Q                 = Q,
                policy            = policy,
                delta_per_state   = delta_per_state,
                max_delta         = max_delta,
                step              = self.numIters,
                epsilon           = self.epsilon,
                discount          = self.discount,
                converged         = converged,
                mode              = "sweep",
            )
            self.visualizer.draw_step(ctx)

            self.V  = newV
            self.pi = policy

            if converged:
                print(f"Converged after {self.numIters} iterations.")
                plt.ioff()
                plt.show()
                return self.V, self.pi

            self._pause()
