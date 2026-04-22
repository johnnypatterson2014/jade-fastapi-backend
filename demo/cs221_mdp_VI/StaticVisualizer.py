"""Non-interactive (static) visualizer for a deterministic NumberLine MDP.

After value iteration converges, this renders a single matplotlib figure
showing the final V values and the optimal policy.  No interaction, no
stepping — just the end result.

Usage
-----
    from MDP import StateRewardNumberLineMDP
    from ValueIteration import (
        build_state_reward_number_line_succAndRewardProb,
        run_VI_over_state_reward_numberLine,
    )
    from StaticVisualizer import NumberLineStaticVisualizer

    mdp  = StateRewardNumberLineMDP(6, 100, 40, 0, 0.5)
    V, pi = run_VI_over_state_reward_numberLine(mdp)
    srp  = build_state_reward_number_line_succAndRewardProb(mdp)
    NumberLineStaticVisualizer().draw(mdp, V, pi, succAndRewardProb=srp)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from MDP import DeterministicNumberLineMDP


# ── style constants (match StepwiseValueIteration for visual consistency) ─────

_A1_COLOR = "#1f6feb"   # blue   — action 1 = left
_A2_COLOR = "#d97706"   # orange — action 2 = right

_END_LEFT_FC,  _END_LEFT_EC  = "#dcfce7", "#16a34a"   # green
_END_RIGHT_FC, _END_RIGHT_EC = "#fef3c7", "#d97706"   # gold
_MID_FC,       _MID_EC       = "#ffffff", "#222222"   # white

_BOX_SIZE = 0.76   # side length of each state square


# ── visualizer ────────────────────────────────────────────────────────────────

class NumberLineStaticVisualizer:
    """Renders the converged V and π of a DeterministicNumberLineMDP once."""

    def draw(self,
             mdp:               DeterministicNumberLineMDP,
             V:                 dict,
             pi:                dict,
             succAndRewardProb: dict = None,
             show:              bool = True):
        """Create and display the figure.  Returns the Figure object.

        Parameters
        ----------
        mdp               : NumberLineMDP-shaped object with
                            ``num_states, terminal_left_reward,
                            terminal_right_reward, each_step_reward,
                            discount, terminalStates``.
        V, pi             : converged value function and policy.
        succAndRewardProb : optional.  If provided, Q(s, a) is computed from
                            it and shown in the top-left/top-right corner of
                            each non-terminal box.
        show              : call ``plt.show()`` at the end (blocking).
        """
        n      = mdp.num_states
        last   = n - 1
        states = list(range(n))
        gamma  = mdp.discount

        # ── compute Q(s,a) if transition structure was provided ──────────────
        Q = {}
        if succAndRewardProb is not None:
            for (s, a), transitions in succAndRewardProb.items():
                Q[(s, a)] = sum(
                    prob * (reward + gamma * V.get(s_next, 0.0))
                    for s_next, prob, reward in transitions
                )

        # ── figure layout ────────────────────────────────────────────────────
        fig_w = max(9.0, min(n * 1.8, 22.0))
        fig_h = 5.8                       # a bit taller to fit legend at top
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        # Reserve the top ~18 % of the figure for suptitle + legend + ax-title
        fig.subplots_adjust(top=0.80, bottom=0.10, left=0.05, right=0.97)

        # Main title (figure level)
        fig.suptitle("Deterministic Number-Line MDP — Value Iteration Result",
                     fontsize=13, fontweight="bold", y=0.97)

        # Legend as horizontal row below the suptitle
        legend_handles = [
            mpatches.Patch(color=_A1_COLOR,     label="π(s) = left  (action 1)"),
            mpatches.Patch(color=_A2_COLOR,     label="π(s) = right (action 2)"),
            mpatches.Patch(color=_END_LEFT_EC,  label="terminal (left)"),
            mpatches.Patch(color=_END_RIGHT_EC, label="terminal (right)"),
        ]
        fig.legend(handles=legend_handles,
                   loc="upper center",
                   bbox_to_anchor=(0.5, 0.92),
                   ncol=4, fontsize=9, framealpha=0.9)

        # Parameter subtitle (axes level, below legend)
        params = (
            f"num_states = {n}   |   "
            f"terminal_left = +{mdp.terminal_left_reward:g}   |   "
            f"terminal_right = +{mdp.terminal_right_reward:g}   |   "
            f"each_step = {mdp.each_step_reward:+g}   |   "
            f"γ = {gamma}"
        )
        ax.set_title(params, fontsize=10, color="#555", pad=8)

        # ── number-line baseline ─────────────────────────────────────────────
        ax.plot([-0.5, last + 0.5], [0, 0],
                color="lightgray", linewidth=1.0, zorder=1)

        # ── per-state rendering ──────────────────────────────────────────────
        _half = _BOX_SIZE / 2
        for s in states:
            is_terminal = s in mdp.terminalStates
            if is_terminal and s == 0:
                fc, ec, lw = _END_LEFT_FC,  _END_LEFT_EC,  3
            elif is_terminal and s == last:
                fc, ec, lw = _END_RIGHT_FC, _END_RIGHT_EC, 3
            else:
                fc, ec, lw = _MID_FC, _MID_EC, 2

            # Square state box centered on (s, 0)
            ax.add_patch(mpatches.Rectangle(
                (s - _half, -_half), _BOX_SIZE, _BOX_SIZE,
                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3,
            ))

            # State index label, above the box
            ax.text(s, 0.55, str(s), ha="center", va="bottom",
                    fontsize=11, fontweight="bold", zorder=4)

            # V value in the center of the box
            ax.text(s, 0.0, f"{V.get(s, 0.0):.2f}",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color=ec, zorder=4)

            # Reward value at the bottom inside the box
            if s == 0:
                reward = mdp.terminal_left_reward
            elif s == last:
                reward = mdp.terminal_right_reward
            else:
                reward = mdp.each_step_reward
            ax.text(s, -0.27, f"r = {reward:g}",
                    ha="center", va="center",
                    fontsize=7, color="#666", style="italic", zorder=4)

            if is_terminal:
                # "END" tag below terminals
                ax.text(s, -0.70, "END", ha="center", va="top",
                        fontsize=9, fontweight="bold", color=ec, zorder=4)
            else:
                # Q values in the top corners (small, color-coded by action)
                q_left  = Q.get((s, 1))
                q_right = Q.get((s, 2))
                if q_left is not None:
                    ax.text(s - 0.30, 0.30, f"{q_left:g}",
                            ha="left", va="top",
                            fontsize=7, fontweight="bold",
                            color=_A1_COLOR, zorder=4)
                if q_right is not None:
                    ax.text(s + 0.30, 0.30, f"{q_right:g}",
                            ha="right", va="top",
                            fontsize=7, fontweight="bold",
                            color=_A2_COLOR, zorder=4)

                # Policy badge below non-terminal boxes
                pi_s     = pi.get(s)
                pi_color = _A1_COLOR if pi_s == 1 else _A2_COLOR
                pi_label = "π = left" if pi_s == 1 else "π = right"
                ax.text(s, -0.70, pi_label, ha="center", va="top",
                        fontsize=10, fontweight="bold", color=pi_color,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  fc="white", ec=pi_color, linewidth=1.2),
                        zorder=4)

        # ── policy arrows ────────────────────────────────────────────────────
        # Left arrows below the number line, right arrows above — so they
        # never overlap even in the oscillating-pair edge case.
        for s in states:
            if s in mdp.terminalStates:
                continue
            pi_s = pi.get(s)
            if pi_s == 1:       # left
                color = _A1_COLOR
                start = (s - _half,     -0.32)
                end   = (s - 1 + _half, -0.32)
            else:               # right
                color = _A2_COLOR
                start = (s + _half,      0.32)
                end   = (s + 1 - _half,  0.32)

            ax.add_patch(mpatches.FancyArrowPatch(
                start, end,
                arrowstyle="->,head_length=8,head_width=5",
                color=color, linewidth=2.0, zorder=2,
            ))

        # ── axes cosmetics ───────────────────────────────────────────────────
        ax.set_xlim(-1, last + 1)
        ax.set_ylim(-1.3, 1.3)
        ax.set_xticks(states)
        ax.set_yticks([])
        ax.set_xlabel("state  s")
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

        if show:
            plt.show()
        return fig
