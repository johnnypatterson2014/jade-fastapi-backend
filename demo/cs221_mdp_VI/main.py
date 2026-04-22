"""Entry point for cs221_mdp_VI.

Runs either plain value iteration (``test_1`` / ``test_2``) or the interactive
stepwise visualizer (``visualize_test_1`` / ``visualize_test_2``).

Usage
-----
    # from the src/ directory:
    python -m cs221_mdp_VI.main

    # or directly:
    cd src/cs221_mdp_VI
    python main.py
"""

import os
import sys

# Make the package's own modules importable regardless of where python is
# launched from (matches the pattern used by cs221_search/main.py).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json

from MDP import (
    DeterministicNumberLineMDP,
    NumberLineMDP,
    StateRewardNumberLineMDP,
)
from ValueIteration import (
    build_deterministic_number_line_succAndRewardProb,
    build_number_line_succAndRewardProb,
    build_state_reward_number_line_succAndRewardProb,
    run_VI_over_deterministic_numberLine,
    run_VI_over_numberLine,
    run_VI_over_state_reward_numberLine,
    valueIteration,
)
from StepwiseValueIteration import (
    NumberLineVisualizer,
    StepwiseValueIteration,
)
from StaticVisualizer import NumberLineStaticVisualizer


# ── helpers ───────────────────────────────────────────────────────────────────

# The gold file lives in src/, one directory up from this package.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GOLD_3A_1 = os.path.join(_SRC_DIR, "3a-1-gold.json")


# ── plain tests (no visualization) ────────────────────────────────────────────

def test_NumberLineMDP():
    """Default 5-state stochastic NumberLineMDP (matches Problem 1)."""
    mdp = NumberLineMDP(10, 50, -5, 2,
                    forward_prob_a1=0.2,
                    forward_prob_a2=0.3)
    V, pi = run_VI_over_numberLine(mdp)
    return V, pi
    


# ── interactive visualizations ────────────────────────────────────────────────

def visualize_test_NumberLineMDP(auto_step: bool = False,
                     step_delay: float = 0.8,
                     state_level: bool = False):
    """Interactive visualization of value iteration on the default 5-state MDP.

    Parameters
    ----------
    auto_step   : bool  — advance on a timer instead of waiting for Enter
    step_delay  : float — seconds between auto steps
    state_level : bool  — if True, pause after each state update (not each sweep)
    """
    mdp = NumberLineMDP(10, 50, -5, 2,
                    forward_prob_a1=0.2,
                    forward_prob_a2=0.3)
    succAndRewardProb = build_number_line_succAndRewardProb(mdp)
    svi = StepwiseValueIteration(
        succAndRewardProb,
        mdp.discount,
        visualizer=NumberLineVisualizer(),
        auto_step=auto_step,
        step_delay=step_delay,
        state_level=state_level,
    )
    return svi.solve()



# ── state-reward deterministic MDP test ───────────────────────────────────────

def test_StateRewardNumberLineMDP():
    """Plain run on a 6-state state-reward deterministic NumberLineMDP.

    Same parameters as test_3 but with STATE-BASED reward semantics:
    V(terminal) = terminal_reward (not 0) and V(non-terminal) represents
    the total discounted return starting from that state.

    Expected converged V: {0:100, 1:50, 2:25, 3:12.5, 4:20, 5:40}
    Expected policy π:    {1:1, 2:1, 3:1, 4:2}   (1=left, 2=right)
    Note: state 3 goes LEFT here (differs from test_3 where state 3 also went
    left); state 4 goes RIGHT because 0.5·40 = 20 > 0.5·V(3) = 6.25.
    """
    mdp = StateRewardNumberLineMDP(
        num_states            = 6,
        terminal_left_reward  = 100,
        terminal_right_reward = 50,
        each_step_reward      = 0,
        discount_factor       = 0.6,
    )
    V, pi = run_VI_over_state_reward_numberLine(mdp)
    return V, pi


def visualize_test_StateRewardNumberLineMDP():
    """Static visualization of the state-reward 6-state deterministic MDP."""
    mdp = StateRewardNumberLineMDP(
        num_states            = 6,
        terminal_left_reward  = 100,
        terminal_right_reward = 50,
        each_step_reward      = 0,
        discount_factor       = 0.6,
    )
    V, pi = run_VI_over_state_reward_numberLine(mdp)
    srp = build_state_reward_number_line_succAndRewardProb(mdp)
    NumberLineStaticVisualizer().draw(mdp, V, pi, succAndRewardProb=srp)
    return V, pi


# ── test dispatch ─────────────────────────────────────────────────────────────

# Manual mode (default) — press Enter each step
# visualize_test_1()

# State-by-state mode — pause after each state update within a sweep
# visualize_test_1(state_level=True)

# Auto mode — advances every step_delay seconds without Enter
# visualize_test_1(auto_step=True, step_delay=0.6)

# Larger MDP (auto by default because many sweeps)
# visualize_test_2()

# Deterministic 6-state MDP (transition-reward semantic) — V(terminal) = 0
# visualize_test_3()

# Deterministic 6-state MDP (state-reward semantic) — V(terminal) = terminal_reward
# visualize_test_4()

# Plain non-visual tests
# test_1()
# test_2()
# test_3()
# test_4()


if __name__ == "__main__":
    # V, pi = test_NumberLineMDP()
    V, pi = visualize_test_NumberLineMDP()
    # V, pi = test_StateRewardNumberLineMDP()
    # V, pi = visualize_test_StateRewardNumberLineMDP()

    print(f"Values: {V}. Final policy: {pi}")
