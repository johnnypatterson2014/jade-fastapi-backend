"""Plain value-iteration algorithm (no visualization).

Same implementation as ``src/submission.py`` Problem 3a — kept separate from
``StepwiseValueIteration.py`` so that running the grader-style algorithm does
not pull in matplotlib.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from MDP import (
    ActionT,
    DeterministicNumberLineMDP,
    NumberLineMDP,
    StateRewardNumberLineMDP,
    StateT,
)


# ── core algorithm ────────────────────────────────────────────────────────────

def valueIteration_full(
    succAndRewardProb: Dict[Tuple[StateT, ActionT], List[Tuple[StateT, float, float]]],
    discount: float,
    epsilon: float = 0.001,
    initialV: Optional[Dict[StateT, float]] = None,
    verbose: bool = True,
) -> Tuple[Dict[StateT, float], Dict[StateT, ActionT], int]:
    """
    Classical synchronous value iteration — returns V, pi, and iteration count.

    Parameters
    ----------
    succAndRewardProb : dict ((state, action) -> list of (nextState, prob, reward))
    discount          : float — MDP discount factor
    epsilon           : float — convergence threshold (max-norm change)
    initialV          : optional dict state -> float that seeds the value
                        function.  Seeded values for states NOT in
                        ``stateActions`` (i.e., terminals) are preserved
                        across every sweep — they are never overwritten by
                        the Bellman update.  Use this to encode a
                        state-based reward semantic where V(terminal) equals
                        the terminal's reward rather than 0.
    verbose           : bool  — if True, print progress to stdout

    Returns
    -------
    (V, pi, numIters)
      V        : dict state -> float — converged value function
      pi       : dict state -> action — greedy policy extracted from V
      numIters : int — number of Bellman sweeps performed
    """
    # States that actually have actions (excludes terminals).
    stateActions = defaultdict(set)
    for state, action in succAndRewardProb.keys():
        stateActions[state].add(action)

    def computeQ(V: Dict[StateT, float], state: StateT, action: ActionT) -> float:
        # Q(s,a) = sum_{s'} T(s,a,s') * [ R(s,a,s') + gamma * V(s') ]
        return sum(
            prob * (reward + discount * V[nextState])
            for nextState, prob, reward in succAndRewardProb[(state, action)]
        )

    def computePolicy(V: Dict[StateT, float]) -> Dict[StateT, ActionT]:
        # pi(s) = argmax_a Q(V, s, a), for every state that has actions.
        return {state: max(actions, key=lambda a: computeQ(V, state, a))
                for state, actions in stateActions.items()}

    def seed(V_dict: defaultdict) -> None:
        """Apply initialV onto V_dict (idempotent)."""
        if initialV is not None:
            V_dict.update(initialV)

    if verbose:
        print("Running valueIteration...")

    V = defaultdict(float)     # terminals default to 0 unless seeded
    seed(V)
    numIters = 0
    while True:
        newV = defaultdict(float)
        seed(newV)              # preserve seeded terminals across sweeps
        # Bellman optimality update over every non-terminal state
        for state, actions in stateActions.items():
            newV[state] = max(computeQ(V, state, a) for a in actions)
        # Converged: stop when the max change is below epsilon
        if all(abs(newV[s] - V[s]) < epsilon for s in stateActions):
            V = newV
            numIters += 1
            break
        V = newV
        numIters += 1

    if verbose:
        print("valueIteration: %d iterations" % numIters)

    return dict(V), computePolicy(V), numIters


def valueIteration(
    succAndRewardProb: Dict[Tuple[StateT, ActionT], List[Tuple[StateT, float, float]]],
    discount: float,
    epsilon: float = 0.001,
) -> Dict[StateT, ActionT]:
    """
    Classical synchronous value iteration — returns only the policy.

    Thin wrapper around ``valueIteration_full`` preserving the grader-style
    signature (dict -> dict).
    """
    _, pi, _ = valueIteration_full(succAndRewardProb, discount, epsilon)
    return pi


# ── succAndRewardProb builders ────────────────────────────────────────────────

def build_number_line_succAndRewardProb(mdp: NumberLineMDP):
    """Build the succAndRewardProb dict for the stochastic NumberLineMDP.

    Uses the forward probabilities configured on the MDP
    (``mdp.forward_prob_a1`` and ``mdp.forward_prob_a2``), so callers can
    instantiate asymmetric walks without touching this builder.

    Terminal states (-n, n) do NOT appear as keys — only as successor states.
    """
    p1  = mdp.forward_prob_a1
    p2  = mdp.forward_prob_a2
    q1  = 1.0 - p1           # backward prob for action 1
    q2  = 1.0 - p2           # backward prob for action 2

    succAndRewardProb = {
        (-mdp.n + 1, 1): [(-mdp.n + 2, p1, mdp.penalty), (-mdp.n, q1, mdp.leftReward)],
        (-mdp.n + 1, 2): [(-mdp.n + 2, p2, mdp.penalty), (-mdp.n, q2, mdp.leftReward)],
        (mdp.n - 1,  1): [(mdp.n - 2,  q1, mdp.penalty), (mdp.n,  p1, mdp.rightReward)],
        (mdp.n - 1,  2): [(mdp.n - 2,  q2, mdp.penalty), (mdp.n,  p2, mdp.rightReward)],
    }
    for s in range(-mdp.n + 2, mdp.n - 1):
        succAndRewardProb[(s, 1)] = [(s + 1, p1, mdp.penalty), (s - 1, q1, mdp.penalty)]
        succAndRewardProb[(s, 2)] = [(s + 1, p2, mdp.penalty), (s - 1, q2, mdp.penalty)]
    return succAndRewardProb


def build_deterministic_number_line_succAndRewardProb(mdp: DeterministicNumberLineMDP):
    """Build the succAndRewardProb dict for a DeterministicNumberLineMDP.

    Each (state, action) maps to a single (nextState, 1.0, reward) tuple.
    Terminal states (0 and num_states-1) do NOT appear as keys.
    """
    last = mdp.num_states - 1

    def _reward_for_arrival(s_next: int) -> float:
        if s_next == 0:
            return mdp.terminal_left_reward
        if s_next == last:
            return mdp.terminal_right_reward
        return mdp.each_step_reward

    succAndRewardProb = {}
    for s in range(1, last):                 # non-terminal states only
        left_next  = s - 1                   # action 1 → left
        right_next = s + 1                   # action 2 → right
        succAndRewardProb[(s, 1)] = [(left_next,  1.0, _reward_for_arrival(left_next))]
        succAndRewardProb[(s, 2)] = [(right_next, 1.0, _reward_for_arrival(right_next))]
    return succAndRewardProb


# ── convenience wrappers ──────────────────────────────────────────────────────

def run_VI_over_numberLine(
    mdp: NumberLineMDP,
) -> Tuple[Dict[StateT, float], Dict[StateT, ActionT]]:
    """Run plain value iteration on a stochastic NumberLineMDP; return (V, pi).

    Matches the (V, pi) return shape of the other ``run_VI_over_*`` wrappers
    so callers can destructure uniformly.
    """
    succAndRewardProb = build_number_line_succAndRewardProb(mdp)
    V, pi, _ = valueIteration_full(succAndRewardProb, mdp.discount)
    return V, pi


def run_VI_over_deterministic_numberLine(
    mdp: DeterministicNumberLineMDP,
) -> Tuple[Dict[StateT, float], Dict[StateT, ActionT]]:
    """Run value iteration on a DeterministicNumberLineMDP; return (V, pi)."""
    succAndRewardProb = build_deterministic_number_line_succAndRewardProb(mdp)
    V, pi, _ = valueIteration_full(succAndRewardProb, mdp.discount)
    return V, pi


# ── state-reward-semantic builders ────────────────────────────────────────────

def build_state_reward_number_line_succAndRewardProb(mdp: StateRewardNumberLineMDP):
    """Build succAndRewardProb for a StateRewardNumberLineMDP.

    Under the state-reward semantic, the Bellman update is:
        V(s) = each_step_reward + γ · max_a V(successor)
    with V(terminal) seeded to the terminal's reward (see
    ``build_state_reward_initial_V`` below).

    The transition reward therefore is ``each_step_reward`` for EVERY
    transition, regardless of destination — it represents the reward earned
    at the source (non-terminal) state at this timestep.
    """
    last = mdp.num_states - 1
    succAndRewardProb = {}
    for s in range(1, last):                 # non-terminal states only
        succAndRewardProb[(s, 1)] = [(s - 1, 1.0, mdp.each_step_reward)]  # left
        succAndRewardProb[(s, 2)] = [(s + 1, 1.0, mdp.each_step_reward)]  # right
    return succAndRewardProb


def build_state_reward_initial_V(mdp: StateRewardNumberLineMDP) -> Dict[StateT, float]:
    """Seed V with terminal rewards for a StateRewardNumberLineMDP.

    Returns a dict suitable for the ``initialV`` argument of
    ``valueIteration_full``: terminals map to their respective rewards, all
    other states are left untouched (valueIteration will default them to 0).
    """
    return {
        0:                    mdp.terminal_left_reward,
        mdp.num_states - 1:   mdp.terminal_right_reward,
    }


def run_VI_over_state_reward_numberLine(
    mdp: StateRewardNumberLineMDP,
) -> Tuple[Dict[StateT, float], Dict[StateT, ActionT]]:
    """Run value iteration on a StateRewardNumberLineMDP; return (V, pi).

    Transition rewards equal ``each_step_reward``; terminal rewards are
    injected as seeded V(terminal) so that they persist across Bellman
    sweeps.  After convergence, V(terminal) equals the terminal's reward
    and V(s) for non-terminal s equals the total discounted return
    starting from s.
    """
    succAndRewardProb = build_state_reward_number_line_succAndRewardProb(mdp)
    initialV          = build_state_reward_initial_V(mdp)
    V, pi, _ = valueIteration_full(
        succAndRewardProb, mdp.discount, initialV=initialV,
    )
    return V, pi
