"""MDP abstractions and the number-line MDP used in XCS221 Problem 1 / 3a."""

import random
from typing import Any, List, Tuple, Union


# ── type aliases ──────────────────────────────────────────────────────────────

StateT  = Union[int, float, Tuple[Union[float, int]]]
ActionT = Any


# ── abstract MDP ──────────────────────────────────────────────────────────────

class MDP:
    """Abstract base class for a Markov Decision Process."""

    # Return the start state.
    def startState(self):
        raise NotImplementedError("Override me")

    # Property holding the set of possible actions at each state.
    @property
    def actions(self) -> List[ActionT]:
        raise NotImplementedError("Override me")

    # Property holding the discount factor.
    @property
    def discount(self):
        raise NotImplementedError("Override me")

    # Property holding the maximum number of steps for simulation.
    @property
    def timeLimit(self) -> int:
        raise NotImplementedError("Override me")

    # Transition the MDP by taking an action.
    def transition(self, action):
        raise NotImplementedError("Override me")


# ── concrete MDP: number line ─────────────────────────────────────────────────

class NumberLineMDP(MDP):
    """
    Stochastic random walk on the integers [-n, n] with two actions, each
    parameterized by its own forward probability:

    - Action 1: forward with prob ``forward_prob_a1``  (default 0.2)
                backward with prob 1 - forward_prob_a1 (default 0.8)
    - Action 2: forward with prob ``forward_prob_a2``  (default 0.3)
                backward with prob 1 - forward_prob_a2 (default 0.7)

    Reward +leftReward on reaching state -n, +rightReward on +n, else penalty.
    """

    def __init__(self,
                 leftReward:      float = 10,
                 rightReward:     float = 50,
                 penalty:         float = -5,
                 n:               int   = 2,
                 forward_prob_a1: float = 0.2,
                 forward_prob_a2: float = 0.3):
        assert 0.0 <= forward_prob_a1 <= 1.0, "forward_prob_a1 must be in [0, 1]"
        assert 0.0 <= forward_prob_a2 <= 1.0, "forward_prob_a2 must be in [0, 1]"
        self.leftReward      = leftReward
        self.rightReward     = rightReward
        self.penalty         = penalty
        self.n               = n
        self.forward_prob_a1 = forward_prob_a1
        self.forward_prob_a2 = forward_prob_a2
        self.terminalStates  = {-n, n}

    def startState(self):
        self.state = 0
        return self.state

    @property
    def actions(self):
        return [1, 2]

    def transition(self, action) -> Tuple[StateT, float, bool]:
        assert self.state not in self.terminalStates, \
            "Attempting to call transition on a terminated MDP."

        if action == 1:
            forward_prob = self.forward_prob_a1
        elif action == 2:
            forward_prob = self.forward_prob_a2
        else:
            raise ValueError("Invalid Action Provided.")

        if random.random() < forward_prob:
            self.state += 1
        else:
            self.state -= 1

        if self.state == self.n:
            reward = self.rightReward
        elif self.state == -self.n:
            reward = self.leftReward
        else:
            reward = self.penalty

        terminal = self.state in self.terminalStates
        return (self.state, reward, terminal)

    @property
    def discount(self):
        return 1.0


# ── concrete MDP: deterministic number line ──────────────────────────────────

class DeterministicNumberLineMDP(MDP):
    """
    Deterministic number-line MDP.

    States are integers ``0, 1, ..., num_states - 1``.  Terminals are the two
    ends: ``0`` (left) and ``num_states - 1`` (right).

    Two actions, both deterministic (probability 1.0 on the single successor):
      - Action 1 = "left"  : s → s - 1
      - Action 2 = "right" : s → s + 1

    Reward depends only on the state arrived in:
      - Landing on state 0                   → ``terminal_left_reward``
      - Landing on state ``num_states - 1``  → ``terminal_right_reward``
      - Landing on any non-terminal          → ``each_step_reward``
    """

    def __init__(self,
                 num_states: int,
                 terminal_left_reward: float,
                 terminal_right_reward: float,
                 each_step_reward: float,
                 discount_factor: float):
        assert num_states >= 3, "Need at least 3 states (two terminals plus one middle)."
        self.num_states            = num_states
        self.terminal_left_reward  = terminal_left_reward
        self.terminal_right_reward = terminal_right_reward
        self.each_step_reward      = each_step_reward
        self._discount             = discount_factor
        self.terminalStates        = {0, num_states - 1}

    def startState(self):
        # Start in (roughly) the middle.
        self.state = (self.num_states - 1) // 2
        return self.state

    @property
    def actions(self):
        return [1, 2]   # 1 = left, 2 = right

    def transition(self, action) -> Tuple[StateT, float, bool]:
        assert self.state not in self.terminalStates, \
            "Attempting to call transition on a terminated MDP."

        if action == 1:          # left
            self.state -= 1
        elif action == 2:        # right
            self.state += 1
        else:
            raise ValueError("Invalid Action Provided.")

        if self.state == 0:
            reward = self.terminal_left_reward
        elif self.state == self.num_states - 1:
            reward = self.terminal_right_reward
        else:
            reward = self.each_step_reward

        terminal = self.state in self.terminalStates
        return (self.state, reward, terminal)

    @property
    def discount(self):
        return self._discount


# ── concrete MDP: deterministic with state-based reward semantics ─────────────

class StateRewardNumberLineMDP(DeterministicNumberLineMDP):
    """
    Deterministic number-line MDP with STATE-BASED reward semantics.

    This class has the same attributes and simulator behavior as
    ``DeterministicNumberLineMDP`` — the difference is purely in how the
    Bellman equation is set up for value iteration (handled by the matching
    builder in ValueIteration.py):

      - Under ``DeterministicNumberLineMDP``, the terminal reward is received
        on the TRANSITION into the terminal, and V(terminal) = 0.  Example:
        V(1) = 100 (one transition left earns +100 immediately).

      - Under ``StateRewardNumberLineMDP``, the reward is intrinsic to the
        state, and V(terminal) = reward(terminal).  V(s) represents the
        total discounted return starting from s, with:

            V(terminal)     = terminal_reward
            V(non-terminal) = each_step_reward + γ · V(successor)

        So V(1) = 0 + γ · 100 = 50 (one γ-discount step away from the +100).

    Choose whichever semantic fits your intuition; the policies are typically
    the same up to a single discount factor on V.
    """
    pass
