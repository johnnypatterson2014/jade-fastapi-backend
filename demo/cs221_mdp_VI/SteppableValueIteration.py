"""Generator form of value iteration. Yields one ValueIterationContext per step."""

from collections import defaultdict
from typing import Generator, List, Dict, Any

from StepwiseValueIteration import ValueIterationContext


def compute_breakdown(ctx: ValueIterationContext) -> List[Dict[str, Any]]:
    """
    Return JSON-ready Q-value breakdown data for each non-terminal state.
    Mirrors the arithmetic shown in NumberLineVisualizer._draw_breakdown.
    """
    result: List[Dict[str, Any]] = []
    for state in sorted(ctx.stateActions.keys()):
        actions_data: List[Dict[str, Any]] = []
        q_values: List[float] = []
        for action in sorted(ctx.stateActions[state]):
            terms: List[Dict[str, Any]] = []
            for next_s, prob, reward in ctx.succAndRewardProb[(state, action)]:
                v_next = ctx.V_old.get(next_s, 0.0)
                term_value = prob * (reward + ctx.discount * v_next)
                direction = "forward" if next_s > state else "backward"
                terms.append({
                    "direction": direction,
                    "prob": prob,
                    "reward": reward,
                    "next_state": next_s,
                    "v_next": v_next,
                    "value": term_value,
                })
            q = ctx.Q.get((state, action), 0.0)
            actions_data.append({"action": action, "terms": terms, "sum": q})
            q_values.append(q)

        result.append({
            "state": state,
            "actions": actions_data,
            "q_values": q_values,
            "new_value": max(q_values) if q_values else 0.0,
        })
    return result


def run_vi_stepwise(
    succAndRewardProb,
    discount: float,
    epsilon: float = 0.001,
    state_level: bool = False,
) -> Generator[ValueIterationContext, None, None]:
    """
    Yield one ValueIterationContext per sweep (or per state update when
    state_level=True). Terminates after emitting the converged context.
    """
    stateActions = defaultdict(set)
    for state, action in succAndRewardProb.keys():
        stateActions[state].add(action)

    V = defaultdict(float)
    numIters = 0

    def computeQ(V_map, state, action):
        return sum(
            prob * (reward + discount * V_map[nextState])
            for nextState, prob, reward in succAndRewardProb[(state, action)]
        )

    def computeAllQ(V_map):
        return {
            (s, a): computeQ(V_map, s, a)
            for s, actions in stateActions.items()
            for a in actions
        }

    def greedy_policy(Q):
        return {
            s: max(actions, key=lambda a: Q[(s, a)])
            for s, actions in stateActions.items()
        }

    while True:
        numIters += 1
        V_old = dict(V)
        Q = computeAllQ(V)
        policy = greedy_policy(Q)

        if state_level:
            V_working = defaultdict(float, V)
            for state in sorted(stateActions.keys()):
                actions = stateActions[state]
                V_working[state] = max(Q[(state, a)] for a in actions)
                delta = {
                    s: abs(V_working[s] - V_old.get(s, 0.0))
                    for s in stateActions
                }
                yield ValueIterationContext(
                    succAndRewardProb=succAndRewardProb,
                    stateActions=dict(stateActions),
                    V_old=V_old,
                    V_new=dict(V_working),
                    Q=Q,
                    policy=policy,
                    delta_per_state=delta,
                    max_delta=max(delta.values()) if delta else 0.0,
                    step=numIters,
                    epsilon=epsilon,
                    discount=discount,
                    converged=False,
                    mode="state",
                    current_state=state,
                )
            newV = defaultdict(float, V_working)
        else:
            newV = defaultdict(float)
            for state, actions in stateActions.items():
                newV[state] = max(Q[(state, a)] for a in actions)

        delta_per_state = {
            s: abs(newV[s] - V_old.get(s, 0.0)) for s in stateActions
        }
        max_delta = max(delta_per_state.values()) if delta_per_state else 0.0
        converged = max_delta < epsilon

        yield ValueIterationContext(
            succAndRewardProb=succAndRewardProb,
            stateActions=dict(stateActions),
            V_old=V_old,
            V_new=dict(newV),
            Q=Q,
            policy=policy,
            delta_per_state=delta_per_state,
            max_delta=max_delta,
            step=numIters,
            epsilon=epsilon,
            discount=discount,
            converged=converged,
            mode="sweep",
        )

        V = newV
        if converged:
            return
