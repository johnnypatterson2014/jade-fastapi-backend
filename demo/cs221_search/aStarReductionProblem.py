
from SearchProblemUtils import (
    Heuristic,
    SearchProblem,
    State,
)

from MapUtils import (
    CityMap,
    checkValid,
    extractPath,
    createGridMap,
    getTotalCost,
    makeGridLabel,
    makeTag,
)
from StepwiseUCS import StepwiseUCS
from typing import List, Tuple
import json


########################################################################################
# Problem 4a: A* to UCS reduction

def aStarReduction(problem: SearchProblem, heuristic: Heuristic) -> SearchProblem:
    class NewSearchProblem(SearchProblem):
        def startState(self) -> State:
            # ### START CODE HERE ###
            return problem.startState()
            # ### END CODE HERE ###

        def isEnd(self, state: State) -> bool:
            # ### START CODE HERE ###
            return problem.isEnd(state)
            # ### END CODE HERE ###

        def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
            # ### START CODE HERE ###
            # Modify each edge cost: c' = c + h(s') - h(s)
            # This makes UCS priority = pastCost(s') + h(s') = A* priority
            return [
                (action, newState, cost + heuristic.evaluate(newState) - heuristic.evaluate(state))
                for action, newState, cost in problem.successorsAndCosts(state)
            ]
            # ### END CODE HERE ###

        @property
        def waypointTags(self):
            # Forward waypointTags from the wrapped problem so that WaypointsVisualizer
            # can display the waypoints table even when the problem is wrapped in A*.
            return getattr(problem, "waypointTags", None)

    return NewSearchProblem()

