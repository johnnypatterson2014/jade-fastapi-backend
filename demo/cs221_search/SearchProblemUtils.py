import heapq
import json
from collections import defaultdict
from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from typing import Dict, Hashable, List, Optional, Set, Tuple

import osmium
from osmium import osm

########################################################################################
# Abstract Interfaces for State, Search Problems, and Search Algorithms.


@dataclass(frozen=True, order=True)
class State:
    """
    A State consists of a string `location` and (possibly null) `memory`.
    Note that `memory` must be a "Hashable" data type -- for example:
        - any non-mutable primitive (str, int, float, etc.)
        - tuples
        - nested combinations of the above

    As you implement different types of search problems throughout the assignment,
    think of what `memory` should contain to enable efficient search!
    """
    location: str
    memory: Optional[Hashable] = None


class SearchProblem:
    # Return the start state.
    def startState(self) -> State:
        raise NotImplementedError("Override me")

    # Return whether `state` is an end state or not.
    def isEnd(self, state: State) -> bool:
        raise NotImplementedError("Override me")

    # Return a list of (action: str, state: State, cost: float) tuples corresponding to
    # the various edges coming out of `state`
    def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        raise NotImplementedError("Override me")


class SearchAlgorithm:
    def __init__(self):
        """
        A SearchAlgorithm is defined by the function `solve(problem: SearchProblem)`

        A call to `solve` sets the following instance variables:
            - self.actions: List of "actions" that takes one from the start state to a
                            valid end state, or None if no such action sequence exists.
                            > Note: For this assignment, an "action" is just the string
                                    "nextLocation" for a state, but in general, an
                                    action could be something like "up/down/left/right"

            - self.pathCost: Sum of the costs along the path, or None if no valid path.

            - self.numStatesExplored: Number of States explored by the given search
                                      algorithm as it attempts to find a satisfying
                                      path. You can use this to gauge the efficiency of
                                      search heuristics, for example.

            - self.pastCosts: Dictionary mapping each State location visited by the
                              SearchAlgorithm to the corresponding cost to get there
                              from the starting location.
        """
        self.actions: List[str] = None
        self.pathCost: float = None
        self.numStatesExplored: int = 0
        self.pastCosts: Dict[str, float] = {}

    def solve(self, problem: SearchProblem) -> None:
        raise NotImplementedError("Override me")


class Heuristic:
    # A Heuristic object is defined by a single function `evaluate(state)` that
    # returns an estimate of the cost of going from the specified `state` to an
    # end state. Used by A*.
    def evaluate(self, state: State) -> float:
        raise NotImplementedError("Override me")


# Data structure for supporting uniform cost search.
class PriorityQueue:
    def __init__(self):
        self.DONE = -100000
        self.heap = []
        self.priorities = {}  # Map from state to priority

    # Insert `state` into the heap with priority `newPriority` if `state` isn't in
    # the heap or `newPriority` is smaller than the existing priority.
    #   > Return whether the priority queue was updated.
    def update(self, state: State, newPriority: float) -> bool:
        oldPriority = self.priorities.get(state)
        if oldPriority is None or newPriority < oldPriority:
            self.priorities[state] = newPriority
            heapq.heappush(self.heap, (newPriority, state))
            return True
        return False

    # Returns (state with minimum priority, priority) or (None, None) if empty.
    def removeMin(self):
        while len(self.heap) > 0:
            priority, state = heapq.heappop(self.heap)
            if self.priorities[state] == self.DONE:
                continue
            self.priorities[state] = self.DONE
            return state, priority
        return None, None

