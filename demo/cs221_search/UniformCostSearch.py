from typing import Dict, List

from SearchProblemUtils import PriorityQueue, SearchAlgorithm, SearchProblem, State


class UniformCostSearch(SearchAlgorithm):
    def __init__(self, verbose: int = 0):
        super().__init__()
        self.verbose = verbose

    def solve(self, problem: SearchProblem) -> None:
        """
        Run Uniform Cost Search on the specified `problem` instance.

        Sets the following instance variables (see `SearchAlgorithm` docstring).
            - self.actions: List[str]
            - self.pathCost: float
            - self.numStatesExplored: int
            - self.pastCosts: Dict[str, float]
        """
        self.actions: List[str] = None
        self.pathCost: float = None
        self.numStatesExplored: int = 0
        self.pastCosts: Dict[str, float] = {}

        frontier = PriorityQueue()
        backpointers = {}

        startState = problem.startState()
        frontier.update(startState, 0.0)

        while True:
            state, pastCost = frontier.removeMin()

            if state is None and pastCost is None:
                if self.verbose >= 1:
                    print("Searched the entire search space!")
                return

            self.pastCosts[state.location] = pastCost
            self.numStatesExplored += 1
            if self.verbose >= 2:
                print(f"Exploring {state} with pastCost {pastCost}")

            if problem.isEnd(state):
                self.actions = []
                while state != startState:
                    action, prevState = backpointers[state]
                    self.actions.append(action)
                    state = prevState
                self.actions.reverse()
                self.pathCost = pastCost
                return

            for action, newState, cost in problem.successorsAndCosts(state):
                if frontier.update(newState, pastCost + cost):
                    backpointers[newState] = (action, state)
