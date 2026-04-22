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

from Heuristics import StraightLineHeuristic


########################################################################################
# Problem 2a: Modeling the Shortest Path Problem.


class ShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path
    from `startLocation` to any location with the specified `endTag`.
    """

    def __init__(self, startLocation: str, endTag: str, cityMap: CityMap):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap

    def startState(self) -> State:
        # ### START CODE HERE ###
        # print(f"ShortestPathProblem :: startState() :: location: {self.startLocation}")
        return State(location=self.startLocation)
        # ### END CODE HERE ###

    def isEnd(self, state: State) -> bool:
        # ### START CODE HERE ###
        isend = self.endTag in self.cityMap.tags[state.location]
        # print(f"cityMap.tags[{state.location}] = {self.cityMap.tags[state.location]}")
        # print(f"ShortestPathProblem :: isEnd() :: Is '{self.endTag}' in cityMap.tags[{state.location}]? : {isend}")
        return isend
        # ### END CODE HERE ###

    def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        # ### START CODE HERE ###
        # print(f"ShortestPathProblem :: successorsAndCosts()")
        children = self.cityMap.distances[state.location].items()
        # for key, value in children:
        #     priorityString = '\t(' + key + '|' + str(value) + ')'
        #     print(priorityString)

        return [
            (neighbor, State(location=neighbor), distance)
            for neighbor, distance in self.cityMap.distances[state.location].items()
        ]
        # ### END CODE HERE ###
        