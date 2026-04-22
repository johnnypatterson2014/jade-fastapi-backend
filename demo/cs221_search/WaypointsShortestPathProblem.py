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
# Problem 3a: Modeling the Waypoints Shortest Path Problem.


class WaypointsShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path from
    `startLocation` to any location with the specified `endTag` such that the path also
    traverses locations that cover the set of tags in `waypointTags`.

    Think carefully about what `memory` representation your States should have!
    """
    def __init__(
        self, startLocation: str, waypointTags: List[str], endTag: str, cityMap: CityMap):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap

        # We want waypointTags to be consistent/canonical (sorted) and hashable (tuple)
        self.waypointTags = tuple(sorted(waypointTags))

    def startState(self) -> State:
        # ### START CODE HERE ###
        # Memory = frozenset of waypointTags already satisfied at the start location
        covered = frozenset(
            tag for tag in self.waypointTags
            if tag in self.cityMap.tags[self.startLocation]
        )
        startState = State(location=self.startLocation, memory=covered)
        # print(f"WaypointsShortestPathProblem :: startState() :: location: {self.startLocation} memory: {covered}")
        return startState
        # ### END CODE HERE ###

    def isEnd(self, state: State) -> bool:
        # ### START CODE HERE ###
        # End when we're at a location with endTag AND all waypoints have been covered

        isEndTag = self.endTag in self.cityMap.tags[state.location]
        isWaypointTags = state.memory == frozenset(self.waypointTags)

        # print(f"cityMap.tags[{state.location}] = {self.cityMap.tags[state.location]}")
        # print(f"WaypointsShortestPathProblem :: isEnd() :: Is '{self.endTag}' in cityMap.tags[{state.location}]? : {isEndTag}")
        # print(f"state.memory: {state.memory}. self.waypointTags: {self.waypointTags}? : {isWaypointTags}")

        return (isEndTag and isWaypointTags)
        # ### END CODE HERE ###

    def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        # ### START CODE HERE ###
        # print(f"WaypointsShortestPathProblem :: successorsAndCosts()")
        results = []
        for neighbor, distance in self.cityMap.distances[state.location].items():
            # Extend covered waypoints with any waypointTags satisfied at the neighbor
            newCovered = state.memory | frozenset(
                tag for tag in self.waypointTags
                if tag in self.cityMap.tags[neighbor]
            )
            # print(f"\t({neighbor}| distance:{distance} memory:{newCovered})")
            results.append((neighbor, State(location=neighbor, memory=newCovered), distance))
        return results
        # ### END CODE HERE ###