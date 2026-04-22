from SearchProblemUtils import (
    Heuristic,
    SearchProblem,
    State,
    SearchAlgorithm,
)

from MapUtils import (
    CityMap,
    checkValid,
    computeDistance,
    createGridMap,
    getTotalCost,
    makeGridLabel,
    makeTag,
)
from UniformCostSearch import UniformCostSearch
from StepwiseUCS import StepwiseUCS
from typing import List, Tuple, Optional
import json


class ZeroHeuristic(Heuristic):
    """Estimates the cost between locations as 0 distance."""
    def __init__(self, endTag: str, cityMap: CityMap):
        self.endTag = endTag
        self.cityMap = cityMap

    def evaluate(self, state: State) -> float:
        return 0.0
    
########################################################################################
# Problem 4b: "straight-line" heuristic for A*

class StraightLineHeuristic(Heuristic):
    """
    Estimate the cost between locations as the straight-line distance.
        > Hint: you might consider using `computeDistance` defined in `MapUtils.py`
    """
    def __init__(self, endTag: str, cityMap: CityMap):
        self.endTag = endTag
        self.cityMap = cityMap

        # Precompute
        # ### START CODE HERE ###
        # Precompute the GeoLocations of all end-tagged locations so that
        # evaluate() doesn't need to scan the entire map on every call.
        self.endGeoLocations = [
            cityMap.geoLocations[loc]
            for loc, tags in cityMap.tags.items()
            if endTag in tags
        ]

        # Compute a scale factor: metres per one unit of edge cost.
        # computeDistance returns metres, but grid edge costs are 1 per step.
        # Dividing by this factor keeps the heuristic in the same units as the
        # edge costs, ensuring admissibility and avoiding negative reduced costs
        # in the A* reduction.
        sampleLoc = next(iter(cityMap.distances))
        sampleNbr = next(iter(cityMap.distances[sampleLoc]))
        self._metres_per_step = computeDistance(
            cityMap.geoLocations[sampleLoc],
            cityMap.geoLocations[sampleNbr]
        )
        # ### END CODE HERE ###

    def evaluate(self, state: State) -> float:
        # ### START CODE HERE ###
        # Return the minimum straight-line distance from the current location
        # to any end-tagged location, scaled to edge-cost units so the heuristic
        # is admissible (straight-line / scale <= actual path cost).
        currentGeo = self.cityMap.geoLocations[state.location]
        straight_line_metres = min(
            computeDistance(currentGeo, endGeo) for endGeo in self.endGeoLocations
        )
        return straight_line_metres / self._metres_per_step
        # ### END CODE HERE ###


########################################################################################
# Problem 4c: "no waypoints" heuristic for A*

class NoWaypointsHeuristic(Heuristic):
    """
    Returns the minimum distance from `startLocation` to any location with `endTag`,
    ignoring all waypoints.
    """
    def __init__(self, endTag: str, cityMap: CityMap):
        """
        Precompute cost of shortest path from each location to a location with the desired endTag
        """
        # Define a reversed shortest path problem from a special END state
        # (which connects via 0 cost to all end locations) to `startLocation`.
        class ReverseShortestPathProblem(SearchProblem):
            def startState(self) -> State:
                """
                Return special "END" state
                """
                # ### START CODE HERE ###
                return State(location="END")
                # ### END CODE HERE ###

            def isEnd(self, state: State) -> bool:
                """
                Return False for each state.
                Because there is *not* a valid end state (`isEnd` always returns False),
                UCS will exhaustively compute costs to *all* other states.
                """
                # ### START CODE HERE ###
                return False
                # ### END CODE HERE ###

            def successorsAndCosts(
                self, state: State
            ) -> List[Tuple[str, State, float]]:
                # If current location is the special "END" state,
                # return all the locations with the desired endTag and cost 0
                # (i.e, we connect the special location "END" with cost 0 to all locations with endTag)
                # Else, return all the successors of current location and their corresponding distances according to the cityMap
                # ### START CODE HERE ###
                if state.location == "END":
                    # Connect virtual END node to every end-tagged location at cost 0
                    return [
                        (loc, State(location=loc), 0)
                        for loc, tags in cityMap.tags.items()
                        if endTag in tags
                    ]
                else:
                    # Reverse edges — since costs are symmetric this is the same as forward
                    return [
                        (neighbor, State(location=neighbor), distance)
                        for neighbor, distance in cityMap.distances[state.location].items()
                    ]
                # ### END CODE HERE ###

        # Call UCS.solve on our `ReverseShortestPathProblem` instance. Because there is
        # *not* a valid end state (`isEnd` always returns False), will exhaustively
        # compute costs to *all* other states.
        # ### START CODE HERE ###
        ucs = UniformCostSearch()
        ucs.solve(ReverseShortestPathProblem())
        # ### END CODE HERE ###

        # Now that we've exhaustively computed costs from any valid "end" location
        # (any location with `endTag`), we can retrieve `ucs.pastCosts`; this stores
        # the minimum cost path to each state in our state space.
        #   > Note that we're making a critical assumption here: costs are symmetric!
        # ### START CODE HERE ###
        self.pastCosts = ucs.pastCosts
        # ### END CODE HERE ###

    def evaluate(self, state: State) -> float:
        # ### START CODE HERE ###
        # Return precomputed min cost from this location to any end-tagged location.
        # Default to 0 if location is already an end state (cost is 0).
        return self.pastCosts.get(state.location, 0)
        # ### END CODE HERE ###
