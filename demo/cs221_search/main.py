import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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
    createStanfordMap,
    locationFromTag,
)
from StepwiseUCS import StepwiseUCS, WaypointsVisualizer
from typing import List, Tuple
import json

from Heuristics import StraightLineHeuristic, ZeroHeuristic, NoWaypointsHeuristic
from ShortestPathProblem import ShortestPathProblem
from WaypointsShortestPathProblem import WaypointsShortestPathProblem
from aStarReductionProblem import aStarReduction

## ---------------------------------------------------------------- helper functions

def printSolution(startLocation, ucs, cityMap, endTag):
    path = extractPath(startLocation, ucs)
    print(f"path: {path}")
    isvalid = checkValid(path, cityMap, startLocation, endTag, [])
    cost = getTotalCost(path, cityMap)
    print(f"cost: {cost}")

## ---------------------------------------------------------------- ShortestPathProblem

def ShortestPathProblem_test_1():
    cityMap=createGridMap(3, 5)
    startLocation=makeGridLabel(0, 0)
    endTag=makeTag("label", makeGridLabel(2, 2))
    # solution: cost=4, steps=12
    ucs = StepwiseUCS(cityMap, startLocation, endTag, verbose=0)
    ucs.solve(ShortestPathProblem(startLocation, endTag, cityMap))
    printSolution(startLocation, ucs, cityMap, endTag)

def ShortestPathProblem_test_2():
    cityMap=createGridMap(30, 30)
    startLocation=makeGridLabel(20, 10)
    endTag=makeTag("x", "5")
    # solution: cost=15
    ucs = StepwiseUCS(cityMap, startLocation, endTag, verbose=0)
    ucs.solve(ShortestPathProblem(startLocation, endTag, cityMap))
    printSolution(startLocation, ucs, cityMap, endTag)

## ---------------------------------------------------------------- WaypointsShortestPathProblem

def WaypointsShortestPathProblem_test_1():
    cityMap=createGridMap(3, 5)
    startLocation=makeGridLabel(0, 0)
    waypointTags=[makeTag("y", 4)]
    endTag=makeTag("label", makeGridLabel(2, 2))
    ucs = StepwiseUCS(cityMap, startLocation, endTag, visualizer=WaypointsVisualizer(), verbose=0)
    ucs.solve(WaypointsShortestPathProblem(startLocation, waypointTags, endTag, cityMap))
    printSolution(startLocation, ucs, cityMap, endTag)

def WaypointsShortestPathProblem_test_2():
    cityMap=createGridMap(6, 5)
    startLocation=makeGridLabel(3, 2)
    waypointTags=[makeTag("y", 3), makeTag("x", 1)]
    endTag=makeTag("label", makeGridLabel(5, 4))
    ucs = StepwiseUCS(cityMap, startLocation, endTag, visualizer=WaypointsVisualizer(), verbose=0)
    ucs.solve(WaypointsShortestPathProblem(startLocation, waypointTags, endTag, cityMap))
    printSolution(startLocation, ucs, cityMap, endTag)


## ---------------------------------------------------------------- aStarReduction + ZeroHeuristic

def aStarReduction_ZeroHeuristic_test():
    cityMap=createGridMap(3, 5)
    startLocation=makeGridLabel(0, 0)
    endTag=makeTag("label", makeGridLabel(2, 2))
    zeroHeuristic = ZeroHeuristic(endTag, cityMap)
    baseProblem = ShortestPathProblem(startLocation, endTag, cityMap)
    aStarProblem = aStarReduction(baseProblem, zeroHeuristic)
    ucs = StepwiseUCS(cityMap, startLocation, endTag, verbose=0)
    ucs.solve(aStarProblem)
    printSolution(startLocation, ucs, cityMap, endTag)

## ---------------------------------------------------------------- aStarReduction + StraightLineHeuristic

def aStarReduction_StraightLineHeuristic_test():
    cityMap=createGridMap(3, 5)
    startLocation=makeGridLabel(0, 0)
    endTag=makeTag("label", makeGridLabel(2, 2))
    heuristic = StraightLineHeuristic(endTag, cityMap)
    heuristicCost = heuristic.evaluate(State(startLocation))
    baseProblem = ShortestPathProblem(startLocation, endTag, cityMap)
    aStarProblem = aStarReduction(baseProblem, heuristic)
    ucs = StepwiseUCS(cityMap, startLocation, endTag, verbose=0)
    ucs.solve(aStarProblem)
    printSolution(startLocation, ucs, cityMap, endTag)


## ---------------------------------------------------------------- aStarReduction + NoWaypointsHeuristic

def aStarReduction_NoWaypointsHeuristic_test_1():
    cityMap=createGridMap(6, 5)
    startLocation=makeGridLabel(3, 2)
    waypointTags=[makeTag("y", 3), makeTag("x", 1)]
    endTag=makeTag("label", makeGridLabel(2, 2))
    heuristic = NoWaypointsHeuristic(endTag, cityMap)
    heuristicCost = heuristic.evaluate(State(startLocation))
    baseProblem = WaypointsShortestPathProblem(startLocation, waypointTags, endTag, cityMap)
    aStarProblem = aStarReduction(baseProblem, heuristic)
    ucs = StepwiseUCS(cityMap, startLocation, endTag, verbose=0)
    ucs.solve(aStarProblem)
    printSolution(startLocation, ucs, cityMap, endTag)

def aStarReduction_NoWaypointsHeuristic_test_2():
    # change start/end points
    cityMap=createGridMap(6, 5)
    startLocation=makeGridLabel(0, 0)
    waypointTags=[makeTag("y", 3), makeTag("x", 1)]
    endTag=makeTag("label", makeGridLabel(5, 4))
    heuristic = NoWaypointsHeuristic(endTag, cityMap)
    heuristicCost = heuristic.evaluate(State(startLocation))
    baseProblem = WaypointsShortestPathProblem(startLocation, waypointTags, endTag, cityMap)
    aStarProblem = aStarReduction(baseProblem, heuristic)
    ucs = StepwiseUCS(cityMap, startLocation, endTag, visualizer=WaypointsVisualizer(), verbose=0)
    ucs.solve(aStarProblem)
    printSolution(startLocation, ucs, cityMap, endTag)


## ---------------------------------------------------------------- test "lower cost found"

def lowerCostFound_test():
    cityMap = createGridMap(4, 3)          # nodes (0,0)…(3,2), all edges cost=1
    # Make the direct (1,1)→(2,1) edge very expensive
    cityMap.distances["1,1"]["2,1"] = 10
    cityMap.distances["2,1"]["1,1"] = 10
    startLocation = "1,1"
    endTag        = makeTag("x", 3)        # rightmost column x=3
    problem = ShortestPathProblem(startLocation, endTag, cityMap)
    ucs     = StepwiseUCS(cityMap, startLocation, endTag, verbose=0)
    ucs.solve(problem)
    printSolution(startLocation, ucs, cityMap, endTag)


## ---------------------------------------------------------------- stanford map

def StanfordMap_test():
    cityMap = createStanfordMap()
    startLocation = locationFromTag(makeTag("landmark", "gates"), cityMap)
    waypointTags  = [makeTag("landmark", "hoover_tower")]
    endTag        = makeTag("landmark", "oval")
    ucs = StepwiseUCS(cityMap, startLocation, endTag, visualizer=WaypointsVisualizer(), verbose=0)
    ucs.solve(WaypointsShortestPathProblem(startLocation, waypointTags, endTag, cityMap))
    printSolution(startLocation, ucs, cityMap, endTag)




## ---------------------------------------------------------------- interactive and non-interactive modes
# Manual mode (default) — press Enter each step
# ucs = StepwiseUCS(cityMap, startLocation, endTag, verbose=0)

# Auto mode — advances every 0.5 seconds
# ucs = StepwiseUCS(cityMap, startLocation, endTag, auto_step=True, verbose=0)

# Auto mode — faster, 0.2 seconds per step
# ucs = StepwiseUCS(cityMap, startLocation, endTag, auto_step=True, step_delay=0.2, verbose=0)





########################################################################################
# Tests

ShortestPathProblem_test_1()
# ShortestPathProblem_test_2()

# WaypointsShortestPathProblem_test_1()
# WaypointsShortestPathProblem_test_2()

# aStarReduction_ZeroHeuristic_test()
# aStarReduction_StraightLineHeuristic_test()
# aStarReduction_NoWaypointsHeuristic_test_1()
# aStarReduction_NoWaypointsHeuristic_test_2()

# lowerCostFound_test()
# StanfordMap_test()
