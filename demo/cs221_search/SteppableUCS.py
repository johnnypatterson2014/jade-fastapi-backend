from typing import Generator, Dict, Any, List, Optional

from SearchProblemUtils import PriorityQueue, SearchProblem
from StepwiseUCS import SearchContext, reconstruct_path


_DONE_PRIORITY = -100000


def _fmt_loc(loc: str) -> str:
    """Same display formatting as StepwiseUCS._format_loc — grid ids as (x,y), geo ids truncated."""
    if "," in loc:
        x, y = loc.split(",")
        return f"({x},{y})"
    return f"\u2026{loc[-8:]}" if len(loc) > 8 else loc


def compute_search_tables(ctx: SearchContext) -> Dict[str, Any]:
    """JSON-ready data mirroring the old sidebar (Current Location / Neighbors / Frontier)."""
    costs_source = (
        ctx.neighbor_costs
        if ctx.neighbor_costs is not None
        else dict(ctx.cityMap.distances[ctx.current_loc])
    )
    neighbors: List[Dict[str, Any]] = [
        {"location": _fmt_loc(nbr), "cost": cost}
        for nbr, cost in sorted(costs_source.items(), key=lambda x: x[1])
    ]

    active_frontier = sorted(
        [
            (state.location, p)
            for state, p in ctx.frontier_priorities.items()
            if p != _DONE_PRIORITY
        ],
        key=lambda x: x[1],
    )
    frontier: List[Dict[str, Any]] = [
        {"location": _fmt_loc(loc), "past_cost": cost}
        for loc, cost in active_frontier
    ]

    tables: Dict[str, Any] = {
        "step": ctx.step,
        "current_location": {
            "location": _fmt_loc(ctx.current_loc),
            "past_cost": ctx.pastCost,
        },
        "neighbors": neighbors,
        "frontier": frontier,
    }

    if ctx.waypoint_tags is not None:
        tags_covered = sorted(ctx.memory) if ctx.memory is not None else []
        tables["waypoints"] = {
            "tags": list(ctx.waypoint_tags),
            "covered": tags_covered,
        }
        tables["current_path"] = {
            "steps": list(ctx.path),
            "tags_covered": tags_covered,
        }

    if ctx.is_final:
        tables["done_info"] = {
            "cost": ctx.pastCost,
            "path": list(ctx.path),
        }

    return tables


def run_ucs_stepwise(problem: SearchProblem, cityMap, startLoc: str, endTag: str) -> Generator[SearchContext, None, None]:
    """
    Generator form of UCS. Each next() yields one SearchContext, then pauses.
    Mirrors StepwiseUCS.solve() semantics without matplotlib or input().
    Closing the generator (gen.close()) unwinds cleanly — equivalent to Ctrl-C.
    """
    frontier = PriorityQueue()
    backpointers = {}
    explored = []
    discovered_edges = {}
    updated_frontier = set()

    startState = problem.startState()
    frontier.update(startState, 0.0)
    num_explored = 0

    while True:
        state, pastCost = frontier.removeMin()
        if state is None:
            return

        num_explored += 1
        explored.append(state.location)

        successors = list(problem.successorsAndCosts(state))
        for _action, newState, cost in successors:
            edge = tuple(sorted([state.location, newState.location]))
            if edge not in discovered_edges:
                discovered_edges[edge] = cost

        path = reconstruct_path(state, startState, backpointers)
        neighbor_locs = set(cityMap.distances[state.location].keys())
        neighbor_costs = {newState.location: cost for _, newState, cost in successors}
        is_final = problem.isEnd(state)

        ctx = SearchContext(
            cityMap=cityMap,
            startLoc=startLoc,
            endTag=endTag,
            current_loc=state.location,
            path=path,
            backpointers=backpointers,
            explored=explored,
            pastCost=pastCost,
            neighbors=neighbor_locs,
            discovered_edges=discovered_edges,
            frontier_priorities=frontier.priorities,
            step=num_explored,
            memory=state.memory,
            waypoint_tags=getattr(problem, "waypointTags", None),
            neighbor_costs=neighbor_costs,
            updated_frontier=updated_frontier,
            is_final=is_final,
        )
        yield ctx

        updated_frontier.discard(state)

        if is_final:
            return

        for action, newState, cost in successors:
            old_priority = frontier.priorities.get(newState)
            if frontier.update(newState, pastCost + cost):
                backpointers[newState] = (action, state)
                if old_priority is not None and old_priority != -100000:
                    updated_frontier.add(newState)
