import os
import sys
import threading
import uuid
from dataclasses import dataclass, field
from typing import Generator, Optional, Tuple

_CS221_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo", "cs221_search")
if _CS221_PATH not in sys.path:
    sys.path.insert(0, _CS221_PATH)

from MapUtils import createGridMap
from ShortestPathProblem import ShortestPathProblem
from StepwiseUCS import WaypointsVisualizer
from SteppableUCS import run_ucs_stepwise, compute_search_tables
from StepRenderer import StepRenderer
from WaypointsShortestPathProblem import WaypointsShortestPathProblem
from Heuristics import NoWaypointsHeuristic, StraightLineHeuristic
from aStarReductionProblem import aStarReduction

_HEURISTICS = {
    "StraightLineHeuristic": StraightLineHeuristic,
    "NoWaypointsHeuristic": NoWaypointsHeuristic,
}


@dataclass
class SearchSession:
    session_id: str
    generator: Generator
    renderer: StepRenderer
    lock: threading.Lock = field(default_factory=threading.Lock)
    done: bool = False
    last_png: Optional[bytes] = None
    last_tables: Optional[dict] = None


_sessions: dict[str, SearchSession] = {}
_sessions_lock = threading.Lock()


def create_shortest_path_session(
    grid_w: int = 3,
    grid_h: int = 5,
    start_location: str = "0,0",
    end_tag: str = "label=2,2",
    waypoint_tags: Optional[list] = None,
    heuristic: Optional[str] = None,
) -> SearchSession:
    """
    Build a search session from user-supplied inputs.

    Base problem: ShortestPathProblem, or WaypointsShortestPathProblem if
    `waypoint_tags` is a non-empty list.
    If `heuristic` is one of the registered names, the base problem is
    wrapped with aStarReduction(baseProblem, heuristic(endTag, cityMap)).
    """
    if grid_w <= 0 or grid_h <= 0:
        raise ValueError("grid_w and grid_h must be positive integers")
    if heuristic and heuristic not in _HEURISTICS:
        raise ValueError(
            f"unknown heuristic '{heuristic}'. Allowed: {sorted(_HEURISTICS.keys())}"
        )
    cityMap = createGridMap(grid_w, grid_h)

    if waypoint_tags:
        base_problem = WaypointsShortestPathProblem(start_location, waypoint_tags, end_tag, cityMap)
        visualizer = WaypointsVisualizer()
    else:
        base_problem = ShortestPathProblem(start_location, end_tag, cityMap)
        visualizer = None

    if heuristic:
        heuristic_instance = _HEURISTICS[heuristic](end_tag, cityMap)
        problem = aStarReduction(base_problem, heuristic_instance)
    else:
        problem = base_problem

    gen = run_ucs_stepwise(problem, cityMap, start_location, end_tag)
    renderer = StepRenderer(cityMap, visualizer=visualizer)
    session = SearchSession(
        session_id=uuid.uuid4().hex,
        generator=gen,
        renderer=renderer,
    )
    with _sessions_lock:
        _sessions[session.session_id] = session
    return session


def get_session(session_id: str) -> Optional[SearchSession]:
    with _sessions_lock:
        return _sessions.get(session_id)


def advance_session(session: SearchSession) -> Tuple[Optional[bytes], Optional[dict], bool]:
    """Advance one step; return (png, tables, done)."""
    with session.lock:
        if session.done:
            return session.last_png, session.last_tables, True
        try:
            ctx = next(session.generator)
        except StopIteration:
            session.done = True
            return session.last_png, session.last_tables, True
        session.last_png = session.renderer.render(ctx)
        session.last_tables = compute_search_tables(ctx)
        if ctx.is_final:
            session.done = True
        return session.last_png, session.last_tables, session.done


def run_session_to_end(session: SearchSession) -> Tuple[Optional[bytes], Optional[dict], bool]:
    """Drive the generator to completion; render only the final context."""
    with session.lock:
        if session.done:
            return session.last_png, session.last_tables, True
        last_ctx = None
        for ctx in session.generator:
            last_ctx = ctx
        session.done = True
        if last_ctx is not None:
            session.last_png = session.renderer.render(last_ctx)
            session.last_tables = compute_search_tables(last_ctx)
        return session.last_png, session.last_tables, True


def cancel_session(session_id: str) -> bool:
    """Close the generator (like Ctrl-C) and dispose the figure. Returns False if not found."""
    with _sessions_lock:
        session = _sessions.pop(session_id, None)
    if session is None:
        return False
    with session.lock:
        try:
            session.generator.close()
        except Exception:
            pass
        session.renderer.close()
    return True
