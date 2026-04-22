import os
import sys
import threading
import uuid
from dataclasses import dataclass, field
from typing import Generator, Optional, Tuple

_MDP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo", "cs221_mdp_VI")
if _MDP_PATH not in sys.path:
    sys.path.insert(0, _MDP_PATH)

from MDP import NumberLineMDP
from ValueIteration import build_number_line_succAndRewardProb
from SteppableValueIteration import run_vi_stepwise, compute_breakdown
from MdpStepRenderer import MdpStepRenderer


@dataclass
class MdpSession:
    session_id: str
    generator: Generator
    renderer: MdpStepRenderer
    discount: float
    lock: threading.Lock = field(default_factory=threading.Lock)
    done: bool = False
    last_png: Optional[bytes] = None
    last_breakdown: Optional[list] = None


_sessions: dict[str, MdpSession] = {}
_sessions_lock = threading.Lock()


def create_number_line_mdp_session(
    left_reward: float = 10,
    right_reward: float = 50,
    penalty: float = -5,
    n: int = 2,
    forward_prob_a1: float = 0.2,
    forward_prob_a2: float = 0.3,
    discount: float = 1.0,
    epsilon: float = 0.001,
) -> MdpSession:
    """Build a NumberLineMDP value-iteration session from user-supplied parameters."""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not (0.0 <= forward_prob_a1 <= 1.0):
        raise ValueError("forward_prob_a1 must be a float in [0, 1]")
    if not (0.0 <= forward_prob_a2 <= 1.0):
        raise ValueError("forward_prob_a2 must be a float in [0, 1]")
    if not (0.0 <= discount <= 1.0):
        raise ValueError("discount must be a float in [0, 1]")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be a positive float")

    mdp = NumberLineMDP(
        leftReward=left_reward,
        rightReward=right_reward,
        penalty=penalty,
        n=n,
        forward_prob_a1=forward_prob_a1,
        forward_prob_a2=forward_prob_a2,
    )
    succAndRewardProb = build_number_line_succAndRewardProb(mdp)
    gen = run_vi_stepwise(succAndRewardProb, discount=discount, epsilon=epsilon)
    renderer = MdpStepRenderer()
    session = MdpSession(
        session_id=uuid.uuid4().hex,
        generator=gen,
        renderer=renderer,
        discount=discount,
    )
    with _sessions_lock:
        _sessions[session.session_id] = session
    return session


def get_mdp_session(session_id: str) -> Optional[MdpSession]:
    with _sessions_lock:
        return _sessions.get(session_id)


def advance_mdp_session(session: MdpSession) -> Tuple[Optional[bytes], Optional[list], bool]:
    with session.lock:
        if session.done:
            return session.last_png, session.last_breakdown, True
        try:
            ctx = next(session.generator)
        except StopIteration:
            session.done = True
            return session.last_png, session.last_breakdown, True
        session.last_png = session.renderer.render(ctx)
        session.last_breakdown = compute_breakdown(ctx)
        if ctx.converged:
            session.done = True
        return session.last_png, session.last_breakdown, session.done


def run_mdp_session_to_end(session: MdpSession) -> Tuple[Optional[bytes], Optional[list], bool]:
    """Drive the generator to convergence; render only the final context."""
    with session.lock:
        if session.done:
            return session.last_png, session.last_breakdown, True
        last_ctx = None
        for ctx in session.generator:
            last_ctx = ctx
        session.done = True
        if last_ctx is not None:
            session.last_png = session.renderer.render(last_ctx)
            session.last_breakdown = compute_breakdown(last_ctx)
        return session.last_png, session.last_breakdown, True


def cancel_mdp_session(session_id: str) -> bool:
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
