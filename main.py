import matplotlib
matplotlib.use("Agg")

import base64
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import io
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from search_session import (
    advance_session,
    cancel_session,
    create_shortest_path_session,
    get_session,
    run_session_to_end,
)
from mdp_session import (
    advance_mdp_session,
    cancel_mdp_session,
    create_number_line_mdp_session,
    get_mdp_session,
    run_mdp_session_to_end,
)


class StartSearchBody(BaseModel):
    grid_w: int = 3
    grid_h: int = 5
    start_location: str = "0,0"
    end_tag: str = "label=2,2"
    waypoint_tags: Optional[List[str]] = None
    heuristic: Optional[str] = None


class StartMdpBody(BaseModel):
    left_reward: float = 10
    right_reward: float = 50
    penalty: float = -5
    n: int = 2
    forward_prob_a1: float = 0.2
    forward_prob_a2: float = 0.3
    discount: Optional[float] = None
    epsilon: Optional[float] = None

app = FastAPI()

@app.get("/")
def testing():
  return {"Testing": "FastAPI"}

@app.get("/plot", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
def get_plot():
    """Endpoint to return the matplotlib plot as an image."""
    # Generate the plot
    fig = create_plot()
    
    # Save the figure to an in-memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0) # Rewind the buffer to the beginning
    
    # Close the figure to free memory
    plt.close(fig)
    
    # Return the buffer as a StreamingResponse with media type image/png
    # headers = {'Content-Disposition': 'inline; filename="plot.png"'}
    return Response(buf.getvalue(), media_type="image/png")

def create_plot():
  x = sp.Symbol('x')
  f = x**2 + 2*x + 1
  f_prime = sp.diff(f, x)  # compute derivative f'(x)
  x_vals = np.linspace(-5,5,100)
  f_lamb = sp.lambdify(x, f, 'numpy')
  f_prime_lamb = sp.lambdify(x, f_prime, 'numpy')

  plt.plot(x_vals, f_lamb(x_vals), label='f(x) = x**2 + 2*x + 1')
  plt.plot(x_vals, f_prime_lamb(x_vals), label="f'(x) = 2x + 2")
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.title('function and its derivative')
  plt.legend()
  plt.grid(True)
  # plt.savefig('math-stats/plot5.png')
  return plt.gcf() # Get the current figure


@app.post("/search/start")
def search_start(body: StartSearchBody = StartSearchBody()):
    """Create a new ShortestPathProblem session. Returns a session_id to use with /search/step."""
    try:
        session = create_shortest_path_session(
            grid_w=body.grid_w,
            grid_h=body.grid_h,
            start_location=body.start_location,
            end_tag=body.end_tag,
            waypoint_tags=body.waypoint_tags,
            heuristic=body.heuristic,
        )
    except ValueError as e:
        return Response(status_code=400, content=str(e).encode(), media_type="text/plain")
    return {"session_id": session.session_id, "status": "running"}


def _search_response(png, tables, done):
    return {
        "image": base64.b64encode(png).decode("ascii"),
        "tables": tables or {},
        "status": "done" if done else "running",
    }


@app.post("/search/step/{session_id}")
def search_step(session_id: str):
    """Advance one step. Returns JSON with base64 image + sidebar tables."""
    session = get_session(session_id)
    if session is None:
        return Response(status_code=404, content=b"session not found")
    png, tables, done = advance_session(session)
    if png is None:
        return Response(status_code=500, content=b"no frame produced")
    return _search_response(png, tables, done)


@app.post("/search/run_to_end/{session_id}")
def search_run_to_end(session_id: str):
    """Drive the search to completion; return JSON with the final frame + tables."""
    session = get_session(session_id)
    if session is None:
        return Response(status_code=404, content=b"session not found")
    png, tables, done = run_session_to_end(session)
    if png is None:
        return Response(status_code=500, content=b"no frame produced")
    return _search_response(png, tables, done)


@app.delete("/search/cancel/{session_id}")
def search_cancel(session_id: str):
    """Cancel and dispose a session — the HTTP equivalent of Ctrl-C."""
    ok = cancel_session(session_id)
    if not ok:
        return Response(status_code=404, content=b"session not found")
    return Response(status_code=204)


@app.post("/mdp/start")
def mdp_start(body: StartMdpBody = StartMdpBody()):
    """Create a new NumberLine value-iteration MDP session."""
    kwargs = {
        "left_reward": body.left_reward,
        "right_reward": body.right_reward,
        "penalty": body.penalty,
        "n": body.n,
        "forward_prob_a1": body.forward_prob_a1,
        "forward_prob_a2": body.forward_prob_a2,
    }
    if body.discount is not None:
        kwargs["discount"] = body.discount
    if body.epsilon is not None:
        kwargs["epsilon"] = body.epsilon
    try:
        session = create_number_line_mdp_session(**kwargs)
    except ValueError as e:
        return Response(status_code=400, content=str(e).encode(), media_type="text/plain")
    return {"session_id": session.session_id, "status": "running"}


def _mdp_response(session, png, breakdown, done):
    return {
        "image": base64.b64encode(png).decode("ascii"),
        "breakdown": breakdown or [],
        "discount": session.discount,
        "status": "done" if done else "running",
    }


@app.post("/mdp/step/{session_id}")
def mdp_step(session_id: str):
    """Advance one VI sweep. Returns JSON with base64 image + Q-value breakdown."""
    session = get_mdp_session(session_id)
    if session is None:
        return Response(status_code=404, content=b"session not found")
    png, breakdown, done = advance_mdp_session(session)
    if png is None:
        return Response(status_code=500, content=b"no frame produced")
    return _mdp_response(session, png, breakdown, done)


@app.post("/mdp/run_to_end/{session_id}")
def mdp_run_to_end(session_id: str):
    """Drive VI to convergence; return JSON with the final frame + breakdown."""
    session = get_mdp_session(session_id)
    if session is None:
        return Response(status_code=404, content=b"session not found")
    png, breakdown, done = run_mdp_session_to_end(session)
    if png is None:
        return Response(status_code=500, content=b"no frame produced")
    return _mdp_response(session, png, breakdown, done)


@app.delete("/mdp/cancel/{session_id}")
def mdp_cancel(session_id: str):
    """Cancel and dispose an MDP session."""
    ok = cancel_mdp_session(session_id)
    if not ok:
        return Response(status_code=404, content=b"session not found")
    return Response(status_code=204)