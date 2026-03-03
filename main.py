import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import io
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse

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