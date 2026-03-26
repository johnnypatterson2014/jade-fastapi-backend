import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set a seed so that the results are consistent.
np.random.seed(3) 

## Function in One Variable
# Newton's method: step to the next point using the formula:
#   x1 = x0 - f'(x0) / f''(x0)

# For example, we will use: f(x)   = e^x - log(x)
#                           f'(x)  = e^x - 1/x
#                           f''(x) = e^x + 1/x^2 

def f_example_1(x):
    return np.exp(x) - np.log(x)

def dfdx_example_1(x):
    return np.exp(x) - 1/x

def d2fdx2_example_1(x):
    return np.exp(x) + 1/(x**2)

x_0 = 1.6
# print(f"f({x_0}) = {f_example_1(x_0)}")
# print(f"f'({x_0}) = {dfdx_example_1(x_0)}")
# print(f"f''({x_0}) = {d2fdx2_example_1(x_0)}")
'''
f(1.6) = 4.483028795149379
f'(1.6) = 4.328032424395115
f''(1.6) = 5.343657424395115
'''

# Plot the function to visualize the global minimum ---------------------------
def plot_f(x_range, y_range, f, ox_position):
    x = np.linspace(*x_range, 100)
    fig, ax = plt.subplots(1,1,figsize=(8,4))

    ax.set_ylim(*y_range)
    ax.set_xlim(*x_range)
    ax.set_ylabel('$f\\,(x)$')
    ax.set_xlabel('$x$')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position(('data', ox_position))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.autoscale(enable=False)
    
    pf = ax.plot(x, f(x), 'k')
    # plt.show()
    return fig, ax

# plot_f([0.001, 2.5], [-0.3, 13], f_example_1, 0.0)

## newton's method ---------------------------
def newtons_method(dfdx, d2fdx2, x, num_iterations=100):
    for iteration in range(num_iterations):
        x = x - dfdx(x) / d2fdx2(x)
        # print(x)
    return x

num_iterations_example_1 = 25; x_initial = 1.6
newtons_example_1 = newtons_method(dfdx_example_1, d2fdx2_example_1, x_initial, num_iterations_example_1)
# print("Newton's method result: x_min =", newtons_example_1)
'''
Newton's method result: x_min = 0.5671432904097838
'''

# compare with gradient descent ---------------------------
def gradient_descent(dfdx, x, learning_rate=0.1, num_iterations=100):
    for iteration in range(num_iterations):
        x = x - learning_rate * dfdx(x)
        # print(x)
    return x

num_iterations = 25; learning_rate = 0.1; x_initial = 1.6
# num_iterations = 25; learning_rate = 0.2; x_initial = 1.6
gd_example_1 = gradient_descent(dfdx_example_1, x_initial, learning_rate, num_iterations)
# print("Gradient descent result: x_min =", gd_example_1) 
'''
Gradient descent result: x_min = 0.5671434156768685
'''

# in this example, newton's method converges after 6 iterations, while gradient descent
# converges after 12 - 15 iterations. However, gradient descent has an advantage: 
# in each step you do not need to calculate second derivative, which in more 
# complicated cases is quite computationally expensive to find. 


## function in 2 variables
# In case of a function in two variables, Newton's method will require even more computations. 
# Starting from the intial point (x0,y0), the step to the next point shoud be done using the expression:
# [ x1 ] = [ x0 ] - H^-1(x0, y0) gradf(x0, y0)
# [ y1 ]   [ y0 ]
#
# where H^-1(x0, y0) is the inverse of the Hessian matrix at point (x0, y0)
# and gradf(x0, y0) is the gradient at that point. 

# For exmaple, we will use:  f(x,y)       = x**4 + 0.8*y**4 + 4*x**2 + 2*y**2 - x*y -0.2*x**2*y
#
#                            gradf(x,y)   = np.array([[4*x**3 + 8*x - y - 0.4*x*y],
#                                                     [3.2*y**3 +4*y - x - 0.2*x**2]])
#
#                            hessian(x,y) = np.array([[12*x**2 + 8 - 0.4*y, -1 - 0.4*x],
#                                                     [-1 - 0.4*x, 9.6*y**2 + 4]])

def f_example_2(x, y):
    return x**4 + 0.8*y**4 + 4*x**2 + 2*y**2 - x*y -0.2*x**2*y

def grad_f_example_2(x, y):
    return np.array([[4*x**3 + 8*x - y - 0.4*x*y],
                     [3.2*y**3 +4*y - x - 0.2*x**2]])

def hessian_f_example_2(x, y):
    hessian_f = np.array([[12*x**2 + 8 - 0.4*y, -1 - 0.4*x],
                         [-1 - 0.4*x, 9.6*y**2 + 4]])
    return hessian_f

x_0, y_0 = 4, 4
# print(f"f{x_0, y_0} = {f_example_2(x_0, y_0)}")
# print(f"grad f{x_0, y_0} = \n{grad_f_example_2(x_0, y_0)}")
# print(f"H{x_0, y_0} = \n{hessian_f_example_2(x_0, y_0)}")
'''
f(4, 4) = 528.0

grad f(4, 4) =
               [[277.6]
                [213.6]]

H(4, 4) =
            [[198.4  -2.6]
             [ -2.6 157.6]]
'''

# plot it
def plot_f_cont_and_surf(f):
    
    fig = plt.figure( figsize=(10,5))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    fig.set_facecolor('#ffffff')
    gs = GridSpec(1, 2, figure=fig)
    axc = fig.add_subplot(gs[0, 0])
    axs = fig.add_subplot(gs[0, 1],  projection='3d')
    
    x_range = [-4, 5]
    y_range = [-4, 5]
    z_range = [0, 1200]
    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X,Y = np.meshgrid(x,y)
    
    cont = axc.contour(X, Y, f(X, Y), cmap='terrain', levels=18, linewidths=2, alpha=0.7)
    axc.set_xlabel('$x$')
    axc.set_ylabel('$y$')
    axc.set_xlim(*x_range)
    axc.set_ylim(*y_range)
    axc.set_aspect("equal")
    axc.autoscale(enable=False)
    
    surf = axs.plot_surface(X,Y, f(X,Y), cmap='terrain', 
                    antialiased=True,cstride=1,rstride=1, alpha=0.69)
    axs.set_xlabel('$x$')
    axs.set_ylabel('$y$')
    axs.set_zlabel('$f$')
    axs.set_xlim(*x_range)
    axs.set_ylim(*y_range)
    axs.set_zlim(*z_range)
    axs.view_init(elev=20, azim=-100)
    axs.autoscale(enable=False)
    # plt.show()
    
    return fig, axc, axs

plot_f_cont_and_surf(f_example_2)

## newton's method for 2 vars ---------------------------
def newtons_method_2(f, grad_f, hessian_f, x_y, num_iterations=100):
    for iteration in range(num_iterations):
        x_y = x_y - np.matmul(np.linalg.inv(hessian_f(x_y[0,0], x_y[1,0])), grad_f(x_y[0,0], x_y[1,0]))
        # print(x_y.T)
    return x_y

num_iterations_example_2 = 25; x_y_initial = np.array([[4], [4]])
newtons_example_2 = newtons_method_2(f_example_2, grad_f_example_2, hessian_f_example_2, 
                                     x_y_initial, num_iterations=num_iterations_example_2)

print("Newton's method result: x_min, y_min =", newtons_example_2.T)
'''
[[2.58273866 2.62128884]]
[[1.59225691 1.67481611]]
[[0.87058917 1.00182107]]
[[0.33519431 0.49397623]]
[[0.04123585 0.12545903]]
[[0.00019466 0.00301029]]
[[-2.48536390e-08  3.55365461e-08]]
[[ 4.15999751e-17 -2.04850948e-17]]
[[0. 0.]]
[[0. 0.]]
...
Newton's method result: x_min, y_min = [[0. 0.]]
'''
# converges after 9 iterations

## compare to gradient descent ---------------------------
def gradient_descent_2(grad_f, x_y, learning_rate=0.1, num_iterations=100):
    for iteration in range(num_iterations):
        x_y = x_y - learning_rate * grad_f(x_y[0,0], x_y[1,0])
        # print(x_y.T)
    return x_y

num_iterations_2 = 300; learning_rate_2 = 0.02; x_y_initial = np.array([[4], [4]])
# num_iterations_2 = 300; learning_rate_2 = 0.03; x_y_initial = np.array([[4], [4]])
gd_example_2 = gradient_descent_2(grad_f_example_2, x_y_initial, learning_rate_2, num_iterations_2)
print("Gradient descent result: x_min, y_min =", gd_example_2) 

