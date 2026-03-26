import numpy as np
import matplotlib.pyplot as plt

# Optimization Using Gradient Descent in One Variable

# example function with one global min: f(x) = e^x - log(x)
# we will use gradient descent to find the global min numerically (as opposed to analytically)

# 1. need to choose a starting point: x0
# 2. calculate the gradient at x0
# 3. find the next step by moving a small distance in the opposite direction of the gradient
#    (the gradient points uphill and we want to move downhill): 
#        x1 = x0 - (a) df/dx(x0)    # x1 = the previous step, subtracting the learning rate (a) 
#                                     times the gradient df/dx at x0
# 4. repeat the process iteratively until either the gradient is almost zero or x1 and x0 
#    are almost the same

# First: define the function we want to minimize and it's derivative
#        f(x)  = e^x - log(x)
#        f'(x) = e^x - 1/x

def f_example_1(x):
    return np.exp(x) - np.log(x)

def dfdx_example_1(x):
    return np.exp(x) - 1/x

# Define x range and function
x_f = np.linspace(0.001, 2.5, 100)
x_fprime = np.linspace(0.2, 2.5, 100)

# gradient descent implementation
def gradient_descent(dfdx, x, learning_rate = 0.1, num_iterations = 100):
    for iteration in range(num_iterations):
        x = x - learning_rate * dfdx(x)
    return x

def gradient_descent_array(dfdx, x, learning_rate = 0.1, num_iterations = 100):
    result_x = []
    result_x.append(x)

    result_y = []
    result_y.append(f_example_1(x))

    for iteration in range(num_iterations):
        x = x - learning_rate * dfdx(x)
        result_x.append(x)
        result_y.append(f_example_1(x))
    # return np.array(result_x), np.array(result_y)
    return result_x, result_y

num_iterations = 25; learning_rate = 0.1; x_initial = 1.6
x_grad, y_grad = gradient_descent_array(dfdx_example_1, x_initial, learning_rate, num_iterations)

# plot function:
plt.plot(x_f, f_example_1(x_f), label="f(x)")
plt.plot(x_fprime, dfdx_example_1(x_fprime), label="f'(x)")
plt.scatter(x_grad, y_grad, color="blue", label="gradient descent")

plt.title("f(x) and f'(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()