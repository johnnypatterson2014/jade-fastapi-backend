import numpy as np
import matplotlib.pyplot as plt

# Example: a function in two variables f(x,y) with one global minimum
# eg. f(x,y) = x^2 + y^2

x_range = np.linspace(-5, 5, 200)
y_range = np.linspace(-5, 5, 200)

def f_xy(x, y):
    return 3 * x**2 + 2 * y**2

def df_dx(x, y):
    return 6*x

def df_dy(x, y):
    return 4*y

X, Y = np.meshgrid(x_range, y_range)
Z = f_xy(X, Y)

# 3D Surface plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap='viridis')
# ax.set_title("f(x, y) = x² + y²")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# plt.show()

# Contour Plot (top-down view)
# plt.contourf(X, Y, Z, levels=20, cmap='viridis')  # filled contour
# plt.colorbar(label="z")
# plt.title("Contour Plot")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# Both side by side
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# cp = ax1.contourf(X, Y, Z, levels=20, cmap='viridis')
# fig.colorbar(cp, ax=ax1)
# ax1.set_title("Contour Plot")
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.plot_surface(X, Y, Z, cmap='viridis')
# ax2.set_title("3D Surface")
# plt.tight_layout()
# plt.show()


# gradient descent
# now have a starting point (x0, y0)
# and have 2 gradients to calculate
#
#        x1 = x0 - (a) df/dx(x0, y0)
#        y1 = y0 - (a) df/dy(x0, y0)

def gradient_descent(dfdx, dfdy, x, y, learning_rate = 0.1, num_iterations = 100):
    for iteration in range(num_iterations):
        x, y = x - learning_rate * dfdx(x, y), y - learning_rate * dfdy(x, y)
    return x, y

# num_iterations = 30; learning_rate = 0.25; x_initial = 0.5; y_initial = 0.6
# a, b = gradient_descent(df_dx, df_dy, x_initial, y_initial, learning_rate, num_iterations)
# print(f"a: {a}, b: {b}")

def gradient_descent_array(dfdx, dfdy, x, y, learning_rate = 0.1, num_iterations = 100):
    result_x = []
    result_x.append(x)

    result_y = []
    result_y.append(y)

    result_xy = []
    result_xy.append(f_xy(x, y))

    for iteration in range(num_iterations):
        x, y = x - learning_rate * dfdx(x, y), y - learning_rate * dfdy(x, y)
        result_x.append(x)
        result_y.append(y)
        result_xy.append(f_xy(x, y))

    return result_x, result_y, result_xy

num_iterations = 30; learning_rate = 0.25; x_initial = 0.5; y_initial = 0.6
a, b, c = gradient_descent_array(df_dx, df_dy, x_initial, y_initial, learning_rate, num_iterations)


# plot graphs
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Surface
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)  # alpha makes it semi-transparent
# Scatter on top
ax.scatter(a, b, c, color='red', s=50, zorder=5, label="points")
ax.set_title("Surface + Scatter")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
plt.show()