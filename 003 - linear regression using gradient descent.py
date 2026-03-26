import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

path = "tvmarketing.csv"
adv = pd.read_csv(path)

# Print some part of the dataset
# print(adv.head())

# show scatter plot
# plt.scatter(adv['TV'], adv['Sales'], color="blue", label="TV marketing")
# plt.xlabel("TV")
# plt.ylabel("Sales")
# plt.legend()
# plt.grid(True)
# plt.show()

X = adv['TV']
Y = adv['Sales']

## Linear Regression with NumPy
# You can use the function np.polyfit(x, y, deg) to fit a polynomial of degree deg to points (x, y)
# minimising the sum of squared errors. You can read more in the documentation. 
# Taking deg = 1 you can obtain the slope m and the intercept b of the linear regression line
m_numpy, b_numpy = np.polyfit(X, Y, 1)
print(f"Linear regression with NumPy. Slope: {m_numpy}. Intercept: {b_numpy}")
'''
Linear regression with NumPy. Slope: 0.04753664043301972. Intercept: 7.032593549127696
'''

x_line = np.linspace(min(X), max(X), 200)
y_line = m_numpy * x_line + b_numpy

# Plot
# plt.scatter(X, Y, color='blue', marker='o', label='Data points')
# plt.plot(x_line, y_line, color='red', label='linear regression numpy')
# plt.legend()
# plt.grid(True)
# plt.show()


## Linear Regression using Gradient Descent
# loss function will be the sum of squares (ie. we want to minimize the sum of squares)
#   which is: L(w,b) = 1/2 (y_predicted - y_i)^2
# cost function will be the average of the loss function values for each of the training
# examples. Which is:
#   E(m,b) = 1/2n * sum_i (y_pred - y_i)^2
#          = 1/2n * sum_i (m * x_i + b - y_i)^2
#
# for gradient descent, we need the partial derivatives:
#   dE/dm = 1/n sum_i (m * x_i + b - y_i) * x_i          m = m - (a) dE/dm
#   dE/db = 1/n sum_i (m * x_i + b - y_i)                b = b - (a) dE/db
#
# it is best to normalize the data first (subtract the mean from each training example)
# and divide by the std dev. Note: subtract the mean centers the data, divide by std dev.
# normalizes/scales the data.

# step 1: normalize the data
X_norm = (X - np.mean(X))/np.std(X)
Y_norm = (Y - np.mean(Y))/np.std(Y)

# step 2: define the cost function
def E(m, b, X, Y):
    return 1/(2*len(Y))*np.sum((m*X + b - Y)**2)

# step 3: define the partial derivatives
def dEdm(m, b, X, Y):
    return 1/len(Y)*np.sum(np.dot(m*X + b - Y, X))
    
def dEdb(m, b, X, Y):
    return 1/len(Y)*np.sum(m*X + b - Y)

# step 4: define gradient descent algorithm
def gradient_descent(dEdm, dEdb, m, b, X, Y, learning_rate = 0.001, num_iterations = 1000, print_cost=False):
    for iteration in range(num_iterations):
        m_new = m - learning_rate*dEdm(m, b, X, Y)
        b_new = b - learning_rate*dEdb(m, b, X, Y)
        m = m_new
        b = b_new
        if print_cost:
            print (f"Cost after iteration {iteration}: {E(m, b, X, Y)}")
        
    return m, b

# step 5: run the gradient descent using an initial starting point (m, b) = (0, 0)
m_initial = 0; b_initial = 0; num_iterations = 30; learning_rate = 1.2
m_gd, b_gd = gradient_descent(dEdm, dEdb, m_initial, b_initial, 
                              X_norm, Y_norm, learning_rate, num_iterations, print_cost=False)

print(f"Linear regression with GD (manually): Slope: {m_gd}. Intercept: {b_gd}")

# step 6: to predict values, don't forget to normalize and denormalize x and y values
# X_prediction = np.array([50, 120, 280])
# Use the same mean and standard deviation of the original training array X
# X_predicted_normalized = (X_prediction - np.mean(X))/np.std(X)
# Y_predicted_gd_norm = m_gd * X_predicted_normalized + b_gd
# Use the same mean and standard deviation of the original training array Y
# Y_predicted_gd = Y_predicted_gd_norm * np.std(Y) + np.mean(Y)

y_line_gd_norm = m_gd * X_norm + b_gd
y_line_gd = y_line_gd_norm * np.std(Y) + np.mean(Y)

# Plot
plt.scatter(X, Y, color='blue', marker='o', label='Data points')
plt.plot(x_line, y_line, color='red', label='linear regression numpy')
plt.plot(X, y_line_gd, color='green', label='linear regression GD')
plt.legend()
plt.grid(True)
plt.show()