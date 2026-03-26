import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.datasets import make_blobs

# Set a seed so that the results are consistent.
np.random.seed(3) 

## Classification problem where a stright line is not adequate
# need a non-linear boundary, so we use a neural net with more than 1 layer

# fig, ax = plt.subplots()
# xmin, xmax = -0.2, 1.4
# x_line = np.arange(xmin, xmax, 0.1)
# # Data points (observations) from two classes.
# ax.scatter(0, 0, color="r")
# ax.scatter(0, 1, color="b")
# ax.scatter(1, 0, color="b")
# ax.scatter(1, 1, color="r")
# ax.set_xlim([xmin, xmax])
# ax.set_ylim([-0.1, 1.1])
# ax.set_xlabel('$x_1$')
# ax.set_ylabel('$x_2$')
# # Example of the lines which can be used as a decision boundary to separate two classes.
# ax.plot(x_line, -1 * x_line + 1.5, color="black")
# ax.plot(x_line, -1 * x_line + 0.5, color="black")
# plt.plot()
# plt.show()

## Neural Network Model with Two Layers
# For our example, we have 2 inputs (x1, x2) as the input layer, with a hidden layer 
# that has 2 perceptrons, and one node in the output layer (for a total of 3
# perceptrons). The 2 hidden layer perceptrons have this equation:
#   z_i = (w1)(x1_i) + (w2)(x2_i) + b (and a sigmoid activation function)
#   
#   Z = WX + b           # W: weights; X: x1, x2 training data; b: bias
#   A = sigmoid(Z)
#
# The output layer perceptron has this equation:
#   z_2 = (w1)(a1) + (w2)(a2) + b  (and a sigmoid activation function)
#   where a1 and a2 are the outputs of the 2 hidden layer perceptrons

## cost function
# Here we will use the log loss function. The cost function will be the average over
# all of the loss function values. 

## optimization: we will use gradient descent

## Dataset ---------------------------
# We will create 2000 data points manually.
m = 2000
samples, labels = make_blobs(n_samples=m, 
                             centers=([2.5, 3], [6.7, 7.9], [2.1, 7.9], [7.4, 2.8]), 
                             cluster_std=1.1,
                             random_state=0)
labels[(labels == 0) | (labels == 1)] = 1
labels[(labels == 2) | (labels == 3)] = 0
X = np.transpose(samples)
Y = labels.reshape((1, m))

# plt.scatter(X[0, :], X[1, :], c=Y, cmap=colors.ListedColormap(['blue', 'red']));
# plt.show()

# print ('The shape of X is: ' + str(X.shape))
# print ('The shape of Y is: ' + str(Y.shape))
# print ('I have m = %d training examples!' % (m))
'''
The shape of X is: (2, 2000)
The shape of Y is: (1, 2000)
I have m = 2000 training examples!
'''

## Activation Function ---------------------------
def sigmoid(z):
    return 1/(1 + np.exp(-z))

## Defining the Neural Network Structure ---------------------------
def layer_sizes(X, Y):
    n_x = X.shape[0]    # size of input layer
    n_h = 2             # size of hidden layer
    n_y = Y.shape[0]    # size of output layer
    return (n_x, n_h, n_y)

(n_x, n_h, n_y) = layer_sizes(X, Y)
'''
The size of the input layer is: n_x = 2
The size of the hidden layer is: n_h = 2
The size of the output layer is: n_y = 1
'''

## Initialize the Model's Parameters ---------------------------
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    W1 = np.random.randn(n_h,n_x) * 0.01    # initialize a matrix of shape (n_h, n_x)
    b1 = np.zeros((n_h,1))                  # initialize with zeros
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

parameters = initialize_parameters(n_x, n_h, n_y)

# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
'''
W1 = [[ 0.01788628  0.0043651 ]
      [ 0.00096497 -0.01863493]]

b1 = [[0.]
      [0.]]

W2 = [[-0.00277388 -0.00354759]]

b2 = [[0.]]
'''

## The Loop: forward propagation ---------------------------
def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- the sigmoid output of the second activation
    cache -- python dictionary containing Z1, A1, Z2, A2 
    (that simplifies the calculations in the back propagation step)
    """
    # Retrieve each parameter from the dictionary "parameters".
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Implement forward propagation to calculate A2.
    Z1 = np.matmul(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

A2, cache = forward_propagation(X, parameters)
# print(A2)
'''
[[0.49920157 0.49922234 0.49921223 ... 0.49921215 0.49921043 0.49920665]]
'''

## cost function ---------------------------
def compute_cost(A2, Y):
    """
    Computes the cost function as a log loss
    
    Arguments:
    A2 -- The output of the neural network of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    cost -- log loss
    """
    m = Y.shape[1]    # Number of examples
    logloss = - np.multiply(np.log(A2),Y) - np.multiply(np.log(1 - A2),1 - Y)
    cost = 1/m * np.sum(logloss)
    return cost

# print("cost = " + str(compute_cost(A2, Y)))
'''
cost = 0.6931477703826823
'''

## backward propagation ---------------------------
def backward_propagation(parameters, cache, X, Y):
    """
    Implements the backward propagation, calculating gradients
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- python dictionary containing Z1, A1, Z2, A2
    X -- input data of shape (n_x, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    grads -- python dictionary containing gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # First, retrieve W from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # Backward propagation: calculate partial derivatives denoted as dW1, db1, dW2, db2 for simplicity. 
    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis = 1, keepdims = True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

grads = backward_propagation(parameters, cache, X, Y)

# print("dW1 = " + str(grads["dW1"]))
# print("db1 = " + str(grads["db1"]))
# print("dW2 = " + str(grads["dW2"]))
# print("db2 = " + str(grads["db2"]))
'''
dW1 = [[-1.49856632e-05  1.67791519e-05]
       [-2.12394543e-05  2.43895135e-05]]

db1 = [[5.11207671e-07]
       [7.06236219e-07]]

dW2 = [[-0.00032641 -0.0002606 ]]

db2 = [[-0.00078732]]
'''

## update parameters ---------------------------
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule
    
    Arguments:
    parameters -- python dictionary containing parameters 
    grads -- python dictionary containing gradients
    learning_rate -- learning rate for gradient descent
    
    Returns:
    parameters -- python dictionary containing updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters".
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Retrieve each gradient from the dictionary "grads".
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    # Update rule for each parameter.
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

## putting it all together ---------------------------
def nn_model(X, Y, n_h, num_iterations=10, learning_rate=1.2, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (n_x, number of examples)
    Y -- labels of shape (n_y, number of examples)
    num_iterations -- number of iterations in the loop
    learning_rate -- learning rate parameter for gradient descent
    print_cost -- if True, print the cost every iteration
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    # Initialize parameters.
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    # Loop.
    for i in range(0, num_iterations):
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "A2, Y". Outputs: "cost".
        cost = compute_cost(A2, Y)
        
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
        
        # Gradient descent parameter update. Inputs: "parameters, grads, learning_rate". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every iteration.
        if print_cost:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

parameters = nn_model(X, Y, n_h=2, num_iterations=3000, learning_rate=1.2, print_cost=False)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
'''
W1 = [[ 1.96413256 -1.74523836]
      [ 2.25122339 -1.97522134]]

b1 = [[-4.84602739]
      [ 6.32005311]]

W2 = [[-7.23772659  7.12083386]]

b2 = [[-3.43159859]]
'''

## make predictions ---------------------------
# Computes probabilities using forward propagation, and make classification 
# to 0/1 using 0.5 as the threshold
def predict(X, parameters):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (blue: 0 / red: 1)
    """
    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5
    return predictions


X_pred = np.array([[2, 8, 2, 8], [2, 8, 8, 2]])
Y_pred = predict(X_pred, parameters)

# print(f"Coordinates (in the columns):\n{X_pred}")
# print(f"Predictions:\n{Y_pred}")
'''
Coordinates (in the columns):
[[2 8 2 8]
 [2 8 8 2]]

Predictions:
[[ True  True False False]]
'''

## Visualize the boundary line ---------------------------
def plot_decision_boundary(predict, parameters, X, Y):
    # Define bounds of the domain.
    min1, max1 = X[0, :].min()-1, X[0, :].max()+1
    min2, max2 = X[1, :].min()-1, X[1, :].max()+1
    # Define the x and y scale.
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    # Create all of the lines and rows of the grid.
    xx, yy = np.meshgrid(x1grid, x2grid)
    # Flatten each grid to a vector.
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((1, len(r1))), r2.reshape((1, len(r2)))
    # Vertical stack vectors to create x1,x2 input for the model.
    grid = np.vstack((r1,r2))
    # Make predictions for the grid.
    predictions = predict(grid, parameters)
    # Reshape the predictions back into a grid.
    zz = predictions.reshape(xx.shape)
    # Plot the grid of x, y and z values as a surface.
    plt.contourf(xx, yy, zz, cmap=plt.cm.Spectral.reversed())
    plt.scatter(X[0, :], X[1, :], c=Y, cmap=colors.ListedColormap(['blue', 'red']));

# Plot the decision boundary.
# plot_decision_boundary(predict, parameters, X, Y)
# plt.title("Decision Boundary for hidden layer size " + str(n_h))
# plt.show()


## Example with a different dataset ---------------------------
n_samples = 2000
samples, labels = make_blobs(n_samples=n_samples, 
                             centers=([2.5, 3], [6.7, 7.9], [2.1, 7.9], [7.4, 2.8]), 
                             cluster_std=1.1,
                             random_state=0)
labels[(labels == 0)] = 0
labels[(labels == 1)] = 1
labels[(labels == 2) | (labels == 3)] = 1
X_2 = np.transpose(samples)
Y_2 = labels.reshape((1,n_samples))

# plt.scatter(X_2[0, :], X_2[1, :], c=Y_2, cmap=colors.ListedColormap(['blue', 'red']))
# plt.show()

parameters_2 = nn_model(X_2, Y_2, n_h=2, num_iterations=3000, learning_rate=1.2, print_cost=False)
# plot_decision_boundary(predict, parameters_2, X_2, Y_2)
# plt.title("Decision Boundary")
# plt.show()

