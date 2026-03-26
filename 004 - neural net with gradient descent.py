import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set a seed so that the results are consistent.
np.random.seed(3) 

## Simple Linear Regression
# Simple Linear Regression Model - find the best fit line to match the data
#    y_hat = wx + b

## Neural Network Model with a Single Perceptron and One Input Node
# forward propagation: 
#   - orgainize all training examples as a vector X of size (1 x m)
#   - perform scalar multiplication and addition: Z = wX + b
#   - y_hat = Z
#
# loss function: 
#   - the difference between the predicted value and the actual value, squared
#     which is: L(w,b) = 1/2 (y_predicted - y_i)^2
#
# cost function:
#   - take the average of the loss function values for each of the training examples
#     which is: 
#        E(m,b) = 1/2n * sum_i (y_pred - y_i)^2
#               = 1/2n * sum_i (m * x_i + b - y_i)^2
#
# backward propagation:
#   - the process of iteratively adjusting w and b, using gradient descent


## general methodology to build a neural network
#    1. Define the neural network structure ( # of input units, # of hidden units, etc)
#    2. Initialize the model's parameters
#    3. Loop:
#         - Implement forward propagation (calculate the perceptron output)
#         - Implement backward propagation (to get the required corrections for the parameters)
#         - Update parameters
#    4. make predictions

path = "tvmarketing.csv"
adv = pd.read_csv(path)

# show scatter plot
# plt.scatter(adv['TV'], adv['Sales'], color="blue", label="TV marketing")
# plt.xlabel("TV")
# plt.ylabel("Sales")
# plt.legend()
# plt.grid(True)
# plt.show()

# normalize the data
X = adv['TV']
Y = adv['Sales']
X_norm = (X - np.mean(X))/np.std(X)
Y_norm = (Y - np.mean(Y))/np.std(Y)


# show scatter plot
# plt.scatter(X_norm, Y_norm, color="blue", label="TV marketing")
# plt.xlabel("TV")
# plt.ylabel("Sales")
# plt.legend()
# plt.grid(True)
# plt.show()

# reshape them to a single row vector
X_norm = np.array(X_norm).reshape((1, len(X_norm)))
Y_norm = np.array(Y_norm).reshape((1, len(Y_norm)))

print ('The shape of X_norm: ' + str(X_norm.shape))
print ('The shape of Y_norm: ' + str(Y_norm.shape))
print ('I have m = %d training examples!' % (X_norm.shape[1]))

'''
The shape of X_norm: (1, 200)
The shape of Y_norm: (1, 200)
I have m = 200 training examples!
'''

## Implementation of the Neural Network Model for Linear Regression
# Defining the Neural Network Structure ---------------------------

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer  (ie. number of nodes)
    n_y -- the size of the output layer (ie. number of nodes)
    """
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    return (n_x, n_y)

(n_x, n_y) = layer_sizes(X_norm, Y_norm)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the output layer is: n_y = " + str(n_y))
'''
The size of the input layer is: n_x = 1
The size of the output layer is: n_y = 1
'''

# Initialize the Model's Parameters ---------------------------
def initialize_parameters(n_x, n_y):
    """
    Returns:
    params -- python dictionary containing your parameters:
                    W -- weight matrix of shape (n_y, n_x)
                    b -- bias value set as a vector of shape (n_y, 1)
    """
    W = np.random.randn(n_y, n_x) * 0.01   # initialized with random numbers
    b = np.zeros((n_y, 1))                 # initialize with zeros
    parameters = {"W": W,
                  "b": b}
    return parameters

parameters = initialize_parameters(n_x, n_y)
print("W = " + str(parameters["W"]))
print("b = " + str(parameters["b"]))


# Forward propagation loop ---------------------------
def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    Y_hat -- The output
    """
    W = parameters["W"]
    b = parameters["b"]
    
    # Forward Propagation to calculate Z.
    Z = np.matmul(W, X) + b
    Y_hat = Z
    return Y_hat

Y_hat = forward_propagation(X_norm, parameters)
print("Some elements of output vector Y_hat:", Y_hat[0, 0:5])


# Define a cost function ---------------------------
# cost function will be the average of the loss function values for each of the training
# examples. Which is:
#   E(m,b) = 1/2n * sum_i (y_pred - y_i)^2
def compute_cost(Y_hat, Y):
    """
    Computes the cost function as a sum of squares
    
    Arguments:
    Y_hat -- The output of the neural network of shape (n_y, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    cost -- sum of squares scaled by 1/(2*number of examples)
    
    """
    # Number of examples.
    m = Y_hat.shape[1]

    # Compute the cost function.
    cost = np.sum((Y_hat - Y)**2)/(2*m)
    return cost

print("cost = " + str(compute_cost(Y_hat, Y_norm)))


# implement backward propagation ---------------------------
# for gradient descent, we need the partial derivatives:
#   dE/dm = 1/n sum_i (y_pred - y_i) * x_i          m = m - (a) dE/dm
#   dE/db = 1/n sum_i (y_pred - y_i)                b = b - (a) dE/db
def backward_propagation(Y_hat, X, Y):
    """
    Implements the backward propagation, calculating gradients
    
    Arguments:
    Y_hat -- the output of the neural network of shape (n_y, number of examples)
    X -- input data of shape (n_x, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    grads -- python dictionary containing gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # Backward propagation: calculate partial derivatives denoted as dW, db for simplicity. 
    dZ = Y_hat - Y
    dW = 1/m * np.dot(dZ, X.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    
    grads = {"dW": dW,
             "db": db}
    
    return grads

grads = backward_propagation(Y_hat, X_norm, Y_norm)

print("dW = " + str(grads["dW"]))
print("db = " + str(grads["db"]))


# update parameters ---------------------------
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule
    
    Arguments:
    parameters -- python dictionary containing parameters 
    grads -- python dictionary containing gradients 
    learning_rate -- learning rate parameter for gradient descent
    
    Returns:
    parameters -- python dictionary containing updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters".
    W = parameters["W"]
    b = parameters["b"]
    
    # Retrieve each gradient from the dictionary "grads".
    dW = grads["dW"]
    db = grads["db"]
    
    # Update rule for each parameter.
    W = W - learning_rate * dW
    b = b - learning_rate * db
    
    parameters = {"W": W,
                  "b": b}
    
    return parameters

parameters_updated = update_parameters(parameters, grads)

print("W updated = " + str(parameters_updated["W"]))
print("b updated = " + str(parameters_updated["b"]))


# put it all together in the neural net model and make predictions ---------------------------
def nn_model(X, Y, num_iterations=10, learning_rate=1.2, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (n_x, number of examples)
    Y -- labels of shape (n_y, number of examples)
    num_iterations -- number of iterations in the loop
    learning_rate -- learning rate parameter for gradient descent
    print_cost -- if True, print the cost every iteration
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to make predictions.
    """
    
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[1]
    
    parameters = initialize_parameters(n_x, n_y)
    
    # Loop
    for i in range(0, num_iterations):
         
        # Forward propagation. Inputs: "X, parameters". Outputs: "Y_hat".
        Y_hat = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "Y_hat, Y". Outputs: "cost".
        cost = compute_cost(Y_hat, Y)
        
        # Backpropagation. Inputs: "Y_hat, X, Y". Outputs: "grads".
        grads = backward_propagation(Y_hat, X, Y)
    
        # Gradient descent parameter update. Inputs: "parameters, grads, learning_rate". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every iteration.
        if print_cost:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

parameters_simple = nn_model(X_norm, Y_norm, num_iterations=30, learning_rate=1.2, print_cost=False)
print("W = " + str(parameters_simple["W"]))
print("b = " + str(parameters_simple["b"]))


# predict some values
def predict(X, Y, parameters, X_pred):
    
    # Retrieve each parameter from the dictionary "parameters".
    W = parameters["W"]
    b = parameters["b"]
    
    # Use the same mean and standard deviation of the original training array X.
    if isinstance(X, pd.Series):
        X_mean = np.mean(X)
        X_std = np.std(X)
        X_pred_norm = ((X_pred - X_mean)/X_std).reshape((1, len(X_pred)))
    else:
        X_mean = np.array(np.mean(X)).reshape((len(X.axes[1]),1))
        X_std = np.array(np.std(X)).reshape((len(X.axes[1]),1))
        X_pred_norm = ((X_pred - X_mean)/X_std)
    # Make predictions.
    Y_pred_norm = np.matmul(W, X_pred_norm) + b
    # Use the same mean and standard deviation of the original training array Y.
    Y_pred = Y_pred_norm * np.std(Y) + np.mean(Y)
    
    return Y_pred[0]

X_pred = np.array([50, 120, 280])
Y_pred = predict(adv["TV"], adv["Sales"], parameters_simple, X_pred)
print(f"TV marketing expenses:\n{X_pred}")
print(f"Predictions of sales:\n{Y_pred}")

# plot
# fig, ax = plt.subplots()
# plt.scatter(adv["TV"], adv["Sales"], color="black")
# plt.xlabel("$x$")
# plt.ylabel("$y$")
# X_line = np.arange(np.min(adv["TV"]),np.max(adv["TV"])*1.1, 0.1)
# Y_line = predict(adv["TV"], adv["Sales"], parameters_simple, X_line)
# ax.plot(X_line, Y_line, "r")
# ax.plot(X_pred, Y_pred, "bo")
# plt.plot()
# plt.show()


### Multiple Linear Regression
# Neural Network Model with a Single Perceptron and Two Input Nodes

df = pd.read_csv('house_prices_train.csv')

X_multi = df[['GrLivArea', 'OverallQual']]
Y_multi = df['SalePrice']

print(X_multi[:5])
print(Y_multi[:5])

# normalize data
X_multi_norm = (X_multi - X_multi.mean()) / X_multi.std()
Y_multi_norm = (Y_multi - Y_multi.mean()) / Y_multi.std()

# - Convert results to the NumPy arrays
# - transpose X_multi_norm to get an array of a shape (2 x m)
# - reshape Y_multi_norm to bring it to the shape (1 x m)
X_multi_norm = np.array(X_multi_norm).T
Y_multi_norm = np.array(Y_multi_norm).reshape((1, len(Y_multi_norm)))

print ('The shape of X: ' + str(X_multi_norm.shape))
print ('The shape of Y: ' + str(Y_multi_norm.shape))
print ('I have m = %d training examples!' % (X_multi_norm.shape[1]))

# Train the model for 100 iterations
parameters_multi = nn_model(X_multi_norm, Y_multi_norm, num_iterations=100, print_cost=True)

print("W = " + str(parameters_multi["W"]))
print("b = " + str(parameters_multi["b"]))

