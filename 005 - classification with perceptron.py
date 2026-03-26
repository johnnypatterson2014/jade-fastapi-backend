import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.datasets import make_blobs

# Set a seed so that the results are consistent.
np.random.seed(3) 

## binary classification
# categorize a sentence as happy or angry
#   x1 = count of word 'aack' in sentence
#   x2 = count of word 'beep' in sentence
#     if x2 > x1 then classify as angry
#     else classify as happy (x2 <= x1)
#
# Here both x1 and x2 will be either 0 or 1. You can plot those points in a plane, and see the 
# points (observations) belong to two classes, "angry" (red) and "happy" (blue), and a 
# straight line can be used as a decision boundary to separate those two classes.

## Single Perceptron Neural Network with Activation Function
#     z_i = (w1)(x1_i) + (w2)(x2_i) + b
#     then apply sigmoid function to convert real  number to a 1 or 0 
#
#     a = sigmoid(z) = 1 / (1 + e^{-z})    # if a > 0.5  then y_hat = 1 (prediction = angry)
#                                          # if a <= 0.5 then y_hat = 0 (prediction = happy)
#
# putting it together, we have:
#          Z = WX + b           # W: weights; X: x1, x2 training data; b: bias
#          A = sigmoid(Z)

## Cost function
# When dealing with classification problems, the most commonly used cost function is the log loss:
#   log loss = log_loss(W, b)
#   cost = avg of each log loss for all data samples
#        = 1/m * sum_i (log_loss(W, b))                 
#
#   log_loss(W, b) = -y_i * log(a_i) - (1 - y_i) * log(1 - a_i)
#
#   idea: minimize the cost function
#     partial derivatives: need dL/dw1, dL/dw2, dL/db
#           dL/dW = [dL/dw1 dL/dw2] = 1/m (A - Y) @ X^T
#           dL/db = 1/m (A - Y) @ 1 
#
#   update parameters:  W = W - alpha * dL/dW
#                       b = b - alpha * dL/db
#
#   prediction:         y_hat = 1 (if a > 0.5)
#                             = 0 otherwise

## Dataset: we will manually create 30 data points
m = 30

X = np.random.randint(0, 2, (2, m))
Y = np.logical_and(X[0] == 0, X[1] == 1).astype(int).reshape((1, m))

print('Training dataset X containing (x1, x2) coordinates in the columns:')
print(X)
print('Training dataset Y containing labels of two classes (0: blue, 1: red)')
print(Y)

print ('The shape of X is: ' + str(X.shape))
print ('The shape of Y is: ' + str(Y.shape))
print ('I have m = %d training examples!' % (X.shape[1]))
'''

Training dataset X containing (x1, x2) coordinates in the columns:
[[0 0 1 1 0 0 0 1 1 1 0 1 1 1 0 1 1 0 0 0 0 1 1 0 0 0 1 0 0 0]
 [0 1 0 1 1 0 1 0 0 1 1 0 0 1 0 1 0 1 1 1 1 0 1 0 0 1 1 1 0 0]]

Training dataset Y containing labels of two classes (0: blue, 1: red)
[[0 1 0 0 1 0 1 0 0 0 1 0 0 0 0 0 0 1 1 1 1 0 0 0 0 1 0 1 0 0]]

The shape of X is: (2, 30)
The shape of Y is: (1, 30)
I have m = 30 training examples!
'''

## Define Activation Function ---------------------------
def sigmoid(z):
    return 1/(1 + np.exp(-z))
    
## Implementation of the Neural Network Model ---------------------------

def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    return (n_x, n_y)

(n_x, n_y) = layer_sizes(X, Y)
'''
n_x = 2  (size of input layer)
n_y = 1  (size of output layer)
'''

def initialize_parameters(n_x, n_y):
    W = np.random.randn(n_y, n_x) * 0.01
    b = np.zeros((n_y, 1))
    parameters = {"W": W,
                  "b": b}
    return parameters

parameters = initialize_parameters(n_x, n_y)
# print("W = " + str(parameters["W"]))
# print("b = " + str(parameters["b"]))
'''
W = [[-0.00768836 -0.00230031]]
b = [[0.]]
'''

##  forward_propagation ---------------------------
def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A -- The output
    """
    W = parameters["W"]
    b = parameters["b"]
    
    # Forward Propagation to calculate Z.
    Z = np.matmul(W, X) + b
    A = sigmoid(Z)

    return A

A = forward_propagation(X, parameters)

print("Output vector A:", A)
'''
Output vector A: [[0.5        0.49942492 0.49807792 0.49750285 0.49942492 0.5
  0.49942492 0.49807792 0.49807792 0.49750285 0.49942492 0.49807792
  0.49807792 0.49750285 0.5        0.49750285 0.49807792 0.49942492
  0.49942492 0.49942492 0.49942492 0.49807792 0.49750285 0.5
  0.5        0.49942492 0.49750285 0.49942492 0.5        0.5       ]]
'''

## compute cost ---------------------------
def compute_cost(A, Y):
    """
    Computes the log loss cost function
    
    Arguments:
    A -- The output of the neural network of shape (n_y, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    cost -- log loss
    
    """
    # Number of examples.
    m = Y.shape[1]

    # Compute the cost function.
    logprobs = - np.multiply(np.log(A),Y) - np.multiply(np.log(1 - A),1 - Y)
    cost = 1/m * np.sum(logprobs)
    
    return cost

# print("cost = " + str(compute_cost(A, Y)))
'''
cost = 0.6916391611507908
'''

## calculate derivatives for backward propagation ---------------------------
def backward_propagation(A, X, Y):
    """
    Implements the backward propagation, calculating gradients
    
    Arguments:
    A -- the output of the neural network of shape (n_y, number of examples)
    X -- input data of shape (n_x, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    grads -- python dictionary containing gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # Backward propagation: calculate partial derivatives denoted as dW, db for simplicity. 
    dZ = A - Y
    dW = 1/m * np.dot(dZ, X.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    
    grads = {"dW": dW,
             "db": db}
    
    return grads

grads = backward_propagation(A, X, Y)

# print("dW = " + str(grads["dW"]))
# print("db = " + str(grads["db"]))
'''
dW = [[ 0.21571875 -0.06735779]]
db = [[0.16552706]]
'''

## update params ---------------------------
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

# print("W updated = " + str(parameters_updated["W"]))
# print("b updated = " + str(parameters_updated["b"]))
'''
W updated = [[-0.26655087  0.07852904]]
b updated = [[-0.19863247]]
'''

## putting it all together ---------------------------
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
         
        # Forward propagation. Inputs: "X, parameters". Outputs: "A".
        A = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "A, Y". Outputs: "cost".
        cost = compute_cost(A, Y)
        
        # Backpropagation. Inputs: "A, X, Y". Outputs: "grads".
        grads = backward_propagation(A, X, Y)
    
        # Gradient descent parameter update. Inputs: "parameters, grads, learning_rate". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every iteration.
        if print_cost:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

parameters = nn_model(X, Y, num_iterations=50, learning_rate=1.2, print_cost=False)
print("W = " + str(parameters["W"]))
print("b = " + str(parameters["b"]))
'''
W = [[-3.57177421  3.24255633]]
b = [[-1.58411051]]
'''

## plot the decision boundary ---------------------------
def plot_decision_boundary(X, Y, parameters):
    W = parameters["W"]
    b = parameters["b"]

    fig, ax = plt.subplots()
    plt.scatter(X[0, :], X[1, :], c=Y, cmap=colors.ListedColormap(['blue', 'red']));
    
    x_line = np.arange(np.min(X[0,:]),np.max(X[0,:])*1.1, 0.1)
    ax.plot(x_line, - W[0,0] / W[0,1] * x_line + -b[0,0] / W[0,1] , color="black")
    plt.plot()
    plt.show()
    
# plot_decision_boundary(X, Y, parameters)

## make some predictions ---------------------------
def predict(X, parameters):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (blue: False / red: True)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A = forward_propagation(X, parameters)
    predictions = A > 0.5
    
    return predictions

X_pred = np.array([[1, 1, 0, 0],
                   [0, 1, 0, 1]])
Y_pred = predict(X_pred, parameters)

# print(f"Coordinates (in the columns):\n{X_pred}")
# print(f"Predictions:\n{Y_pred}")
'''
Coordinates (in the columns):
[[1 1 0 0]
 [0 1 0 1]]

Predictions:
[[False False False  True]]
'''

## Performance on a Larger Dataset ---------------------------
n_samples = 1000
samples, labels = make_blobs(n_samples=n_samples, 
                             centers=([2.5, 3], [6.7, 7.9]), 
                             cluster_std=1.4,
                             random_state=0)

X_larger = np.transpose(samples)
Y_larger = labels.reshape((1,n_samples))

# plt.scatter(X_larger[0, :], X_larger[1, :], c=Y_larger, cmap=colors.ListedColormap(['blue', 'red']))
# plt.show()

## train the neural net
parameters_larger = nn_model(X_larger, Y_larger, num_iterations=100, learning_rate=1.2, print_cost=False)
# print("W = " + str(parameters_larger["W"]))
# print("b = " + str(parameters_larger["b"]))
'''
W = [[1.01643208 1.13651775]]
b = [[-10.65346577]]
'''

plot_decision_boundary(X_larger, Y_larger, parameters_larger)
