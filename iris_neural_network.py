import pandas as pd
import numpy as np
from urllib.request import urlretrieve
import matplotlib.pyplot as plot
import time
import math


SIZE_HIDDEN = 6
LEARNING_RATE = 0.001
ITERATIONS = 51000


def get_sizes(X, Y):
    nx = X.shape[0]
    ny = Y.shape[0]
    nh = SIZE_HIDDEN

    return (nx, nh, ny)

def set_params(nx, nh, ny):
    np.random.seed(2) #set an independent seed for genrating seed

    #setting up weights and biases for each layer of the neural network by using laws of matrix multiplication
    W1 = np.random.rand(nh, nx) * 0.0001  # normalised by multiplying 0.0001
    B1 = np.zeros(shape=(nh, 1))
    W2 = np.random.rand(ny, nh) * 0.0001
    B2 = np.zeros(shape=(ny, 1))

    #parameter dictionary
    params = {
        'W1' : W1,
        'B1' : B1,
        'W2' : W2,
        'B2' : B2,
    }

    return params

def fwd_prop(X, params):

    W1 = params['W1']
    B1 = params['B1']
    W2 = params['W2']
    B2 = params['B2']

    T1 = np.dot(W1, X) + B1
    act_T1 = np.tanh(T1)

    T2 = (np.dot(W2, act_T1) + B2)*0.01
    O = 1/(1+np.exp(-T2))

    temp_data = {
        'T1' : T1,
        'act_T1' : act_T1,
        'T2' : T2,
        'O' : O,
    }

    return O, temp_data

def cost_function(O, Y, params):

    m = Y.shape[1] # number of training examples

    # Retrieve W1 and W2 from parameters
    W1 = params['W1']
    W2 = params['W2']

    epsilon = 1e-5
    # Compute the cross-entropy cost
    logprobs = np.multiply(Y, np.log(O + epsilon)) + np.multiply((1 - Y), np.log((1 - O)+epsilon))
    cost = - np.sum(logprobs) / m

    return cost


def bk_prop(X, Y, params, temp_data):

    m = X.shape[1]

    W1 = params['W1']
    W2 = params['W2']
    act_T1 = temp_data['act_T1']
    O = temp_data['O']

    """
    d_W2 = np.dot(act_T1, (2*(Y - O) * np.dot((O - 1).T, O)))
    d_W1 = np.dot(X,  (np.dot(2*(Y - O) * ((O - 1).T*O), W2.T) * np.dot((act_T1 - 1).T, act_T1)))

    d_B1 = d_B2 = 0
    """
    d_T2 = O - Y
    d_W2 = (1 / m) * np.dot(d_T2, act_T1.T)
    d_B2 = (1 / m) * np.sum(d_T2, axis=1, keepdims=True)

    d_T1 = np.multiply(np.dot(W2.T, d_T2), 1 - np.power(act_T1, 2))
    d_W1 = (1 / m) * np.dot(d_T1, X.T)
    d_B1 = (1 / m) * np.sum(d_T1, axis=1, keepdims=True)

    grads = {
     "d_W1": d_W1,
     "d_B1": d_B1,
     "d_W2": d_W2,
     "d_B2": d_B2
    }

    return grads

def param_update(params, grads, alpha = LEARNING_RATE):

    W1 = params['W1']
    W2 = params['W2']
    B1 = params['B1']
    B2 = params['B2']

    d_W1 = grads['d_W1']
    d_B1 = grads['d_B1']
    d_W2 = grads['d_W2']
    d_B2 = grads['d_B2']

    W1 = W1 + alpha*d_W1
    B1 = B1 + alpha*d_B1
    W2 = W2 + alpha*d_W2
    B2 = B2 + alpha*d_B2

    params = {
        'W1' : W1,
        'B1' : B1,
        'W2' : W2,
        'B2' : B2,
    }

    return params

def predict(params, X):
    O, temp_data = fwd_prop(X, params)
    predictions = np.round(O)

    return predictions

def neural_network(X, Y, iters=ITERATIONS, print_cost=False):
    np.random.seed(2)
    (nx, nh, ny) = get_sizes(X, Y)

    params = set_params(nx, nh, ny)
    W1 = params['W1']
    B1 = params['B1']
    W2 = params['W2']
    B2 = params['B2']

    #Applying the gradient descent algorithm
    for i in range(0, iters):

        O, temp_data = fwd_prop(X, params)

        cost = cost_function(O, Y, params)
        grads = bk_prop(X, Y, params, temp_data)
        params = param_update(params, grads)

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    return params

iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
urlretrieve(iris)

df = pd.read_csv(iris, sep=',')
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df.columns = attributes

df.loc[df['class'] == 'Iris-setosa', 'class'] = 1
df.loc[df['class'] == 'Iris-versicolor', 'class'] = 2
df.loc[df['class'] == 'Iris-virginica', 'class'] = 3
df = df[df['class']!=3]
#print(df.head())

X = df[['petal_length', 'petal_width']].values.T
Y = df[['class']].values.T
Y = Y.astype('uint8')


params = neural_network(X, Y , iters=ITERATIONS, print_cost=True)

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[0, :].min() - 0.25, X[0, :].max() + 0.25
    y_min, y_max = X[1, :].min() - 0.25, X[1, :].max() + 0.25

    h = 0.001
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#try to plot a contour for the predictions
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plot.contourf(xx, yy, Z, cmap=plot.cm.Spectral)
    plot.ylabel('x2')
    plot.xlabel('x1')
    plot.scatter(X[0, :], X[1, :], c=y, cmap=plot.cm.Spectral)

plot_decision_boundary(lambda x: predict(params, x.T), X, Y[0,:])
plot.title("Decision Boundary : Hidden Layers = 6")
plot.xlabel('Petal Length')
plot.ylabel('Petal Width')
plot.show()
