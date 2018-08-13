import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from itertools import chain

X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1,
                           weights=None, flip_y=0.01, class_sep=0.85, hypercube=True, shift=0.0,
                           scale=1.0, shuffle=True, random_state=0)

# Split the dataset into training and validation segments
X, X_val, y, y_val = train_test_split(X, y, test_size=0.33, random_state=1234)

plt.scatter(X[:, 0], X[:, 1], s=60, c=y, cmap=plt.cm.coolwarm)
plt.xlabel('Feature $x_1$', fontsize=15)
plt.ylabel('Feature $x_2$', fontsize=15)
plt.title("2D Multi-Classification Dataset", fontsize=15)

# plt.show()



clf = LogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False,
                           fit_intercept=True, intercept_scaling=1.0, max_iter=200,
                           multi_class='multinomial', n_jobs=1, penalty='l2', random_state=0,
                           refit=True, scoring=None, solver='sag', tol=0.0001, verbose=0)
clf.fit(X, y)


def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X_val[:, 0].min() - 0.5, X_val[:, 0].max() + 0.5
    y_min, y_max = X_val[:, 1].min() - 0.5, X_val[:, 1].max() + 0.5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Plot the function decision boundary. For that, we will assign
    # a color to each point in the mesh [x_min, x_max] by [y_min, y_max].
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_val[:, 0], X_val[:, 1], s=60, c=y_val, cmap=plt.cm.coolwarm)


pred_func = lambda x: clf.predict(x)
plot_decision_boundary(pred_func)
plt.xlabel('Feature $x_1$', fontsize=15)
plt.ylabel('Feature $x_2$', fontsize=15)
plt.title("Logistic Regression", fontsize=15)

# plt.show()


def sigmoid(z):
    sigmoid = 1.0 / (1.0 + np.exp(-z))
    dsigmoid = sigmoid * (1.0 - sigmoid)

    return sigmoid, dsigmoid


def tanh(x):
    tanh = np.tanh(x)
    dtanh = 1.0 - np.square(tanh)

    return tanh, dtanh


def relu(z):
    relu = z * (z > 0)
    drelu = 1.0 * (z > 0)

    return relu, drelu


# lost function
def softmax(z):
    exp = np.exp(z - np.max(z))
    probs = exp / np.sum(exp, axis=1, keepdims=True)

    return probs


def reshape(array):
    W1 = np.reshape(array[0:input_dim * hidden_dim], (input_dim, hidden_dim))
    W2 = np.reshape(array[input_dim * hidden_dim:hidden_dim * (input_dim + output_dim)], (hidden_dim, output_dim))
    b1 = array[hidden_dim * (input_dim + output_dim):hidden_dim * (input_dim + output_dim + 1)]
    b2 = array[hidden_dim * (input_dim + output_dim + 1):]

    return W1, W2, b1, b2


def forward(params, x, predict=False):
    W1, W2, b1, b2 = reshape(params)

    # Forward propagation
    z2 = np.dot(x, W1) + b1
    #a2, _ = tanh(z2)
    a2, _ = relu(z2)
    z3 = np.dot(a2, W2) + b2
    probs = softmax(z3)

    return np.argmax(probs, axis=1) if predict else probs

def compute_loss(params):
    """ Function to compute the average loss over the dataset. """

    W1, W2, b1, b2 = reshape(params)

    # Forward propagation
    probs = forward(params, X, predict=False)

    # Compute the loss
    correct_logprobs = np.log(probs[range(num_examples), y])
    data_loss = -np.sum(correct_logprobs) / num_examples  # data loss

    reg_loss = weight_decay / 2.0 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))  # weight_decay is lambda

    # Optionally add weight decay regularization term to loss
    loss = data_loss + reg_loss  # total loss including regularization

    return loss

def train_nn(hidden_dim, num_passes=200, update_params=True, dropout_ratio=None, print_loss=None):
    """
    This function learns parameters for the neural network via backprop and batch gradient descent.

    Arguments
    ----------
    hidden_dim : Number of units in the hidden layer
    num_passes : Number of passes through the training data for gradient descent
    update_params : If True, update parameters via gradient descent
    dropout_ratio : Percentage of units to drop out
    print_loss : If integer, print the loss every integer iterations

    Returns
    -------
    params : updated model parameters (weights) stored as an unrolled vector
    grad : gradient computed from backprop stored as an unrolled vector
    """

    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(1234)
    W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
    W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
    b1 = np.ones((1, hidden_dim))
    b2 = np.ones((1, output_dim))

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z2 = np.dot(X, W1) + b1
        a2, da2 = relu(z2)
        if dropout_ratio is not None:
            p = 1.0 - dropout_ratio  # Probability of dropping a unit
            u2 = (np.random.rand(*a2.shape) < p) / p
            a2 *= u2
        z3 = np.dot(a2, W2) + b2
        probs = softmax(z3)

        # Back propagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1.0
        delta2 = np.dot(delta3, W2.T) * da2
        if dropout_ratio is not None:
            delta2 *= u2
        dW2 = np.dot(a2.T, delta3)
        db2 = delta3
        dW1 = np.dot(X.T, delta2)
        db1 = delta2

        # Scale gradient by the number of examples and add regularization to the weight terms.
        # We do not regularize the bias terms.
        dW2 = (dW2 / num_examples) + weight_decay * W2
        dW1 = (dW1 / num_examples) + weight_decay * W1
        db2 = np.sum(db2, axis=0, keepdims=True) / num_examples
        db1 = np.sum(db1, axis=0, keepdims=True) / num_examples

        if update_params:
            # Gradient descent parameter update
            W1 += -lr * dW1
            b1 += -lr * db1
            W2 += -lr * dW2
            b2 += -lr * db2

        # Unroll model parameters and gradient and store them as a long vector
        params = np.asarray(list(chain(*[W1.flatten(), W2.flatten(), b1.flatten(), b2.flatten()])))
        grad = np.asarray(list(chain(*[dW1.flatten(), dW2.flatten(), db1.flatten(), db2.flatten()])))

        # Optionally print the loss after some number of iterations
        if (print_loss is not None) and (i % print_loss == 0):
            print("Loss after iteration %i: %f" % (i, compute_loss(params)))

    return params, grad





num_examples = X.shape[0] # number of examples
input_dim = X.shape[1] # input layer dimensionality
#hidden_dim = 50 # hidden layer dimensionality
output_dim = len(np.unique(y)) # output layer dimensionality

#lr = 0.01 # learning rate for gradient descent
#weight_decay = 0.0005 # regularization strength, smaller values provide greater strength


# Fit a model with a 3-dimensional hidden layer
hidden_dim = 50

# Gradient descent parameters
lr = 0.01 # learning rate for gradient descent
#weight_decay = 0.0005 # regularization strength, smaller values provide greater strength
weight_decay = 0.0005 # regularization strength, smaller values provide greater strength

# Fit neural network
params, _ = train_nn(hidden_dim, num_passes=20000, dropout_ratio=None, print_loss=1000)

# Evaluate accuracy performance on validation set
nn_preds = forward(params, X_val, predict=True)

print("\nClassification Report for Neural Network:\n")
#print(classification_report(y_val, nn_preds))
print("Validation Accuracy for Neural Network: {:.6f}".format(np.mean(y_val==nn_preds)))


# Plot the decision boundary
pred_func = lambda x: forward(params, x, predict=True)
plot_decision_boundary(pred_func)
plt.title("Decision Boundary for Hidden Layer Size {:d}".format(hidden_dim))

plt.show()
