import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    params = params
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    h = sigmoid(np.dot(data, W1) + b1)
    theta = np.dot(h, W2) + b2
    y = softmax(theta)
    cost = -np.sum(labels * np.log(y))
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    gradb2 = y - labels
    gradW2 = np.dot(h.T, gradb2)
    gradb1 = np.dot(gradb2, W2.T) * sigmoid_grad(h)
    gradW1 = np.dot(data.T, gradb1)
    gradb2 = np.sum(gradb2, axis=0)
    gradb1 = np.sum(gradb1, axis=0)
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
                           gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 1
    dimensions = [2, 1, 2]
    data = np.array ([[0,1]])
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, i] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )
    #params = np.array([-0.44982774, -1.47693554,  1.12238716, -0.54880661,  1.29102617,  1.26607999, -0.18698024])


    print params
    print 'params.shape', params.shape
    print 'labels', labels
    gradcheck_naive(lambda params1: forward_backward_prop(data, labels, params1,
                                                          dimensions), params)




if __name__ == "__main__":
    sanity_check()
    # your_sanity_checks()