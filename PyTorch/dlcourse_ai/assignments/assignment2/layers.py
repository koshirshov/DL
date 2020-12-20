import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = np.sum(W**2)*reg_strength
    grad = 2*W*reg_strength
    
    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    EPS = 1e-5
    max_pred = np.max(predictions, axis=1, keepdims=True) 
    predictions -= max_pred
    probabilities = np.exp(predictions) / np.sum(np.exp(predictions), axis = 1, keepdims=True)
    mask_target = np.zeros(probabilities.shape)
    
    if probabilities.ndim == 1:
        mask_target[target_index] = 1
    elif target_index.ndim == 1:
        mask_target[tuple(np.arange(0, probabilities.shape[0])), tuple(target_index)] = 1
    else:
        mask_target[tuple(np.arange(0, probabilities.shape[0])), tuple(target_index.T[0])] = 1
    #print(mask_target * np.log(probabilities + EPS))
    loss = -np.sum(mask_target * np.log(probabilities + EPS))
    dprediction = probabilities
    dprediction[mask_target.astype(bool)] = dprediction[mask_target.astype(bool)]-1

    return loss, dprediction



class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        result = np.maximum(X, 0)
        self.X = X
        return result

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = d_out*(self.X > 0)
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO_: Implement forward pass
        # Your final implementation shouldn't have any loops
        # raise Exception("Not implemented!")
        W = self.W.value
        B = self.B.value
        self.X = X
        out = np.dot(X, W) + B
        return out

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B
        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO_: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        # raise Exception("Not implemented!")
        X = self.X
        W = self.W.value

        d_W = np.dot(X.T, d_out)
        d_B = np.dot(np.ones((X.shape[0], 1)).T, d_out)
        d_X = np.dot(d_out, W.T)

        self.W.grad += d_W
        self.B.grad += d_B

        return d_X

    def params(self):
        return {'W': self.W, 'B': self.B}