import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    loss = np.sum(W**2)*reg_strength
    grad = 2*W*reg_strength

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    # TODO copy from the previous assignment
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
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
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
    
    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below

        X_padded = np.zeros((batch_size, height + 2 * self.padding, width + 2 * self.padding, self.in_channels))
        X_padded[:, self.padding:self.padding + height, self.padding:self.padding + width, :] = X
        self.X_cache = (X, X_padded)
        X_padded = X_padded[:, :, :, :, np.newaxis]

        W = self.W.value[np.newaxis, :, :, :, :]

        out_height = height - self.filter_size + 2 * self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1
        out = np.zeros((batch_size, out_height, out_width, self.out_channels))

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                X_slice = X_padded[:, y:y + self.filter_size, x:x + self.filter_size, :, :]
                out[:, y, x, :] = np.sum(X_slice * self.W.value, axis=(1, 2, 3)) + self.B.value

        return out

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients
        X, X_padded = self.X_cache

        batch_size, height, width, channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        X_grad = np.zeros_like(X_padded)

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)

                X_slice = X_padded[:, y:y + self.filter_size, x:x + self.filter_size, :, np.newaxis]
                grad = d_out[:, y, x, np.newaxis, np.newaxis, np.newaxis, :]
                self.W.grad += np.sum(grad * X_slice, axis=0)

                X_grad[:, y:y + self.filter_size, x:x + self.filter_size, :] += np.sum(self.W.value * grad, axis=-1)

        self.B.grad += np.sum(d_out, axis=(0, 1, 2))

        return X_grad[:, self.padding:self.padding + height, self.padding:self.padding + width, :]

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool
        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        self.X = X.copy()

        out_height = int((height - self.pool_size) / self.stride) + 1
        out_width = int((width - self.pool_size) / self.stride) + 1
        
        out = np.zeros((batch_size, out_height, out_width, channels))
        
        for y in range(out_height):
            for x in range(out_width):
                X_slice = X[:, y:y + self.pool_size, x:x + self.pool_size, :]
                out[:, y, x, :] = np.amax(X_slice, axis=(1, 2))

        return out

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape

        out_height = int((height - self.pool_size) / self.stride) + 1
        out_width = int((width - self.pool_size) / self.stride) + 1

        out = np.zeros_like(self.X)

        for y in range(out_height):
            for x in range(out_width):
                X_slice = self.X[:, y:y + self.pool_size, x:x + self.pool_size, :]
                grad = d_out[:, y, x, :][:, np.newaxis, np.newaxis, :]
                mask = (X_slice == np.amax(X_slice, (1, 2))[:, np.newaxis, np.newaxis, :])
                out[:, y:y + self.pool_size, x:x + self.pool_size, :] += grad * mask

        return out

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        self.X_shape = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}