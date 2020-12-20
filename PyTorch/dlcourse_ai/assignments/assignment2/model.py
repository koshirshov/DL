import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    if predictions.ndim == 1:
        predictions -= np.max(predictions)
        res = np.exp(predictions) / np.sum(np.exp(predictions))
    else:
        predictions -= np.max(predictions, axis = 1, keepdims = True)
        res = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
    # Your final implementation shouldn't have any loops
    return res


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu_layer = ReLULayer()
        self.layer2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        
        # the cleaning of gradients
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)
        
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        # forward pass
        out1 = self.layer1.forward(X)
        out_relu = self.relu_layer.forward(out1)
        out2 = self.layer2.forward(out_relu)
        loss, d_preds = softmax_with_cross_entropy(out2, y)
        # backward pass
        d_out2 = self.layer2.backward(d_preds)
        d_out_relu = self.relu_layer.backward(d_out2)
        d_out1 = self.layer1.backward(d_out_relu)
        
        # After that, implement l2 regularization on all params
        # add regularization
        for param in self.params().values():
            loss_l2, grad_l2 = l2_regularization(param.value, self.reg)
            loss += loss_l2
            param.grad += grad_l2

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        
        out1 = self.layer1.forward(X)
        out_relu = self.relu_layer.forward(out1)
        predictions = self.layer2.forward(out_relu)
        
        probs = softmax(predictions)
        pred = np.argmax(probs, axis=1)        
        return pred

    def params(self):
        # TODO Implement aggregating all of the params
        result = {
            "W1": self.layer1.params()["W"],
            "B1": self.layer1.params()["B"],
            "W2": self.layer2.params()["W"],
            "B2": self.layer2.params()["B"]
        }

        return result
