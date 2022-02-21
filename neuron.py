import numpy as np


class Neuron:
    _weights = []
    output = 0
    delta = 0
    bias = 1

    @staticmethod
    def transfer(activation):
        return np.tanh(activation)

    @staticmethod
    def transfer_derivative(output):
        return 1.0 - np.tanh(output) ** 2

    @staticmethod
    def softmax(x, softmax_sum):
        return np.exp(x) / softmax_sum

    @staticmethod
    def softmax_derivative(x, y, softmax_sum):
        result = np.exp(x) * np.exp(y) / (softmax_sum ** 2)

        return result

    def fire(self, values, softmax=False):
        if len(values) != len(self._weights) - 1:
            raise Exception()

        value = np.dot(values, self._weights[:-1])
        if softmax:
            value = value + self._weights[-1] * self.bias
        else:
            value = Neuron.transfer(value + self._weights[-1] * self.bias)
        self.output = value
        return value

    @staticmethod
    def generate_weights(size):
        return np.random.rand(1, size)[0]

    def load_weights(self, loaded_weights):
        self._weights = loaded_weights
