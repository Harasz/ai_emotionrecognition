from neuron import Neuron
import numpy as np


class Backward_propagation:
    def __init__(self, _ne, backward_propagation_vectors, backward_propagation_labels):
        self.network = _ne
        self.weights = Neuron.generate_weights(self.network.get_size())
        self.vectors = backward_propagation_vectors
        self.labels = backward_propagation_labels

    def back_propagation(self):
        for i in reversed(range(len(self.network._layers))):
            layer = self.network._layers[i]
            errors = list()
            if i != len(self.network._layers) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network._layers[i + 1]:
                        error += neuron._weights[j] * neuron.delta
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(self.labels[j] - neuron.output)
            for j in range(len(layer)):
                neuron = layer[j]
                neuron.delta = errors[j] * Neuron.transfer_derivative(neuron.output)
            layer[-1].delta = errors[-1] * Neuron.softmax_derivative(layer[-1].output, layer[-2].output,
                                                                     sum(np.exp([layer[-1].output, layer[-2].output])))
            layer[-2].delta = errors[-2] * Neuron.softmax_derivative(layer[-2].output, layer[-1].output,
                                                                     sum(np.exp([layer[-1].output, layer[-2].output])))

    def node_value(self, l_rate, b_rate):
        for i in range(len(self.network._layers)):
            if i != 0:
                inputs = [neuron.output for neuron in self.network._layers[i - 1]]
                for neuron in self.network._layers[i]:
                    for j in range(len(inputs)):
                        neuron._weights[j] += l_rate * neuron.delta * inputs[j]
                    neuron._weights[-1] += b_rate * neuron.delta

    def backward_propagation_train(self, l_rate, b_rate, number_epoch):
        for epoch in range(number_epoch):
            sum_error = 0
            for i in range(len(self.vectors)):
                back_input = self.vectors[i]
                label = self.labels[i]
                outputs = self.network.run(back_input)
                if label == 0:
                    backward_expected = [0, 1]
                else:
                    backward_expected = [1, 0]
                sum_error += sum([(backward_expected[i] - outputs[i]) ** 2 for i in range(len(backward_expected))])
                self.back_propagation()
                self.node_value(l_rate, b_rate)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
