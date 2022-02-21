from neuron import Neuron
import numpy as np


class NeuralNetwork:
    _layers = []
    _weights = []


    def add_layer(self, size=1):
        layer = []
        for i in range(size):
            layer.append(Neuron())
        self._layers.append(layer)

    def get_size(self):
        result = 0
        for layer_idx in range(len(self._layers[:-1])):
            result += len(self._layers[layer_idx]) * len(self._layers[layer_idx + 1]) + len(self._layers[layer_idx])
        return result

    def load_weights(self, loaded_weights):
        if len(loaded_weights) != self.get_size():
            raise Exception()

        cursor = 0
        last_layer_size = len(self._layers[0])
        for layer in self._layers[1:]:
            for neuron in layer:
                neuron.load_weights(loaded_weights[cursor: cursor + last_layer_size + 1])
                cursor += last_layer_size
            last_layer_size = len(layer)
        self._weights = loaded_weights

    def get_weights(self):
        return self._weights

    def get_layers(self):
        return self._layers

    def run(self, inputs):
        if len(inputs) != len(self._layers[0]):
            raise Exception()

        values = inputs
        layerIdx = 0
        for layer in self._layers[1:]:
            next_values = []
            layerIdx += 1
            for neuron in layer:
                if layerIdx == len(self._layers) - 1:
                    next_values.append(neuron.fire(values, softmax=True))
                else:
                    next_values.append(neuron.fire(values))
            if layerIdx == len(self._layers) - 1:
                next_values = [Neuron.softmax(i, sum(np.exp(next_values))) for i in next_values]
            values = next_values
        return values
