# DataFrame
import pandas as pd
import re
import math
# Word2vec
import gensim
import math
import numpy as np
from random import randint

# DATASET
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
TRAIN_SIZE = 0.8

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# WORD2VEC
W2V_SIZE = 10
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 5

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)


class Data:
    rawData = None
    data = None
    paddedData = None

    def readDB(self, name, columns):
        self.rawData = pd.read_csv(name, names=columns)
        self.rawData = ProcessingData.shuffle(self.rawData)
        self.clearGarbage()
        self.data = Data.toList(self.rawData)
        return self.data

    @staticmethod
    def toList(database):
        return [_text.split() for _text in database.text]

    @staticmethod
    def makeVoc(list):
        flatten_x = np.concatenate(list)
        return flatten_x

    @staticmethod
    def makeDic(listOfWords):
        dict = {}
        for item in listOfWords:
            if item in dict:
                dict[item] += 1
            else:
                dict[item] = 1
        return dict

    @staticmethod
    def wordtoid(dict):
        temp = 0
        newDic = {}
        dict = sorted(dict, key=dict.get, reverse=True)
        for item in dict:
            newDic[temp] = item
            temp += 1
        return newDic

    @staticmethod
    def reversedDict(dict):
        reverseDic = {}
        for key in dict:
            temp = dict[key]
            reverseDic[temp] = key
        return reverseDic

    @staticmethod
    def textToSequence(baza, dict):
        for i in range(len(baza)):
            for j in range(len(baza[i])):
                baza[i][j] = dict.get(baza[i][j])
        return baza

    def padseq(self, size):
        final = []
        for sequences in self.data:
            result = [0] * size
            for idx in range(len(sequences)):
                lenght = len(sequences)
                result[size - idx - 1] = sequences[lenght - idx - 1]
            final.append(result)
        self.paddedData = final

    def padtovec(self, w2v_model):
        emtpy_arr = [0] * W2V_SIZE
        vectors = []
        labels = []
        flatten = lambda t: [item for sublist in t for item in sublist]
        idx = 0
        for record in self.paddedData:
            result = []
            for word in record:
                if word != 0:
                    result.append(w2v_model.wv[word])
                else:
                    result.append(emtpy_arr)
            vectors.append(flatten(result))
            labels.append(self.rawData.loc[idx]["target"])

            idx += 1
        return vectors, labels

    def clearGarbage(self):
        for sentencesIdx in range(len(self.rawData)):
            # print(type(self.rawData.iloc[sentencesIdx].text))
            phase1 = re.sub('@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+', ' ', self.rawData.iloc[sentencesIdx].text)
            phase2 = " ".join(phase1.split())
            phase3 = ' '.join([w for w in phase2.split() if len(w) > 1])
            phase4 = phase3.lower()
            phase5 = phase4.strip()
            self.rawData.at[sentencesIdx, 'text'] = phase5


class ProcessingData:
    @staticmethod
    def shuffle(X):
        for i in range(len(X) - 1, 0, -1):
            j = randint(0, i)
            X.iloc[i], X.iloc[j] = X.iloc[j], X.iloc[i]
        return X

    @staticmethod
    def split_set(X):
        length_train = round(len(X) * 0.7)
        return X[:length_train], X[length_train:]


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
    def softmax(x, sumka):
        return np.exp(x) / sumka

    @staticmethod
    def softmax_derivative(x, y, sumka):
        result = np.exp(x) * np.exp(y) / (sumka ** 2)

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

    def load_weights(self, weights):
        self._weights = weights


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

    def load_weights(self, weights):
        if len(weights) != self.get_size():
            raise Exception()

        cursor = 0
        last_layer_size = len(self._layers[0])
        for layer in self._layers[1:]:
            for neuron in layer:
                neuron.load_weights(weights[cursor: cursor + last_layer_size + 1])
                cursor += last_layer_size
            last_layer_size = len(layer)
        self._weights = weights

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


class Back:
    def __init__(self, _ne, vectors, labels):
        self.network = _ne
        self.weights = Neuron.generate_weights(self.network.get_size())
        self.vectors = vectors
        self.labels = labels

    def backprop(self, row, l_rate):
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
            layer[-1].delta = errors[-1] * Neuron.softmax_derivative(layer[-1].output, layer[-2].output, sum(np.exp([layer[-1].output, layer[-2].output])))
            layer[-2].delta = errors[-2] * Neuron.softmax_derivative(layer[-2].output, layer[-1].output, sum(np.exp([layer[-1].output, layer[-2].output])))


    def upwe(self, row, l_rate, b_rate):
        for i in range(len(self.network._layers)):
            inputs = row
            if i != 0:
                inputs = [neuron.output for neuron in self.network._layers[i - 1]]
                for neuron in self.network._layers[i]:
                    for j in range(len(inputs)):
                        neuron._weights[j] += l_rate * neuron.delta * inputs[j]
                    neuron._weights[-1] += b_rate * neuron.delta

    def backtrain(self, l_rate, b_rate, nepoch):
        for epoch in range(nepoch):
            sum_error = 0
            for i in range(len(self.vectors)):
                input = self.vectors[i]
                label = self.labels[i]
                outputs = self.network.run(input)

                if label == 0:
                    expected = [0, 1]
                else:
                    expected = [1, 0]
                sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                self.backprop(self.network, expected)
                self.upwe(input, l_rate, b_rate)

            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


DataInstance = Data()

bazadanych = DataInstance.readDB("plik.csv", DATASET_COLUMNS)
train, val = ProcessingData.split_set(bazadanych)

w2v_model = gensim.models.word2vec.Word2Vec(vector_size=W2V_SIZE,
                                            window=W2V_WINDOW,
                                            min_count=1,
                                            workers=8,
                                            sentences=bazadanych)

w2v_model.train(bazadanych, total_examples=len(bazadanych), epochs=W2V_EPOCH)
DataInstance.padseq(280)
vectors, labels = DataInstance.padtovec(w2v_model)

vectorval = vectors[:1000]
labelsval = labels[:1000]

ne = NeuralNetwork()
ne.add_layer(2800)
ne.add_layer(175)
ne.add_layer(10)
ne.add_layer(2)
weights = Neuron.generate_weights(ne.get_size())
ne.load_weights(weights)

acc = 0
backwa = Back(ne, vectors, labels)
backwa.backtrain(0.5, 0.3, 10)

for item in range(len(vectorval)):
    inputa = vectorval[item]
    out = labelsval[item]
    neo = ne.run(inputa)

    if out == 0:
        expected = [0, 1]
    else:
        expected = [1, 0]

    if neo[0] < neo[1]:
        ouf = [0, 1]
    else:
        ouf = [1, 0]
    if ouf == expected:
        acc += 1
print(acc / len(vectorval))
