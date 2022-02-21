from backward_propagation import Backward_propagation
from processing_data import ProcessingData
from network import NeuralNetwork
from neuron import Neuron
from data import Data

# Word2vec
import gensim

# DATASET
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
TRAIN_SIZE = 0.8

# TEXT CLEANING
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

DataInstance = Data()

data_base = DataInstance.read_DB("plik.csv", DATASET_COLUMNS)
train, val = ProcessingData.split_set(data_base)

w2v_model = gensim.models.word2vec.Word2Vec(vector_size=W2V_SIZE,
                                            window=W2V_WINDOW,
                                            min_count=1,
                                            workers=8,
                                            sentences=data_base)

w2v_model.train(data_base, total_examples=len(data_base), epochs=W2V_EPOCH)
DataInstance.pad_sequence(280)
vectors, labels = DataInstance.pad_to_vector(w2v_model)

vector_val = vectors[:1000]
labels_val = labels[:1000]

ne = NeuralNetwork()
ne.add_layer(2800)
ne.add_layer(175)
ne.add_layer(10)
ne.add_layer(2)
weights = Neuron.generate_weights(ne.get_size())
ne.load_weights(weights)

acc = 0
backward_propagation = Backward_propagation(ne, vectors, labels)
backward_propagation.backward_propagation_train(0.5, 0.3, 10)

for item in range(len(vector_val)):
    algorithm_input = vector_val[item]
    out = labels_val[item]
    neo = ne.run(algorithm_input)

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
print(acc / len(vector_val))
