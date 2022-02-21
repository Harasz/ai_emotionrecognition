from processing_data import ProcessingData
import pandas as pd
import re


W2V_SIZE = 10


class Data:
    rawData = None
    data = None
    paddedData = None

    def read_DB(self, name, columns):
        self.rawData = pd.read_csv(name, names=columns)
        self.rawData = ProcessingData.shuffle(self.rawData)
        self.clear_garbage()
        self.data = Data.toList(self.rawData)
        return self.data

    @staticmethod
    def toList(data_set):
        return [_text.split() for _text in data_set.text]

    def pad_sequence(self, size):
        final = []
        for sequences in self.data:
            result = [0] * size
            for idx in range(len(sequences)):
                length = len(sequences)
                result[size - idx - 1] = sequences[length - idx - 1]
            final.append(result)
        self.paddedData = final

    def pad_to_vector(self, model):
        emtpy_arr = [0] * W2V_SIZE
        vectors_pad = []
        labels_pad = []
        flatten_pad = lambda t: [item for sublist in t for item in sublist]
        idx = 0
        for record in self.paddedData:
            result = []
            for word in record:
                if word != 0:
                    result.append(model.wv[word])
                else:
                    result.append(emtpy_arr)
            vectors_pad.append(flatten_pad(result))
            labels_pad.append(self.rawData.loc[idx]["target"])

            idx += 1
        return vectors_pad, labels_pad

    def clear_garbage(self):
        for sentencesIdx in range(len(self.rawData)):
            phase1 = re.sub('@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+', ' ', self.rawData.iloc[sentencesIdx].text)
            phase2 = " ".join(phase1.split())
            phase3 = ' '.join([w for w in phase2.split() if len(w) > 1])
            phase4 = phase3.lower()
            phase5 = phase4.strip()
            self.rawData.at[sentencesIdx, 'text'] = phase5
