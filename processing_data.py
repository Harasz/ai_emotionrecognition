from random import randint


class ProcessingData:
    @staticmethod
    def shuffle(x):
        for i in range(len(x) - 1, 0, -1):
            j = randint(0, i)
            x.iloc[i], x.iloc[j] = x.iloc[j], x.iloc[i]
        return x

    @staticmethod
    def split_set(x):
        length_train = round(len(x) * 0.7)
        return x[:length_train], x[length_train:]