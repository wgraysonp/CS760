import numpy as np
from collections import Counter
import time


class KNN:

    def __init__(self, D, k=1):
        self.D = D
        self.k = k
        self.predictions = None
        self.confidences = None

    def predict(self, x):
        #func = np.vectorize(self._predict, signature='(m) -> ()')
        #self.predictions = func(x)
        self.predictions = np.zeros(x.shape[0])
        self.confidences = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            self.predictions[i], self.confidences[i] = self._predict(x[i, :])

    def _predict(self, x):
        data_dim = len(x)
        diffs = np.linalg.norm(self.D[:, :data_dim] - x, axis=1)
        nn = []
        for i in range(self.k):
            j = np.argpartition(diffs, i)[i]
            nn.append(self.D[j, data_dim])

        b = Counter(nn)
        mc = b.most_common(1).pop()
        positivity_conf = np.sum(nn)
        return mc[0], positivity_conf

    def get_accuracy(self, y):
        assert self.predictions is not None, "run predict first"
        error = self.predictions - y
        correct = error[error == 0.0]
        return len(correct)/len(error)

    def get_precision(self, y):
        assert self.predictions is not None, "run predict first"
        pp = np.sum(self.predictions)  # predicted positive
        tp = np.sum(y[self.predictions == 1.])  # true positive
        return tp/pp

    def get_recall(self, y):
        assert self.predictions is not None, "run predict first"
        ap = np.sum(y)
        tp = np.sum(y[self.predictions == 1.])
        return tp/ap






