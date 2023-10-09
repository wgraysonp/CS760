import numpy as np
from tqdm import tqdm

class LogisticRegression:

    def __init__(self, D):
        self.D = D
        self.X_train = self.D[:, :-1]
        self.y_train = self.D[:, -1].reshape(-1, 1)
        self.p = 1e-5*np.ones(shape=(self.X_train.shape[1], 1))
        self.predictions = None
        self.confidences = None

    def fit(self, lr=None, max_iter=1000, verbose=False):
        self.p = self.sgd(self.p, lr=lr, max_iter=max_iter, verbose=verbose)

    def predict(self, x):
        self.predictions = np.zeros(x.shape[0])
        self.confidences = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            self.confidences[i] = self.sigmoid(np.dot(x[i, :], self.p))
            self.predictions[i] = 1 if self.sigmoid(np.dot(x[i, :], self.p)) >= 0.5 else 0

    def get_accuracy(self, y):
        assert self.predictions is not None, "run predict first"
        error = self.predictions - y
        correct = error[error == 0]
        return len(correct) / len(error)

    def get_precision(self, y):
        assert self.predictions is not None, "run predict first"
        pp = np.sum(self.predictions)  # predicted positive
        tp = np.sum(y[self.predictions == 1])  # true positive
        return tp / pp

    def get_recall(self, y):
        assert self.predictions is not None, "run predict first"
        ap = np.sum(y)
        tp = np.sum(y[self.predictions == 1])
        return tp / ap

    def loss(self, p):
        X, y = self.X_train, self.y_train
        loss = -y*np.log(self.sigmoid(np.dot(X, p))) - (1 - y)*np.log(1 - self.sigmoid(np.dot(X, p)))
        loss = np.mean(loss)
        return loss

    def sgd(self, p, lr=None, max_iter=1000, verbose=False):
        X = self.X_train
        y = self.y_train
        if lr is None:
            L = 0.25*np.mean([np.linalg.norm(self.X_train[i, :])**2 for i in range(self.X_train.shape[0])])
            lr = 1/L

        print("step size: {}".format(lr))

        for i in tqdm(range(max_iter)):
            grad = np.multiply(X, (self.sigmoid(np.dot(X, p)) - y))
            grad = np.mean(grad, axis=0).reshape(-1, 1)
            p = p - lr*grad
            if verbose:
                if i % 1000 == 999:
                    l = self.loss(p)
                    print("loss: {}".format(l))

        return p


    @staticmethod
    def sigmoid(x):
        # truncate the range for numerical stability
        return np.maximum(np.minimum(1/(1 + np.exp(-x)), 1 - 1e-10), 1e-10)



