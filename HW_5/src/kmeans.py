import numpy as np
import random


class KMeans:

    def __init__(self, D, k=3):

        self.D = D
        self.k = k
        self.cluster_ids = np.zeros(len(D))
        self.C = None

    def fit(self, max_iter=1000, verbose=False):
        '''
        Run the k-means clustering algorithm
        '''
        n = len(self.D)
        C = np.array(self._init_centers())
        C_old = C.copy()
        diff = np.inf
        counter = 0
        while counter < max_iter and diff > 0:
            dists = np.zeros(shape=(n, self.k))
            for i in range(len(C)):
                dists[:, i] = np.linalg.norm(self.D - C[i], axis=1)
            self.cluster_ids = np.argmin(dists, axis=1)

            for i in range(self.k):
                mask = (self.cluster_ids == i)
                C[i] = np.mean(self.D[mask, :], axis=0)

            diff = np.linalg.norm(C_old - C)
            C_old = C.copy()
            counter += 1
            if verbose:
                print("iteration: %d" % counter)
                print("center difference: {:3f}".format(diff))

        self.C = C

    def evaluate(self):
        """
        Evaluate the k means objective $\sum_{i = 1}^n \sum_{k = 1}^K 1(f(X_i) = k)||X_i - \mu_k||^2$
        :return: k means objective value
        """
        J = 0

        for i in range(self.k):
            mask = (self.cluster_ids == i)
            norms = np.linalg.norm(self.D[mask, :] - self.C[i], axis=1)
            norms = norms**2
            J = J + np.sum(norms)

        return J

    def _init_centers(self):
        '''
        Uses k-means++ algorithm to select initial cluster centers
        :return C: list of initial k-means centers
        '''
        C = []
        n = len(self.D)  # number of samples
        idx = random.randrange(0, n, 1)
        C.append(self.D[idx, :])

        for j in range(1, self.k):
            diffs = np.zeros(shape=(n, len(C)))
            for i in range(len(C)):
                diffs[:, i] = np.linalg.norm(self.D - C[i], axis=1)

            diffs = np.min(diffs, axis=1)
            p = diffs**2/(np.sum(diffs**2))
            idx = np.random.choice(n, p=p)
            C.append(self.D[idx, :])

        return C









