import numpy as np


class GMM:

    def __init__(self, D, n_classes=3):
        self.D = D
        self.n_classes = n_classes

        d = self.D.shape[1]  # data dimension

        self.mu = [np.random.randn(2) for i in range(n_classes)]  # initial mean
        self.cov = [np.eye(2) for i in range(n_classes)]   # initial covariance
        self.phi =[1/n_classes]*n_classes # class probabilities

        self.cluster_ids = np.zeros(len(D))
        self._W = None

    def fit(self, n_iter=100, verbose=True):
        i = 0
        while i < n_iter:
            self._update_posteriors()
            self._update_parameters()
            if verbose:
                if i % 10 == 0:
                    print('Likelihood: {}'.format(self._log_likelihood()))
            i += 1
        self.cluster_ids = np.argmax(self._W, axis=1)

    def _update_parameters(self):
        W = self._W
        self.phi = [np.mean(W[:, i]) for i in range(self.n_classes)]
        self.mu = [np.average(self.D, weights=W[:, i], axis=0) for i in range(self.n_classes)]
        for i in range(self.n_classes):
            sample_covs = np.zeros(shape=(self.D.shape[0], 2, 2))
            for j in range(self.D.shape[0]):
                x = self.D[j, :] - self.mu[i]
                sample_covs[j, :, :] = np.outer(x, x)
            self.cov[i] = np.average(sample_covs, axis=0, weights=W[:, i])

    def _update_posteriors(self):
        n = self.D.shape[0]
        W = np.zeros(shape=(n, self.n_classes))
        d = self.D.shape[1]
        for j in range(n):
            for i in range(self.n_classes):
                W[j, i] = gaussian(self.D[j, :], self.mu[i], self.cov[i], d=d)*self.phi[i]

            W[j, :] = W[j, :]/np.sum(W[j, :])

        self._W = W

    def _log_likelihood(self):
        L = np.sum(self._W, axis=1)
        L = np.log(L)
        L = np.sum(L)
        return L

    def evaluate(self):
        """
        Evaluate the objective $\sum_{i = 1}^n \sum_{k = 1}^K 1(f(X_i) = k)||X_i - \mu_k||^2$
        :return: objective value
        """
        J = 0

        for i in range(self.n_classes):
            mask = (self.cluster_ids == i)
            norms = np.linalg.norm(self.D[mask, :] - self.mu[i], axis=1)
            norms = norms ** 2
            J = J + np.sum(norms)

        return J


def gaussian(x, mu, sigma, d=2):
    p = -0.5*(x - mu)@np.linalg.inv(sigma)@(x - mu)
    p = np.exp(p)
    D = (2*np.pi)**(-d/2)*(np.linalg.det(sigma))**(-1/2)
    return max(D*p, 1e-20)




