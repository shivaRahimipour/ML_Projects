import numpy as np
import scipy as sp
from scipy import stats
from EM_DiscriminateAnalysis import DiscriminantAnalysis


class Main:
    def __init__(self):
        # multivariate_normal(mean, cov,)
        self.X1 = np.random.multivariate_normal([0, 3], [[0.5, 0], [0, 0.8]], 20)
        self.X2 = np.random.multivariate_normal([20, 10], np.identity(2), 50)
        self.Xall = np.vstack((self.X1, self.X2))
        # Xall=np.vstack((Xall,np.random.multivariate_normal([0, 3], [[0.5, 0], [0, 0.8]], 20)))

    def run(self):
        X1component = self.componentNumberSelection(self.X1)
        X2component = self.componentNumberSelection(self.X2)

        self.totalG = X1component + X2component
        (ml, mean, cov, pi) = DiscriminantAnalysis(self.totalG, self.Xall).fit()

        X1 = np.random.multivariate_normal([0.5, 2.5], [[0.5, 0], [0, 0.8]], 10)
        X2 = np.random.multivariate_normal([22, 11], np.identity(2), 10)
        X = np.vstack((X1, X2))
        self.compute_z_ig(mean, cov, pi, self.totalG, X)

    def componentNumberSelection(self, X):
        bictemp = np.zeros(10)
        for g in range(1, 11):
            MaximizedLogLikelihood, mean, cov, pi = DiscriminantAnalysis(g, X).fit()
            bictemp[g - 1] = 2 * MaximizedLogLikelihood - self.freeparameters(g, len(X[0])) * np.log(len(X))
        return np.argmax(bictemp, axis=0) + 1

    def freeparameters(self, G, p):
        return G - 1 + G * p + (G * p * (p + 1) / 2)

    def compute_z_ig(self, mean, cov, pi, g, data):
        z_ig = np.asmatrix(np.empty((len(data), g), dtype=float))
        for i in range(len(data)):
            for j in range(g):
                z_ig[i, j] = sp.stats.multivariate_normal.pdf(data[i, :], mean[j, :].A1, cov[j, :]) * \
                             pi[j]

            # sum in all components
            sum_all = 0
            for h in range(g):
                sum_all += sp.stats.multivariate_normal.pdf(data[i, :], mean[h, :].A1, cov[h, :])
            z_ig[i, :] /= sum_all

        print('Test MAP results')
        mapp = np.zeros(len(data))
        mapp = np.argmax(z_ig, axis=1)
        print(mapp.T)
