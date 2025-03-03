import numpy as np
import scipy as sp
from scipy import stats

"EM Algorithm for Gaussian Model-Based Clustering"


class ModelBasedClustering:
    def __init__(self):
        self.n = 70  # data
        self.p = 2  # dimension
        self.G = 2  # components
        self.expected_z_ig = np.asmatrix(np.empty((self.n, self.G), dtype=float))
        self.current_pi_g = np.ones(self.G) / self.G
        self.current_mean_g = np.asmatrix(np.random.random((self.G, self.p)))
        self.current_cov_g = np.array([np.asmatrix(np.identity(self.p)) for i in range(self.G)])
        # multivariate_normal(mean, cov,)
        self.X = np.random.multivariate_normal([0, 3], [[0.5, 0], [0, 0.8]], 20)
        self.X = np.vstack((self.X, np.random.multivariate_normal([20, 10], np.identity(2), 50)))


    def fit(self):
        preloglike = 0
        loglike = 1
        err = 1e-10
        count = 0
        while loglike - preloglike > err:
            preloglike = self.logliklihood()
            self.compute_z_ig()
            self.updateParam()
            loglike = self.logliklihood()
            count += 1
            print('Iteration %d: log-likelihood is %.6f' % (count, loglike))
        print('Terminate at %d-th iteration:log-likelihood is %.6f' % (count, loglike))
        print('MAP results')
        mapp = np.zeros(self.n)
        mapp = np.argmax(self.expected_z_ig, axis=1)
        print(mapp.T)
        print('Mean results')
        print(self.current_mean_g)
        print('COV results')
        print(self.current_cov_g)
        print('pi results')
        print(self.current_pi_g)
        
    def GM(self, x, mean, cov):

        return sp.stats.multivariate_normal.pdf(x, mean.A1, cov)

    "E-step: compute expected value of the maximum log liklihood"

    def compute_z_ig(self):
        for i in range(self.n):
            for j in range(self.G):
                self.expected_z_ig[i, j] = self.GM(self.X[i, :],
                                                   self.current_mean_g[j, :],
                                                   self.current_cov_g[j, :]) * \
                                           self.current_pi_g[j]

            # sum in all components
            sum_all = 0
            for h in range(self.G):
                sum_all += self.GM(self.X[i, :], self.current_mean_g[h, :], self.current_cov_g[h, :]) * self.current_pi_g[h]
            self.expected_z_ig[i, :] /= sum_all
        # assert self.expected_z_ig[i, :].sum() - 1 < 1e-4 , "sum Zig is not equal to 1"

    "M-Step: update parameters"

    def updateParam(self):
        for j in range(self.G):
            "update current_p_g"
            ng = np.sum(self.expected_z_ig[:, j])
            self.current_pi_g[j] = (ng / self.n)

            "update current_mean_g, current_cov_g"
            sumXm = 0
            sumXc = 0
            for i in range(self.n):
                sumXm += self.expected_z_ig[i, j] * self.X[i, :]
                sumXc += self.expected_z_ig[i, j] * (np.transpose(self.X[i, :] - self.current_mean_g[j, :]) *
                                                     (self.X[i, :] - self.current_mean_g[j, :]))

            self.current_mean_g[j] = (sumXm / ng)
            self.current_cov_g[j] = (sumXc / ng)

            "update expected_z_ig"
            # self.compute_z_ig()

    "compute log-liklihood"

    def logliklihood(self):
        loglike = 0
        for i in range(self.n):
            tmp = 0
            for j in range(self.G):
                tmp += self.GM(self.X[i, :], self.current_mean_g[j, :], self.current_cov_g[j, :]) * \
                       self.current_pi_g[j]

            loglike += np.log(tmp)
        return loglike








