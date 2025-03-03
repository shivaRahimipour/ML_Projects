import numpy as np
import scipy as sp
from scipy import stats

"EM Algorithm for Gaussian Model-Based classification"


class ModelBasedClassification:
    def __init__(self):
        self.n = 200  # of observation
        self.p = 2  # of dimension
        self.k = 150  # of labeled observations
        self.G = 2  # of known classes
        self.H = 3  # of known and unknown classes
        self.expected_z_ih = np.asmatrix(np.zeros((self.n, self.H), dtype=float))
        # print(self.expected_z_ih)
        self.current_pi_g = np.ones(self.H) / self.H
        self.current_mean_g = np.asmatrix(np.random.random((self.H, self.p)))
        self.current_cov_g = np.array([np.asmatrix(np.identity(self.p)) for i in range(self.H)])
        # multivariate_normal(mean, cov,count)
        self.X = np.random.multivariate_normal([0, 3], [[0.5, 0], [0, 0.8]], 80)
        self.X = np.vstack((self.X, np.random.multivariate_normal([20, 10], np.identity(2), 70)))
        self.X = np.vstack((self.X, np.random.multivariate_normal([20, 10], np.identity(2), 30)))
        self.X = np.vstack((self.X, np.random.multivariate_normal([0, 3], [[0.5, 0], [0, 0.8]], 20)))

        for i in range(self.n):
            if i <= 80:
                self.expected_z_ih[i, :] = [0, 0, 1]
            if 80 < i <= 150:
                self.expected_z_ih[i, :] = [0, 1, 0]

    def fit(self):
        preloglike = 0
        loglike = 1
        err = 1e-2
        count = 0
        while loglike - preloglike > err:
            preloglike = self.logliklihood()
            self.compute_z_ig()
            self.updateparam()
            loglike = self.logliklihood()
            count += 1
            print('Iteration %d: log-likelihood is %.6f' % (count, loglike))
        print('Terminate at %d-th iteration:log-likelihood is %.6f' % (count, loglike))
        print('MAP results')
        mapp = np.zeros(self.n)
        mapp = np.argmax(self.expected_z_ih, axis=1)
        print(mapp.T)

    def GM(self, x, mean, cov):
        return sp.stats.multivariate_normal.pdf(x, mean.A1, cov)

    "E-step: compute expected value of the maximum log liklihood"

    def compute_z_ig(self):
        for i in range(self.k + 1, self.n):
            for j in range(self.H):
                self.expected_z_ih[i, j] = self.GM(self.X[i, :],
                                                   self.current_mean_g[j, :],
                                                   self.current_cov_g[j, :]) * \
                                           self.current_pi_g[j]

            # sum in all components
            sum_all = 0
            for h in range(self.H):
                sum_all += self.GM(self.X[i, :], self.current_mean_g[h, :], self.current_cov_g[h, :])
            self.expected_z_ih[i, :] /= sum_all
        # assert self.expected_z_ih[i, :].sum() - 1 < 1e-4 , "sum Zig is not equal to 1"

    "M-Step: update parameters"

    def updateparam(self):
        for j in range(self.H):
            "update current_p_g"
            ng = np.sum(self.expected_z_ih[:, j])
            self.current_pi_g[j] = (ng / self.n)

            "update current_mean_g, current_cov_g"
            sumXm = 0
            sumXc = 0
            for i in range(self.n):
                sumXm += self.expected_z_ih[i, j] * self.X[i, :]
                sumXc += self.expected_z_ih[i, j] * (np.transpose(self.X[i, :] - self.current_mean_g[j, :]) *
                                                     (self.X[i, :] - self.current_mean_g[j, :]))

            self.current_mean_g[j] = (sumXm / ng)
            self.current_cov_g[j] = (sumXc / ng)

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


