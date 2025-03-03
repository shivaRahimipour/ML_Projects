import numpy as np
from numpy import linalg as la


def factorAnalysis(x, numFactors, numIterations):
    n, p = x.shape
    mu = x.mean(axis=0)  # mean of each dim
    Sx = 0
    for i in range(n):
        Sx += np.transpose(x[i, :] - mu) * (x[i, :] - mu)
    Sx /= n
    # factorLoadings = np.random.normal(0, Sx, (n, numFactors))
    scaling = np.power(la.det(Sx), 1. / n)
    factorLoadings = np.random.normal(0, np.sqrt(scaling / numFactors), (p, numFactors))

    Psi = np.diag(np.diag(Sx))
    oldL = -np.inf
    # x -= mu
    # cov = (x.T * x) / numData
    # I = np.eye(numFactors)
    # Psi = np.diag(np.diag(cov))
    # scaling = np.power(la.det(cov), 1. / numData)
    # W = np.random.normal(0, np.sqrt(scaling / numFactors), (numDim, numFactors))
    for i in range(numIterations):
        ffT = np.dot(np.mat(factorLoadings), np.mat(factorLoadings.T))
        Beta = np.dot(factorLoadings.T, la.pinv(ffT + Psi))
        Teta = np.eye(numFactors) - np.dot(Beta, factorLoadings) + np.dot(Beta, np.dot(Sx, Beta.T))
        factorLoadings = np.dot(Sx, np.dot(Beta.T, la.pinv(Teta)))
        Psi = np.diag(Sx - np.dot(factorLoadings, np.dot(Beta, Sx)))

        # Log Likelihood
        Sigma = factorLoadings * factorLoadings.T + Psi

        logSigma = np.log(la.det(Sigma) + 1e-10)
        # pinv:pseudo-inverse of a matrix, trace:sum along diagonals of the array
        # L = -n / 2 * (np.trace(la.pinv(Sigma) * Sx.T) + p * np.log(2 * np.pi) + logSigma)
        L = -n / 2 * (np.trace(Sx * la.pinv(Sigma)) + p * np.log(2 * np.pi) + logSigma)

        if (L - oldL) < (1e-6):
            break
        oldL = L

    return factorLoadings, Psi, mu