# alternating expectation-conditional maximization (AECM) algorithm for
# Parameter estimation for members of the PGMM family (CCU)
import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
from numpy import linalg as la


def AECM_Algo(X, g, numIterations):
    n, p = X.shape
    z_ig = np.asmatrix(np.zeros((n, g), dtype=float))
    Pi_g = np.ones(g) / g
    mu_g = np.asmatrix(np.random.random((g, p)))
    # initialize z
    kmeans = KMeans(n_clusters=g, random_state=0).fit(X)
    S = np.array([np.asmatrix(np.identity(p)) for i in range(g)])

    for i in range(n):
        z_ig[i, kmeans.labels_[i]] = 1

    # initialize pi_g, mu_g, S_Tilda, factorLoading, Psi
    S_Tilda = np.identity(p)
    for j in range(g):
        n_g = np.sum(z_ig[:, j])
        for i in range(n):
            mu_g[j, :] += z_ig[i, j] * X[i, :]
        mu_g[j, :] /= n_g
        Pi_g[j] = n_g / n

        for i in range(n):
            S[j] += z_ig[i, j] * (np.transpose(X[i, :] - mu_g[j, :]) * (X[i, :] - mu_g[j, :]))
        S[j] /= n_g
        S_Tilda += Pi_g[j] * np.array(S[j])
    eigVals, eigVecs = np.linalg.eig(S_Tilda)
    d = np.sqrt(eigVals)
    P = eigVecs
    factorLoading = d * P
    Psi = np.diag(S_Tilda - factorLoading * factorLoading.T)

    old_L = -np.inf
    counter = 1;
    for iter in range(numIterations):
        for j in range(g):
            n_g = np.sum(z_ig[:, j])
            for i in range(n):
                mu_g[j, :] += z_ig[i, j] * X[i, :]
            mu_g[j, :] /= n_g
            Pi_g[j] = n_g / n
            if (counter != 1):
                # upadte z
                z_ig[i, j] = Pi_g[j] * GM(X[i, :], mu_g[j], factorLoading * factorLoading.T + Psi[j])
                sumProb = 0
                for h in range(g):
                    prob = GM(X[i, :], mu_g[j], factorLoading * factorLoading.T + Psi[j])
                    sumProb += prob;
                z_ig[i, :] /= sumProb
            for i in range(n):
                S[j] += z_ig[i, j] * (np.transpose(X[i, :] - mu_g[j, :]) * (X[i, :] - mu_g[j, :]))
            S[j] /= n_g
            S_Tilda += Pi_g[j] * np.array(S[j])
            Beta_hat = np.dot(factorLoading.T, la.pinv(np.dot(factorLoading, factorLoading.T) + Psi))
            p, q = factorLoading.shape
            Teta_Tilda = np.identity(q) - Beta_hat * factorLoading + Beta_hat * S_Tilda * Beta_hat.T

            # update factorloading, Psi
            factorLoading_new = S_Tilda * Beta_hat.T * la.pinv(Teta_Tilda)
            Psi_new = np.diag(S_Tilda - factorLoading_new * Beta_hat * S_Tilda)
            # update z
            z_ig[i, j] = Pi_g[j] * GM(X[i, :], mu_g[j], factorLoading * factorLoading.T + Psi[j])
            sumProb = 0
            for h in range(g):
                prob = GM(X[i, :], mu_g[j], factorLoading * factorLoading.T + Psi[j])
                sumProb += prob;
            z_ig[i, :] /= sumProb

            log_old = np.inf
            log = 0
            log_next = 0
            log += z_ig[i, j] * np.log(Pi_g[j] * GM(X[i, :], mu_g[j], factorLoading * factorLoading.T + Psi[j]))
            log_next += z_ig[i, j] * np.log(
                Pi_g[j] * GM(X[i, :], mu_g[j], factorLoading_new * factorLoading_new.T + Psi_new[j]))

        counter += 1
        # aitken acceleration
        a_k = (log_next - log) / (log - log_old)
        l_inf = log + (log_next - log) / (1 - a_k)
        if np.abs(l_inf - log) < 1e-9:
            print('MAP results after break')
            mapp = np.zeros(n)
            mapp = np.argmax(z_ig, axis=1)
            print(mapp.T)
            break

        factorLoading = factorLoading_new
        Psi = Psi_new
        log_old = log
    if counter >= numIterations:
        print('MAP results')
        mapp = np.zeros(n)
        mapp = np.argmax(z_ig, axis=1)
        print(mapp.T)


def GM(x, mean, cov):
    cov = np.where(np.isfinite(cov), cov, 0)
    min_eig = np.min(np.real(np.linalg.eigvals(cov)))
    if min_eig < 0:
        cov -= 10 * min_eig * np.eye(*cov.shape)
    return sp.stats.multivariate_normal.pdf(x, mean.A1, la.det(cov) + 1e-10)
