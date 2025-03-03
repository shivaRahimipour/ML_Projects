import numpy as np
from numpy import linalg as la
import scipy as sp
from scipy import stats


def EM_MCFA(X, g, numFactors):
    X = np.asmatrix(X.astype(float))
    n, p = X.shape
    z_ig = np.asmatrix(np.random.random((n, g)))  # np.asmatrix(np.empty((n, g), dtype=float))
    Pi_g = np.ones(g) / g
    Xi_g = np.asmatrix(np.random.random((numFactors, g)))  # np.asmatrix(np.empty((numFactors, g), dtype=float))
    Omega_g = np.array([np.asmatrix(np.identity(numFactors)) for i in range(g)])
    factorLoadings = np.asmatrix(np.random.random((p, numFactors)))
    Psi = np.diag(np.diag(np.identity(p)))
    Beta_g = np.array([np.asmatrix(np.empty((numFactors, p), dtype=float)) for i in range(g)])
    pre_ll = -np.inf
    for iter in range(1000):
        for j in range(g):
            Xi_g_new = np.asmatrix(np.zeros((numFactors, g), dtype=float))
            Omega_g_new = np.array([np.asmatrix(np.identity(numFactors)) for i in range(g)])
            Psi_new = np.asmatrix(np.identity(p))
            n_g = np.sum(z_ig[:, j])
            Pi_g[j] = n_g / n
            # compute Beta
            Beta_g[j] = Omega_g[j] * factorLoadings.T * la.inv(factorLoadings * Omega_g[j] * factorLoadings.T + Psi)
            E_U_ig = np.asmatrix(np.zeros((numFactors, g), dtype=float))
            E_UUt_ig = np.array([np.asmatrix(np.identity(numFactors)) for i in range(g)])
            factorLoadings_sub1 = 0
            factorLoadings_sub2 = 0

            for i in range(n):
                elm = Beta_g[j] * (np.asmatrix(X[i, :]).T - factorLoadings * Xi_g[:, j])
                # update pi, xi, omega, factorLoading, psi
                Xi_g_new[:, j] += z_ig[i, j] * elm
                Omega_g_new[j] += z_ig[i, j] * elm * elm.T
                E_U_ig[:, j] = Xi_g[:, j] + elm
                E_UUt_ig[j] = Omega_g[j] - Beta_g[j] * factorLoadings * Omega_g[j] + E_U_ig[:, j] * E_U_ig[:, j].T
                factorLoadings_sub1 += z_ig[i, j] * X[i, :].T * (Xi_g[:, j] + elm).T
                factorLoadings_sub2 += z_ig[i, j] * (np.identity(numFactors) - Beta_g[j] * factorLoadings) * Omega_g[j]
                factorLoadings_sub2 += z_ig[i, j] * (Xi_g[:, j] + elm) * (Xi_g[:, j] + elm).T
                Psi_new += z_ig[i, j] * (
                        np.asmatrix(X[i, :]).T * np.asmatrix(X[i, :]) - \
                        factorLoadings * E_U_ig[:, j] * np.asmatrix(X[i, :]) - \
                        np.asmatrix(X[i, :]).T * E_U_ig[:, j].T * np.asmatrix(factorLoadings.T) + \
                        factorLoadings * E_UUt_ig[j] * factorLoadings.T)

            Xi_g_new[:, j] /= n_g
            Xi_g_new[:, j] += Xi_g[:, j]
            Omega_g_new[j] /= n_g
            Omega_g_new[j] += (np.identity(numFactors) - Beta_g[j] * factorLoadings) * Omega_g[j]

        factorLoadings_new = factorLoadings_sub1 * la.pinv(factorLoadings_sub2)
        Psi_new = np.diag(np.diag(np.asmatrix(Psi_new))) / n

        for i in range(n):
            sumProb = 0
            for j in range(g):
                # update z
                z_ig[i, j] = Pi_g[j] * GM(X[i, :], factorLoadings * Xi_g[:, j],
                                          factorLoadings * Omega_g[j] * factorLoadings.T + Psi)
                if z_ig[i, j] == 0:
                    z_ig[i, j] += 1e-10
                sumProb += z_ig[i, j]
            z_ig[i, :] /= (sumProb + .1)

        # compute log likelihood
        ll = 0
        for i in range(n):
            tmp = 0
            for j in range(g):
                Ui_g = Xi_g[:, j] + Beta_g[j] * (X[i, :].T - factorLoadings * Xi_g[:, j])
                # Ui_g = np.dot(X[i, :],la.inv(factorLoadings * Omega_g[j] * factorLoadings.T + Psi) * factorLoadings)
                elm = Pi_g[j] * GM(X[i, :], factorLoadings * Ui_g, Psi) * GM(Ui_g.T, Xi_g[:, j], Omega_g[j])
                if elm != 0:
                    tmp += z_ig[i, j] * np.log(elm)
            ll += tmp
        if ll - pre_ll < 1e-20:
            print('MAP results after break')
            mapp = np.zeros(n)
            mapp = np.argmax(z_ig, axis=1)
            print(mapp.T)
            print(iter)
            break;

        factorLoadings = factorLoadings_new
        Xi_g = Xi_g_new
        Omega_g = Omega_g_new
        Psi = Psi_new
        pre_ll = ll

    print('MAP results')
    mapp = np.zeros(n)
    mapp = np.argmax(z_ig, axis=1)
    print(mapp.T)


def GM(x, mean, cov):
    # cov = np.where(np.isfinite(cov), cov, 0)
    # cov = np.where(np.isnan(cov), cov, 0)
    min_eig = np.min(np.real(np.linalg.eigvals(cov)))
    if min_eig < 0:
        cov -= 10 * min_eig * np.eye(*cov.shape)
    return sp.stats.multivariate_normal.pdf(x, mean.A1, cov, allow_singular=True)
