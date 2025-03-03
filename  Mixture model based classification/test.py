import numpy as np
from numpy import linalg as la

##clustering
from EM_Clustering import ModelBasedClustering
g=ModelBasedClustering()
g.fit()


##discriminant analysis
#from BIC_NumberOfComponents import Main
#dA=Main()
#dA.run()

##factor analysis
#from EM_FactorAnalysis import factorAnalysis
##scores = np.asmatrix(np.genfromtxt("http://www-eio.upc.es/~jan/Data/BookOpenClosed.dat",delimiter="\t"))
#scores = np.asmatrix(np.genfromtxt("BookOpenClosed.dat",delimiter="\t"))
#[W, Psi, mu] = factorAnalysis(scores, 2, 1000)
#print("factor loading matrix:")
#print(W)
#print("Psi matrix:")
#print(Psi)
#print("mean matrix:")
#print(mu)
##dimension reduction
#ffT = np.dot(np.mat(W), np.mat(W.T))
#Beta = np.dot(W.T, la.pinv(W*W.T + Psi))
#z = np.dot(scores,la.inv(W*W.T+Psi)*W)
#print("z is")
#print(z)


##AECM_PGMM
# X = np.random.multivariate_normal([0, 3], [[0.5, 0], [0, 0.8]], 20)
# X = np.vstack((X, np.random.multivariate_normal([20, 10], np.identity(2), 50)))
#
# from AECM_PGMM import AECM_Algo
# AECM_Algo(X,2,1000)

##EM for Mixuture of Common Factor Analysis
# import pandas as pd
#
# X = np.random.multivariate_normal([0, 3], [[0.5, 0], [0, 0.8]], 20)
# X = np.vstack((X, np.random.multivariate_normal([20, 10], np.identity(2), 50)))
# #X = np.vstack((X, np.random.multivariate_normal([50, 100], np.identity(2), 10)))
#
# scores = np.asmatrix(np.genfromtxt("new.dat",delimiter=","))
# df = pd.read_csv('shiva.csv',sep=',',skiprows=1)
#
#
# from EM_MCFA import EM_MCFA
# EM_MCFA(X,2,2)
