import os
import sys

import numpy as np
import scipy.sparse as sp

from scipy.stats import multivariate_normal, poisson

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":

    n = 5

    s2 = 1
    phi = np.sqrt(0.5)
    denom = s2 * (1 - phi**2)

    theta1 = np.exp(s2)

    diag = [(1 + phi**2) / denom] * n
    diag[0] = diag[-1] = 1 / denom
    off_diag = [-phi / denom] * (n - 1)

    # print(diag)
    # print(off_diag)

    Q = sp.diags([diag, off_diag, off_diag], [0, -1, 1])
    Cov = np.linalg.inv(Q.toarray())

    # print(Q.toarray())
    # print(np.linalg.inv(Q.toarray()))
    # print(Q.toarray() @ np.linalg.inv(Q.toarray()))

    ##############################################################

    mv = multivariate_normal(mean=np.zeros(n), cov=Cov, seed=3)

    intercept = 2
    eta = mv.rvs() + intercept

    print(eta)

    E = [1] * n
    y = poisson.rvs(E * np.exp(eta), random_state=3)

    print(y)
