import numpy as np
import scipy.sparse as sp
from scipy.sparse import diags

if __name__ == "__main__":

    np.random.seed(5)

    # epsilon N(0, (1-h^2)I), 0 < h^2 < 1,
    # beta prior on h^2

    # \beta ~ N(0, h^2 \Phi)
    # dim(Z_i) = (M,1)
    # alpha ~ N(0, \sigma_a^2 I), \alpha \in R^m
    # \sigma_a^2 large & fixed

    # dim(\Phi) = (b,b)

    no = 1000  # number of observations
    b = 2  # number of latent variables (number of features)
    m = 2  # number of annotations per feature

    # generate random Z -> needs to be loaded with the model
    z = np.random.rand(b, m)
    np.save("inputs_brainiac/z.npy", z)

    # Generate random h^2 with a Beta prior defined on the interval [0, 1]
    # TODO: how to estimate alpha_beta and beta_beta?
    # alpha_beta = 5.0
    # beta_beta = 1.0
    #h2 = np.random.beta(alpha_beta, beta_beta)
    h2 = 0.95
    print("h2: ", h2)
    
    # \sigma_a^2: large and fixed
    sigma_a2 = 1
    print("sigma_a2: ", sigma_a2)

    # sample alpha from N(0, \sigma_a^2 I)
    alpha = np.random.normal(2, np.sqrt(sigma_a2), (m, 1))
    #alpha = np.ones((m, 1))
    # print(alpha)

    theta_original = np.concatenate(([h2], alpha.flatten()))
    print(theta_original)

    # save original hyperparameters
    np.save("inputs_brainiac/theta_original.npy", theta_original)

    # \Phi = 1 / \sum_k=1^B exp(Z^k \alpha) * diag(exp(Z_1 \alpha), exp(Z_2 \alpha), ... )
    print("z : ", z)
    print("alpha : ", alpha)
    exp_Z_alpha = np.exp(z @ alpha)
    # print(exp_Z_alpha)
    sum_exp_Z_alpha = np.sum(exp_Z_alpha)
    print(sum_exp_Z_alpha)

    normalized_exp_Z_alpha = exp_Z_alpha / sum_exp_Z_alpha
    print("normalized_exp(Z*alpha): ", normalized_exp_Z_alpha)

    h2_phi = h2 * normalized_exp_Z_alpha.flatten()
    Qprior = diags(1 / h2_phi)
    print("Qprior: \n", Qprior.toarray())

    # save Qprior as a sparse matrix
    sp.save_npz("inputs_brainiac/Qprior_original.npz", Qprior)

    # sample full model: Y = a \beta + \epsilon
    # X random covariates of dimension (no, b)
    a = np.random.rand(no, b)
    # np.save("a.npy", a)
    a_sp = sp.csc_matrix(a)
    sp.save_npz("inputs_brainiac/a.npz", a_sp)

    # beta ~ N(0, (h^2 \Phi)^-1)
    var = 1 / Qprior.diagonal()
    print(var)
    beta = np.random.normal(0, np.sqrt(var)).reshape(b, 1)
    np.save("inputs_brainiac/beta_original.npy", beta.flatten())
    # print(beta)

    # beta regression parameters with
    eps = np.random.normal(0, np.sqrt(1 - h2), (no, 1))
    y = a @ beta + eps
    np.save("y.npy", y)

    # construct Qconditional
    Qconditional = Qprior + 1 / (1 - h2) * a_sp.T @ a_sp
    sp.save_npz("inputs_brainiac/Qconditional_original.npz", Qconditional)

    # recover beta
    # beta_initial = beta
    # grad_y = - 1 / (1 - h2) * (a @ beta - y)
    # information_vector = -1 * Qprior @ beta + a_sp.T @ grad_y

    beta_initial = np.zeros((b, 1))
    grad_y = - 1 / (1 - h2) * (-y)
    information_vector = a_sp.T @ grad_y   

    beta_recovered = beta_initial + np.linalg.solve(Qconditional.toarray(), information_vector)
    print("beta recovered: ", beta_recovered.flatten())
    print("beta original : ", beta.flatten())
    print("norm(diff) : ", np.linalg.norm(beta_recovered - beta))