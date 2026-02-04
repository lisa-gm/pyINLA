import cupy as cp

cp.random.seed(63)

if __name__ == "__main__":
    N = 100

    A = cp.random.rand(N, N).astype(cp.float64)
    B = cp.random.rand(N, N).astype(cp.float64)

    # 1. Perform matrix multiplication
    C = A @ B

    # 2. Perform a cholesky decomposition
    C = C @ C.T + N * cp.eye(N)  # Make it symmetric positive definite
    L = cp.linalg.cholesky(C)

    # 3. Solve a linear system
    b = cp.random.rand(N).astype(cp.float64)
    x = cp.linalg.solve(C, b)

    print(x[:10])
