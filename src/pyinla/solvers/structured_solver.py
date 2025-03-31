# Copyright 2024-2025 pyINLA authors. All rights reserved.

from warnings import warn

from cupy.cuda import nvtx

from pyinla import NDArray, sp, xp
from pyinla.configs.pyinla_config import SolverConfig
from pyinla.core.solver import Solver
from pyinla.utils import print_msg

try:
    from serinv.algs import pobtaf, pobtas, pobtasi, pobtf, pobts, pobtsi
except ImportError as e:
    warn(f"The serinv package is required to use the SerinvSolver: {e}")


class SerinvSolver(Solver):
    """Serinv Solver class."""

    def __init__(
        self,
        config: SolverConfig,
        **kwargs,
    ) -> None:
        """Initializes the SerinV solver."""
        super().__init__(config)

        self.diagonal_blocksize: int = kwargs.get("diagonal_blocksize", None)
        if self.diagonal_blocksize is None:
            raise KeyError("Missing required keyword argument: 'diagonal_blocksize'")

        self.arrowhead_blocksize: int = kwargs.get("arrowhead_blocksize", 0)

        self.n_diag_blocks: int = kwargs.get("n_diag_blocks", None)
        if self.n_diag_blocks is None:
            raise KeyError("Missing required keyword argument: 'n_diag_blocks'")

        # --- Initialize memory for BTA-array storage
        self.A_diagonal_blocks: NDArray = xp.empty(
            (self.n_diag_blocks, self.diagonal_blocksize, self.diagonal_blocksize),
            dtype=xp.float64,
        )
        self.A_lower_diagonal_blocks: NDArray = xp.empty(
            (
                self.n_diag_blocks - 1,
                self.diagonal_blocksize,
                self.diagonal_blocksize,
            ),
            dtype=xp.float64,
        )

        self.A_arrow_bottom_blocks: NDArray = None
        self.A_arrow_tip_block: NDArray = None

        if self.arrowhead_blocksize > 0:
            self.A_arrow_bottom_blocks = xp.empty(
                (self.n_diag_blocks, self.arrowhead_blocksize, self.diagonal_blocksize),
                dtype=xp.float64,
            )
            self.A_arrow_tip_block = xp.empty(
                (self.arrowhead_blocksize, self.arrowhead_blocksize),
                dtype=xp.float64,
            )

        # Print the allocated memory for the BTA-array
        total_bytes: int = (
            self.A_diagonal_blocks.nbytes
            + self.A_lower_diagonal_blocks.nbytes
            + self.A_arrow_bottom_blocks.nbytes
            + self.A_arrow_tip_block.nbytes
        )
        total_gb: int = total_bytes / (1024**3)
        print_msg(f"Allocated memory for SerinvSolver: {total_gb:.2f} GB", flush=True)

    def cholesky(self, A: sp.sparse.spmatrix) -> None:
        """Compute Cholesky factor of input matrix."""

        nvtx.RangePush("spmatrix_to_structured")
        self._spmatrix_to_structured(A)
        nvtx.RangePop()

        nvtx.RangePush("pobtaf")
        if self.A_arrow_bottom_blocks is not None:
            pobtaf(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                self.A_arrow_bottom_blocks,
                self.A_arrow_tip_block,
            )
        else:
            pobtf(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
            )
        nvtx.RangePop()

    def solve(self, rhs: NDArray) -> NDArray:
        """Solve linear system using Cholesky factor."""

        if self.A_arrow_bottom_blocks is not None:
            pobtas(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                self.A_arrow_bottom_blocks,
                self.A_arrow_tip_block,
                rhs,
                trans="N",
            )
            pobtas(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                self.A_arrow_bottom_blocks,
                self.A_arrow_tip_block,
                rhs,
                trans="C",
            )
        else:
            pobts(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                rhs,
                trans="N",
            )
            pobts(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                rhs,
                trans="C",
            )

        return rhs

    def logdet(self) -> float:
        """Compute logdet of input matrix using Cholesky factor."""

        logdet: float = 0.0

        for i in range(self.n_diag_blocks):
            logdet += xp.sum(xp.log(self.A_diagonal_blocks[i].diagonal()))

        logdet += xp.sum(xp.log(self.A_arrow_tip_block.diagonal()))

        return 2 * logdet

    def selected_inversion(self, A: sp.sparse.spmatrix, **kwargs) -> None:
        """Compute selected inversion of input matrix using Cholesky factor."""

        self._spmatrix_to_structured(A)

        if self.A_arrow_bottom_blocks is not None:
            pobtaf(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                self.A_arrow_bottom_blocks,
                self.A_arrow_tip_block,
            )

            pobtasi(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                self.A_arrow_bottom_blocks,
                self.A_arrow_tip_block,
            )
        else:
            self.bt_dense_to_arrays(A, self.diagonal_blocksize, self.n_diag_blocks)

            pobtf(self.A_diagonal_blocks, self.A_lower_diagonal_blocks)

            pobtsi(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
            )

    def spmatrix_to_structured(
        self,
        A: sp.sparse.spmatrix,
    ) -> None:
        """Map sp.spmatrix to BT or BTA."""
        # About 3x faster...
        A_csc = sp.sparse.csc_matrix(A)

        self.A_diagonal_blocks[:] = 0.0
        self.A_lower_diagonal_blocks[:] = 0.0
        self.A_arrow_bottom_blocks[:] = 0.0
        self.A_arrow_tip_block[:] = 0.0

        for i in range(self.n_diag_blocks):
            block_slice = A_csc[
                i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
            ].tocoo()
            self.A_diagonal_blocks[i][
                block_slice.row, block_slice.col
            ] = block_slice.data

            if i < self.n_diag_blocks - 1:
                block_slice = A_csc[
                    (i + 1)
                    * self.diagonal_blocksize : (i + 2)
                    * self.diagonal_blocksize,
                    i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                ].tocoo()
                self.A_lower_diagonal_blocks[i][
                    block_slice.row, block_slice.col
                ] = block_slice.data

            block_slice = A_csc[
                -self.arrowhead_blocksize :,
                i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
            ].tocoo()
            self.A_arrow_bottom_blocks[i][
                block_slice.row, block_slice.col
            ] = block_slice.data

        block_slice = A_csc[
            -self.arrowhead_blocksize :, -self.arrowhead_blocksize :
        ].tocoo()
        self.A_arrow_tip_block[block_slice.row, block_slice.col] = block_slice.data

    def bta_dense_to_arrays(
        self,
        A: NDArray,
        diagonal_blocksize: int,
        arrowhead_blocksize: int,
        n_diag_blocks: int,
    ):
        """Converts a block tridiagonal arrowhead matrix from a dense representation to arrays of blocks.

        Parameters
        ----------
        A : ArrayLike
            Dense representation of the block tridiagonal arrowhead matrix.
        diagonal_blocksize : int
            Size of the diagonal blocks.
        arrowhead_blocksize : int
            Size of the arrowhead blocks.
        n_diag_blocks : int
            Number of diagonal blocks.

        Returns
        -------
        A_diagonal_blocks : ArrayLike
            The diagonal blocks of the block tridiagonal with arrowhead matrix.
        A_lower_diagonal_blocks : ArrayLike
            The lower diagonal blocks of the block tridiagonal with arrowhead matrix.
        A_upper_diagonal_blocks : ArrayLike
            The upper diagonal blocks of the block tridiagonal with arrowhead matrix.
        A_lower_arrow_blocks : ArrayLike
            The arrow bottom blocks of the block tridiagonal with arrowhead matrix.
        A_upper_arrow_blocks : ArrayLike
            The arrow right blocks of the block tridiagonal with arrowhead matrix.
        A_arrow_tip_block : ArrayLike
            The arrow tip block of the block tridiagonal with arrowhead matrix.

        Notes
        -----
        - The BTA matrix in array representation will be returned according
        to the array module of the input matrix, A.
        """
        # xp, _ = _get_module_from_array(A)

        A_diagonal_blocks = xp.zeros(
            (n_diag_blocks, diagonal_blocksize, diagonal_blocksize),
            dtype=A.dtype,
        )

        A_lower_diagonal_blocks = xp.zeros(
            (n_diag_blocks - 1, diagonal_blocksize, diagonal_blocksize),
            dtype=A.dtype,
        )
        A_upper_diagonal_blocks = xp.zeros(
            (n_diag_blocks - 1, diagonal_blocksize, diagonal_blocksize),
            dtype=A.dtype,
        )

        A_lower_arrow_blocks = xp.zeros(
            (n_diag_blocks, arrowhead_blocksize, diagonal_blocksize),
            dtype=A.dtype,
        )

        A_upper_arrow_blocks = xp.zeros(
            (n_diag_blocks, diagonal_blocksize, arrowhead_blocksize),
            dtype=A.dtype,
        )

        A_arrow_tip_block = xp.zeros(
            (arrowhead_blocksize, arrowhead_blocksize),
            dtype=A.dtype,
        )

        for i in range(n_diag_blocks):
            A_diagonal_blocks[i] = A[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            ]
            if i > 0:
                A_lower_diagonal_blocks[i - 1] = A[
                    i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                    (i - 1) * diagonal_blocksize : i * diagonal_blocksize,
                ]
            if i < n_diag_blocks - 1:
                A_upper_diagonal_blocks[i] = A[
                    i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                    (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
                ]

            A_lower_arrow_blocks[i] = A[
                -arrowhead_blocksize:,
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            ]

            A_upper_arrow_blocks[i] = A[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                -arrowhead_blocksize:,
            ]

        A_arrow_tip_block[:] = A[-arrowhead_blocksize:, -arrowhead_blocksize:]

        return (
            A_diagonal_blocks,
            A_lower_diagonal_blocks,
            A_upper_diagonal_blocks,
            A_lower_arrow_blocks,
            A_upper_arrow_blocks,
            A_arrow_tip_block,
        )

    def bt_dense_to_arrays(
        A: NDArray,
        diagonal_blocksize: int,
        n_diag_blocks: int,
    ):
        """Converts a block tridiagonal arrowhead matrix from a dense representation to arrays of blocks.

        Parameters
        ----------
        A : ArrayLike
            Dense representation of the block tridiagonal arrowhead matrix.
        diagonal_blocksize : int
            Size of the diagonal blocks.
        n_diag_blocks : int
            Number of diagonal blocks.

        Returns
        -------
        A_diagonal_blocks : ArrayLike
            The diagonal blocks of the block tridiagonal with arrowhead matrix.
        A_lower_diagonal_blocks : ArrayLike
            The lower diagonal blocks of the block tridiagonal with arrowhead matrix.
        A_upper_diagonal_blocks : ArrayLike
            The upper diagonal blocks of the block tridiagonal with arrowhead matrix.

        Notes
        -----
        - The BT matrix in array representation will be returned according
        to the array module of the input matrix, A.
        """

        A_diagonal_blocks = xp.zeros(
            (n_diag_blocks, diagonal_blocksize, diagonal_blocksize),
            dtype=A.dtype,
        )

        A_lower_diagonal_blocks = xp.zeros(
            (n_diag_blocks - 1, diagonal_blocksize, diagonal_blocksize),
            dtype=A.dtype,
        )
        A_upper_diagonal_blocks = xp.zeros(
            (n_diag_blocks - 1, diagonal_blocksize, diagonal_blocksize),
            dtype=A.dtype,
        )

        for i in range(n_diag_blocks):
            A_diagonal_blocks[i] = A[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            ]
            if i > 0:
                A_lower_diagonal_blocks[i - 1] = A[
                    i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                    (i - 1) * diagonal_blocksize : i * diagonal_blocksize,
                ]
            if i < n_diag_blocks - 1:
                A_upper_diagonal_blocks[i] = A[
                    i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                    (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
                ]

        return (
            A_diagonal_blocks,
            A_lower_diagonal_blocks,
            A_upper_diagonal_blocks,
        )

    def _bta_arrays_to_dense(
        self,
        A_diagonal_blocks: NDArray,
        A_lower_diagonal_blocks: NDArray,
        A_upper_diagonal_blocks: NDArray,
        A_arrow_bottom_blocks: NDArray,
        A_arrow_right_blocks: NDArray,
        A_arrow_tip_block: NDArray,
    ):
        """Converts arrays of blocks to a block tridiagonal arrowhead matrix in a dense representation.

        Parameters
        ----------
        A_diagonal_blocks : NDArray
            The diagonal blocks of the block tridiagonal with arrowhead matrix.
        A_lower_diagonal_blocks : NDArray
            The lower diagonal blocks of the block tridiagonal with arrowhead matrix.
        A_upper_diagonal_blocks : NDArray
            The upper diagonal blocks of the block tridiagonal with arrowhead matrix.
        A_arrow_bottom_blocks : NDArray
            The arrow bottom blocks of the block tridiagonal with arrowhead matrix.
        A_arrow_right_blocks : NDArray
            The arrow right blocks of the block tridiagonal with arrowhead matrix.
        A_arrow_tip_block : NDArray
            The arrow tip block of the block tridiagonal with arrowhead matrix.

        Returns
        -------
        A : NDArray
            Dense representation of the block tridiagonal arrowhead matrix.

        Notes
        -----
        - The BTA matrix in array representation will be returned according
        to the array module of the input matrix, A_diagonal_blocks.
        """
        # xp, _ = _get_module_from_array(A_diagonal_blocks)

        diagonal_blocksize = A_diagonal_blocks.shape[1]
        arrowhead_blocksize = A_arrow_bottom_blocks.shape[1]
        n_diag_blocks = A_diagonal_blocks.shape[0]

        A = xp.zeros(
            (
                diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
                diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
            ),
            dtype=A_diagonal_blocks.dtype,
        )
        print("_bta_arrays_to_dense: dim(A): ", A.shape)
        print("dim(A_diagonal_blocks): ", A_diagonal_blocks.shape)
        print("dim(A_lower_diagonal_blocks): ", A_lower_diagonal_blocks.shape)
        print("dim(A_upper_diagonal_blocks): ", A_upper_diagonal_blocks.shape)
        print("dim(A_arrow_bottom_blocks): ", A_arrow_bottom_blocks.shape)
        print("dim(A_arrow_right_blocks): ", A_arrow_right_blocks.shape)
        print("dim(A_arrow_tip_block): ", A_arrow_tip_block.shape)

        for i in range(n_diag_blocks):
            A[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            ] = A_diagonal_blocks[i]
            if i > 0:
                A[
                    i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                    (i - 1) * diagonal_blocksize : i * diagonal_blocksize,
                ] = A_lower_diagonal_blocks[i - 1]
            if i < n_diag_blocks - 1:
                A[
                    i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                    (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
                ] = A_upper_diagonal_blocks[i]

            A[
                -arrowhead_blocksize:,
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            ] = A_arrow_bottom_blocks[i]

            A[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                -arrowhead_blocksize:,
            ] = A_arrow_right_blocks[i]

        A[-arrowhead_blocksize:, -arrowhead_blocksize:] = A_arrow_tip_block[:]

        return A

    def _structured_to_spmatrix(self, A: sp.sparse.spmatrix) -> sp.sparse.spmatrix:
        """Map BT or BTA matrix to sp.spmatrix using sparsity pattern provided in A."""

        # A is assumed to be symmetric, only use lower triangular part
        B = sp.sparse.csc_matrix(sp.sparse.tril(sp.sparse.csc_matrix(A)))

        for col in range(A.shape[1]):
            start = B.indptr[col]
            end = B.indptr[col + 1]

            rows = B.indices[start:end]
            for row in rows:
                block_i = row // self.diagonal_blocksize
                block_j = col // self.diagonal_blocksize
                local_i = row % self.diagonal_blocksize
                local_j = col % self.diagonal_blocksize

                if block_i == block_j and block_i < self.n_diag_blocks:
                    B[row, col] = self.A_diagonal_blocks[block_i, local_i, local_j]

                elif block_i == block_j + 1 and block_j < self.n_diag_blocks - 1:
                    B[row, col] = self.A_lower_diagonal_blocks[
                        block_j, local_i, local_j
                    ]
                # arrowhead
                elif block_i == self.n_diag_blocks and block_j < self.n_diag_blocks:
                    B[row, col] = self.A_arrow_bottom_blocks[block_j, local_i, local_j]
                elif block_i == self.n_diag_blocks and block_j == self.n_diag_blocks:
                    B[row, col] = self.A_arrow_tip_block[local_i, local_j]

        # symmetrize B
        B = B + sp.sparse.tril(B, k=-1).T

        return B


def diagonally_dominant_bta(
    n: int,
    b: int,
    a: int,
    direction: str = "downward",
    device_arr: bool = False,
    symmetric: bool = True,
    seed: int = 63,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    xp.random.seed(seed)

    A_diagonal_blocks = xp.random.rand(n, b, b)
    A_lower_diagonal_blocks = xp.random.rand(n - 1, b, b)
    A_upper_diagonal_blocks = xp.random.rand(n - 1, b, b)
    A_arrow_tip_block = xp.random.rand(a, a)

    if direction == "downward" or direction == "down-middleward":
        A_lower_arrow_blocks = xp.random.rand(n, a, b)
        A_upper_arrow_blocks = xp.random.rand(n, b, a)
    elif direction == "upward" or direction == "up-middleward":
        A_lower_arrow_blocks = xp.random.rand(n, b, a)
        A_upper_arrow_blocks = xp.random.rand(n, a, b)

    if symmetric:
        for n_i in range(n):
            A_diagonal_blocks[n_i] = A_diagonal_blocks[n_i] + A_diagonal_blocks[n_i].T

            if n_i < n - 1:
                A_upper_diagonal_blocks[n_i] = A_lower_diagonal_blocks[n_i].T

            A_upper_arrow_blocks[n_i] = A_lower_arrow_blocks[n_i].T

        A_arrow_tip_block[:] = A_arrow_tip_block[:] + A_arrow_tip_block[:].T

    A_arrow_tip_block[:] += xp.diag(xp.sum(xp.abs(A_arrow_tip_block[:]), axis=1))

    if direction == "downward" or direction == "down-middleward":
        for n_i in range(n):
            A_diagonal_blocks[n_i] += xp.diag(
                xp.sum(xp.abs(A_diagonal_blocks[n_i]), axis=1)
            )

            if n_i > 0:
                A_diagonal_blocks[n_i] += xp.diag(
                    xp.sum(xp.abs(A_lower_diagonal_blocks[n_i - 1]), axis=1)
                )

            if n_i < n - 1:
                A_diagonal_blocks[n_i] += xp.diag(
                    xp.sum(xp.abs(A_upper_diagonal_blocks[n_i]), axis=1)
                )

            A_diagonal_blocks[n_i] += xp.diag(
                xp.sum(xp.abs(A_upper_arrow_blocks[n_i]), axis=1)
            )

            A_arrow_tip_block[:] += xp.diag(
                xp.sum(xp.abs(A_lower_arrow_blocks[n_i]), axis=1)
            )

    elif direction == "upward" or direction == "up-middleward":
        for n_i in range(n - 1, -1, -1):
            A_diagonal_blocks[n_i] += xp.diag(
                xp.sum(xp.abs(A_diagonal_blocks[n_i]), axis=1)
            )

            if n_i < n - 1:
                A_diagonal_blocks[n_i] += xp.diag(
                    xp.sum(xp.abs(A_upper_diagonal_blocks[n_i]), axis=1)
                )

            if n_i > 0:
                A_diagonal_blocks[n_i] += xp.diag(
                    xp.sum(xp.abs(A_lower_diagonal_blocks[n_i - 1]), axis=1)
                )

            A_diagonal_blocks[n_i] += xp.diag(
                xp.sum(xp.abs(A_lower_arrow_blocks[n_i]), axis=1)
            )

            A_arrow_tip_block[:] += xp.diag(
                xp.sum(xp.abs(A_upper_arrow_blocks[n_i]), axis=1)
            )

    return (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_lower_arrow_blocks,
        A_upper_diagonal_blocks,
        A_upper_arrow_blocks,
        A_arrow_tip_block,
    )


def bta_to_dense(
    A_diagonal_blocks: NDArray,
    A_lower_diagonal_blocks: NDArray,
    A_lower_arrow_blocks: NDArray,
    A_upper_diagonal_blocks: NDArray,
    A_upper_arrow_blocks: NDArray,
    A_arrow_tip_block: NDArray,
    direction: str = "downward",
) -> NDArray:
    n = A_diagonal_blocks.shape[0]
    b = A_diagonal_blocks.shape[1]
    a = A_arrow_tip_block.shape[0]

    A = xp.zeros((n * b + a, n * b + a), dtype=A_diagonal_blocks.dtype)

    if direction == "downward" or direction == "down-middleward":
        for n_i in range(n):
            A[n_i * b : (n_i + 1) * b, n_i * b : (n_i + 1) * b] = A_diagonal_blocks[n_i]
            if n_i > 0:
                A[
                    n_i * b : (n_i + 1) * b, (n_i - 1) * b : n_i * b
                ] = A_lower_diagonal_blocks[n_i - 1]
            if n_i < n - 1:
                A[
                    n_i * b : (n_i + 1) * b, (n_i + 1) * b : (n_i + 2) * b
                ] = A_upper_diagonal_blocks[n_i]
            A[n_i * b : (n_i + 1) * b, -a:] = A_upper_arrow_blocks[n_i]
            A[-a:, n_i * b : (n_i + 1) * b] = A_lower_arrow_blocks[n_i]
        A[-a:, -a:] = A_arrow_tip_block[:]

    if direction == "upward" or direction == "up-middleward":
        for n_i in range(n - 1, -1, -1):
            A[
                n_i * b + a : (n_i + 1) * b + a, n_i * b + a : (n_i + 1) * b + a
            ] = A_diagonal_blocks[n_i]
            if n_i > 0:
                A[
                    n_i * b + a : (n_i + 1) * b + a, (n_i - 1) * b + a : n_i * b + a
                ] = A_lower_diagonal_blocks[n_i - 1]
            if n_i < n - 1:
                A[
                    n_i * b + a : (n_i + 1) * b + a,
                    (n_i + 1) * b + a : (n_i + 2) * b + a,
                ] = A_upper_diagonal_blocks[n_i]
            A[n_i * b + a : (n_i + 1) * b + a, :a] = A_lower_arrow_blocks[n_i]
            A[:a, n_i * b + a : (n_i + 1) * b + a] = A_upper_arrow_blocks[n_i]
        A[:a, :a] = A_arrow_tip_block[:]

    return A


# np.random.seed(41)
# path = os.path.dirname(__file__)


if __name__ == "__main__":
    diagonal_blocksize = 2
    arrowhead_blocksize = 1
    n_diag_blocks = 3

    # Define the required keyword arguments
    kwargs = {
        "diagonal_blocksize": diagonal_blocksize,
        "arrowhead_blocksize": arrowhead_blocksize,
        "n_diag_blocks": n_diag_blocks,
    }

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_lower_arrow_blocks,
        A_upper_diagonal_blocks,
        A_upper_arrow_blocks,
        A_arrow_tip_block,
    ) = diagonally_dominant_bta(n_diag_blocks, diagonal_blocksize, arrowhead_blocksize)

    A_dense = bta_to_dense(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_lower_arrow_blocks,
        A_upper_diagonal_blocks,
        A_upper_arrow_blocks,
        A_arrow_tip_block,
    )

    print("A_dense: \n", A_dense)

    # save spy(A_dense)
    # plt.figure()
    # plt.spy(A_dense)
    # plt.savefig("A_dense.png")

    # save as coo
    A_csc = sp.sparse.csc_matrix(A_dense)
    nnz_A = A_csc.nnz

    L = A_dense.copy()
    L = xp.linalg.cholesky(L)

    L_inv = sp.linalg.solve_triangular(L, xp.eye(L.shape[0]), lower=True)
    A_inv_ref = L_inv.T @ L_inv

    # Create a SolverConfig instance
    config = SolverConfig()

    # Create an instance of the SerinvSolver class
    solver = SerinvSolver(config, **kwargs)

    # Now you can use the solver instance
    print("Solver initialized successfully.")

    solver.cholesky(A_csc)

    # call selected inversion
    solver.selected_inversion(A_csc)

    # if arrowhead_blocksize > 0:
    Ainv = bta_to_dense(
        solver.A_diagonal_blocks,
        solver.A_lower_diagonal_blocks,
        solver.A_arrow_bottom_blocks,
        solver.A_lower_diagonal_blocks.transpose(0, 2, 1),
        solver.A_arrow_bottom_blocks.transpose(0, 2, 1),
        solver.A_arrow_tip_block,
    )

    print("A_inv[:6, :6]: \n", Ainv[-6:, -6:])

    print("A_inv_ref[:6, :6]: \n", A_inv_ref[-6:, -6:])
