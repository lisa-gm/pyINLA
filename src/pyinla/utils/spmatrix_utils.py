
from pyinla import sp, xp

def bdiag_tiling(
        sparse_blocks: list[sp.sparse.spmatrix],
    ) -> sp.sparse.spmatrix:
    """ Tile in a block-diagonal fashion the given sparse blocks.
    This is equivalent to the scipy.sparse.block_diag function but is needed
    as the latter is not supported in cupyx.scipy.sparse.

    return a COO matrix
    """
    # Get the shape of the blocks
    block_shapes = [block.shape for block in sparse_blocks]

    out_data = []
    out_row = []
    out_col = []

    for i, block in enumerate(sparse_blocks):
        coo_block = block.tocoo()

        out_data.append(coo_block.data)
        out_row.append(coo_block.row + sum([block_shape[0] for block_shape in block_shapes[:i]]))
        out_col.append(coo_block.col + sum([block_shape[1] for block_shape in block_shapes[:i]]))

    out_data = xp.concatenate(out_data)
    out_row = xp.concatenate(out_row)
    out_col = xp.concatenate(out_col)

    out = sp.sparse.coo_matrix((out_data, (out_row, out_col)), shape=(sum([block_shape[0] for block_shape in block_shapes]), sum([block_shape[1] for block_shape in block_shapes])))

    return out
