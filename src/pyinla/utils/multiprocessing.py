# Copyright 2024 pyINLA authors. All rights reserved.

from pyinla import MPI_AVAILABLE, ArrayLike

if MPI_AVAILABLE:
    from mpi4py import MPI


def synchronize(comm: MPI.Comm = MPI.COMM_WORLD):
    """
    Synchronize all processes within the given communication group.

    Parameters:
    -----------
    comm (MPI.Comm), optional:
        The communication group to synchronize. Default is MPI.COMM_WORLD.
    """
    if MPI_AVAILABLE:
        comm.Barrier()


def allreduce(
    sendbuf: ArrayLike,
    recvbuf: ArrayLike,
    op: str = "sum",
    comm: MPI.Comm = MPI.COMM_WORLD,
):
    """
    Perform a reduction operation across all processes within the given communication group.

    Parameters:
    -----------
    sendbuf (ArrayLike):
        The buffer to send.
    recvbuf (ArrayLike):
        The buffer to receive.
    op ():
        The reduction operation.
    comm (MPI.Comm), optional:
        The communication group. Default is MPI.COMM_WORLD.
    """
    if MPI_AVAILABLE:
        if op == "sum":
            comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)


def bcast(
    data: ArrayLike,
    root: int = 0,
    comm: MPI.Comm = MPI.COMM_WORLD,
):
    """
    Broadcast data from the root process to all other processes within the given communication group.

    Parameters:
    -----------
    data (ArrayLike):
        The data to broadcast.
    root (int), optional:
        The root process. Default is 0.
    comm (MPI.Comm), optional:
        The communication group. Default is MPI.COMM_WORLD.
    """
    if MPI_AVAILABLE:
        comm.Bcast(data, root=root)
