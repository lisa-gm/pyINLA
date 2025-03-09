# Copyright 2024-2025 pyINLA authors. All rights reserved.

from pyinla import ArrayLike, backend_flags, comm_rank

if backend_flags["mpi_avail"]:
    from mpi4py import MPI


def print_msg(*args, **kwargs):
    """
    Print a message from a single process.

    Parameters:
    -----------
    *args:
        Variable length argument list.
    **kwargs:
        Arbitrary keyword arguments.
    """
    if comm_rank == 0:
        print(*args, **kwargs)


def synchronize(comm=None):
    """
    Synchronize all processes within the given communication group.

    Parameters:
    -----------
    comm, optional:
        The communication group to synchronize. Default is MPI.COMM_WORLD.
    """
    if backend_flags["mpi_avail"]:
        if comm is None:
            comm = MPI.COMM_WORLD
        comm.Barrier()


def allreduce(
    recvbuf: ArrayLike,
    op: str = "sum",
    comm: MPI.Comm = None,
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
    if backend_flags["mpi_avail"]:
        if comm is None:
            comm = MPI.COMM_WORLD
        if op == "sum":
            comm.Allreduce(MPI.IN_PLACE, recvbuf, op=MPI.SUM)


def bcast(
    data: ArrayLike,
    root: int = 0,
    comm=None,
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
    if backend_flags["mpi_avail"]:
        if comm is None:
            comm = MPI.COMM_WORLD
        comm.Bcast(data, root=root)


def smartsplit(
    comm: MPI.Comm,
    n_parallelizable_evaluations: int,  
    tag: str,
) -> tuple[MPI.Comm, int]:
    if backend_flags["mpi_avail"]:
        rank = comm.Get_rank()
        size = comm.Get_size()

        group_size = size // n_parallelizable_evaluations
        if size < n_parallelizable_evaluations:
            # Less processes than n_parallelizable_evaluations
            color_new_group = rank
        else:
            # More (or =) processes than n_parallelizable_evaluations
            color_new_group = rank // group_size
        key_new_group = rank
        comm_new_group = comm.Split(color_new_group, key_new_group)

        if size > n_parallelizable_evaluations and rank >= (group_size * n_parallelizable_evaluations):
                # Remainder processes are terminated because cannot be assigned to any group
                print(f"Rank: {rank} terminated at the '{tag}' level because you need a multiple of {n_parallelizable_evaluations} processes in the calling comm_group")
                exit()
    else:
        comm_new_group = comm
        color_new_group = 0

    return comm_new_group, color_new_group