# Copyright 2024-2025 DALIA authors. All rights reserved.
from dataclasses import dataclass

import numpy as np

from dalia import ArrayLike, backend_flags, comm_rank
from dalia.utils.gpu_utils import get_array_module_name, get_device, get_host

if backend_flags["mpi_avail"]:
    from mpi4py import MPI

if backend_flags["cupy_avail"]:
    import cupy as cp


@dataclass
class DummyCommunicator:
    """Communicator class to handle MPI communication when
    MPI is not available.
    """

    size: int = 1
    rank: int = 0


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


def synchronize(comm):
    """
    Synchronize all processes within the given communication group.

    Parameters:
    -----------
    comm, optional:
        The communication group to synchronize. Default is MPI.COMM_WORLD.
    """
    if backend_flags["mpi_avail"]:
        if backend_flags["cupy_avail"]:
            cp.cuda.runtime.deviceSynchronize()
        comm.Barrier()


def synchronize_gpu():
    """
    Synchronize GPU operations.
    """
    if backend_flags["cupy_avail"]:
        cp.cuda.runtime.deviceSynchronize()


def allreduce(
    recvbuf: ArrayLike,
    comm,
    op: str = "sum",
    factor: int = 1,
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
    comm (CommunicatorType), optional:
        The communication group. Default is MPI.COMM_WORLD.
    """
    if backend_flags["mpi_avail"]:
        if (
            get_array_module_name(recvbuf) == "cupy"
            and not backend_flags["mpi_cuda_aware"]
        ):
            recvbuf_comm = get_host(recvbuf)
        else:
            recvbuf_comm = recvbuf

        if op == "sum":
            comm.Allreduce(MPI.IN_PLACE, recvbuf_comm, op=MPI.SUM)
            recvbuf_comm *= factor

        if (
            get_array_module_name(recvbuf) == "cupy"
            and not backend_flags["mpi_cuda_aware"]
        ):
            # Check if recvbuff is an array or a scalar
            if recvbuf.size > 1:
                recvbuf[:] = get_device(recvbuf_comm)
            else:
                return get_device(recvbuf_comm)

        if recvbuf.size == 1:
            return recvbuf_comm


def allgather(
    obj: ArrayLike,
    comm,
):
    if backend_flags["mpi_avail"]:
        if get_array_module_name(obj) == "cupy" and not backend_flags["mpi_cuda_aware"]:
            obj_comm = get_host(obj)
            gathered_objs = comm.allgather(obj_comm)
            # Convert gathered numpy arrays back to cupy arrays
            return [get_device(arr) for arr in gathered_objs]
        else:
            return comm.allgather(obj)


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
    comm (CommunicatorType), optional:
        The communication group. Default is MPI.COMM_WORLD.
    """

    if backend_flags["mpi_avail"]:
        if data.ndim == 0:
            comm.Bcast(data, root=root)
        else:
            comm.Bcast(data[:], root=root)


def get_active_comm(
    comm,
    n_parallelizable_evaluations: int,
    tag: str,
):
    """Return a CommunicatorType made out of all the processes that can be active
    given the number of parallelizable evaluations."""
    if backend_flags["mpi_avail"]:
        rank = comm.Get_rank()
        size = comm.size
        group_size = size // n_parallelizable_evaluations

        if size > n_parallelizable_evaluations and rank >= (
            group_size * n_parallelizable_evaluations
        ):
            # Remainder processes are excluded because they cannot be assigned to any group
            print(
                f"Rank: {rank} won't party tonight at '{tag}' level because you need a multiple of {n_parallelizable_evaluations} processes in the calling comm_group"
            )
            color = MPI.UNDEFINED
        else:
            color = 0

        active_comm = comm.Split(color, rank)
    else:
        active_comm = comm

    if color == MPI.UNDEFINED:
        exit()

    return active_comm


def smartsplit(
    comm,
    n_parallelizable_evaluations: int,
    tag: str,
    min_group_size: int = 1,
) -> tuple:
    if backend_flags["mpi_avail"]:
        if comm.size < min_group_size:
            raise ValueError(
                f"Initial CommunicatorType size must be at least {min_group_size} to fulfill the split requirements."
            )

        # Checks for compatibility of given comm sizes
        min_comm = get_active_comm(comm, min_group_size, tag="minimum_comm")
        active_comm = get_active_comm(
            min_comm, n_parallelizable_evaluations * min_group_size, tag
        )
        rank = active_comm.Get_rank()
        size = active_comm.size

        # print(f"Rank: {rank}, size: {size} at '{tag}' level", flush=True)

        # Compute the group size, given its minimum
        group_size = size // n_parallelizable_evaluations
        if group_size < min_group_size:
            group_size = min_group_size

        # Split the CommunicatorType
        color_new_group = rank // group_size
        key_new_group = rank
        comm_new_group = active_comm.Split(color_new_group, key_new_group)
        # print(f"FLG2: new group rank: {comm_new_group.rank}, new group size: {comm_new_group.size} at '{tag}' level", flush=True)
    else:
        active_comm = comm
        comm_new_group = comm
        color_new_group = 0

    return active_comm, comm_new_group, color_new_group

def check_vector_consistency(
    theta: ArrayLike,
    comm,
):
    """
    Check if all processes have the same theta.

    Parameters:
    -----------
    theta (ArrayLike):
        The theta to check.
    comm (CommunicatorType), optional:
        The communication group. Default is MPI.COMM_WORLD.
    """

    synchronize(comm = comm)

    theta_ref = theta.copy()
    bcast(theta_ref, root=0, comm=comm)

    if backend_flags["cupy_avail"]:
        if (
            get_array_module_name(theta) == "cupy"
        ):
            norm_diff = cp.linalg.norm(theta - theta_ref)
        else:
            norm_diff = np.linalg.norm(theta - theta_ref)
    else:
        norm_diff = np.linalg.norm(theta - theta_ref)

    if norm_diff > 1e-10:
        # Print indices and values where theta and theta_ref differ
        if backend_flags["cupy_avail"]:
            if (
                get_array_module_name(theta) == "cupy"
            ):
                diff_indices = cp.where(theta != theta_ref)[0]
                for idx in diff_indices:
                    print(f"Process {comm.Get_rank()} difference at index {idx}: theta_ref={theta_ref[idx]}, theta={theta[idx]}")
            else:
                diff_indices = np.where(theta != theta_ref)[0]
                for idx in diff_indices:
                    print(f"Process {comm.Get_rank()} difference at index {idx}: theta_ref={theta_ref[idx]}, theta={theta[idx]}")
                    
        else:
            diff_indices = np.where(theta != theta_ref)[0]
            for idx in diff_indices:
                print(f"Process {comm.Get_rank()} difference at index {idx}: theta_ref={theta_ref[idx]}, theta={theta[idx]}")
        raise ValueError(
            f"Process {comm.Get_rank()} has a different theta than the reference process."
            f" Expected: {theta_ref}, but got:  {theta}. diff = {norm_diff:.4e}"
            f""
        )
