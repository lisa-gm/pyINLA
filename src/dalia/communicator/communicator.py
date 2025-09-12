# Copyright 2024-2025 DALIA authors. All rights reserved.

import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Generator, List, Optional, Tuple

import numpy as np

from dalia import backend_flags
from dalia.communicator.communicator_config import CommunicatorConfig

if backend_flags["cupy_avail"]:
    import cupy as cp

if backend_flags["mpi_avail"]:
    from mpi4py import MPI

    # Set the MPI operation mapping
    mpi_op = {
        "sum": MPI.SUM,
        "prod": MPI.PROD,
        "min": MPI.MIN,
        "max": MPI.MAX,
    }

    if backend_flags["nccl_avail"]:
        from cupy.cuda import nccl

        # Set the NCCL Datatype Mapping
        nccl_datatype = {
            np.float32: nccl.NCCL_FLOAT,
            cp.float32: nccl.NCCL_FLOAT,
            np.complex64: nccl.NCCL_FLOAT,
            cp.complex64: nccl.NCCL_FLOAT,
            np.float64: nccl.NCCL_DOUBLE,
            cp.float64: nccl.NCCL_DOUBLE,
            np.complex128: nccl.NCCL_DOUBLE,
            cp.complex128: nccl.NCCL_DOUBLE,
        }

        # Set the NCCL Operation Mapping
        nccl_op = {
            "sum": cp.cuda.nccl.NCCL_SUM,
            "prod": cp.cuda.nccl.NCCL_PROD,
            "min": cp.cuda.nccl.NCCL_MIN,
            "max": cp.cuda.nccl.NCCL_MAX,
        }


@dataclass
class CollectiveConfig:
    allreduce: str = "default"
    allgather: str = "default"
    bcast: str = "default"
    reduce: str = "default"
    reduce_scatter: str = "default"
    send_recv: str = "default"

    # Collectives that are not supported by NCCL
    allgatherv: str = "default"
    alltoall: str = "default"


class Communicator:
    """
    A unified communication interface supporting multiple parallel communication libraries.

    This class provides a high-level abstraction for parallel communication operations,
    supporting MPI (host and device), NCCL, and single-process modes. It automatically
    selects the most appropriate communication backend based on availability and
    configuration.

    Parameters
    ----------
    config : CommunicatorConfig, optional
        Configuration object specifying communication library preferences and settings.
        Default is CommunicatorConfig().
    communicator : MPI communicator, optional
        Base MPI communicator to use. If None, uses MPI.COMM_WORLD when MPI is available.
        Default is None (no communication library).

    Attributes
    ----------
    tag : str
        Identifier tag for this communicator instance.
    rank : int
        Rank of the current process in the communicator.
    size : int
        Total number of processes in the communicator.
    """

    def __init__(
        self,
        config: CommunicatorConfig = CommunicatorConfig(),
        communicator: Optional[Any] = None,
    ) -> None:
        self.tag = config.tag

        # Validate the availability of the communication library and set most performant default
        self._general_comm_lib = self._resolve_and_validate_comm_lib(
            comm_lib=config.comm_lib
        )
        self._collective_config: CollectiveConfig = (
            self._resolve_and_validate_collective_config(config)
        )

        # Initialize the communicators
        self._base_comm, self.rank, self.size = self._initialize_base_communicator(
            communicator
        )
        self._xccl_comm = self._initialize_xccl_communicator()

    def _resolve_and_validate_comm_lib(
        self,
        comm_lib: str,
        nccl_supported: bool = True,
    ) -> str:
        """
        Resolve 'default' comm_lib to the best available option and validate availability.

        Parameters
        ----------
        comm_lib : str
            Communication library specification. Can be 'default', 'nccl', 'device_mpi',
            'host_mpi', or 'none'.
        nccl_supported : bool, optional
            Whether NCCL is supported for this operation. Default is True.

        Returns
        -------
        str
            The resolved communication library name.

        Raises
        ------
        RuntimeError
            If the requested communication library is not available.
        ValueError
            If an unknown communication library is specified.
        """
        if comm_lib == "default":
            # Priority order: nccl > device_mpi > host_mpi > none
            if backend_flags["nccl_avail"] and nccl_supported:
                return "nccl"
            elif backend_flags["mpi_cuda_aware"]:
                return "device_mpi"
            elif backend_flags["mpi_avail"]:
                return "host_mpi"
            else:
                return "none"
        else:
            # Validate that the specified comm_lib is actually available
            self._validate_comm_lib_availability(comm_lib)
            return comm_lib

    def _validate_comm_lib_availability(self, comm_lib: str) -> None:
        """
        Validate that the specified communication library is available on the system.

        Parameters
        ----------
        comm_lib : str
            Communication library to validate. Must be one of 'nccl', 'device_mpi',
            'host_mpi', or 'none'.

        Raises
        ------
        RuntimeError
            If the requested communication library is not available on the system.
        ValueError
            If an unknown communication library is specified.
        """
        if comm_lib == "nccl":
            if not backend_flags["nccl_avail"]:
                raise RuntimeError(
                    "NCCL communication library was requested but is not available. "
                    "Ensure NCCL is installed and USE_NCCL=1 environment variable is set."
                )
        elif comm_lib == "device_mpi":
            if not backend_flags["mpi_cuda_aware"]:
                raise RuntimeError(
                    "CUDA-aware MPI was requested but is not available. "
                    "Ensure MPI is compiled with CUDA support and MPI_CUDA_AWARE=1 environment variable is set."
                )
        elif comm_lib == "host_mpi":
            if not backend_flags["mpi_avail"]:
                raise RuntimeError(
                    "MPI communication library was requested but is not available. "
                    "Ensure mpi4py is installed and MPI is properly configured."
                )
        elif comm_lib != "none":
            raise ValueError(
                f"Unknown communication library '{comm_lib}'. "
                "Valid options are: 'default', 'host_mpi', 'device_mpi', 'nccl', 'none'"
            )

    def _initialize_base_communicator(
        self, base_comm: Optional[Any] = None
    ) -> Tuple[Optional[Any], int, int]:
        """
        Initialize MPI communicator and attributes based on general comm lib.

        Parameters
        ----------
        base_comm : MPI communicator, optional
            Base MPI communicator to use. If None and MPI is available, uses MPI.COMM_WORLD.

        Returns
        -------
        tuple
            A 3-tuple containing (communicator, rank, size).
        """
        # Initialize MPI communicator and attributes based on general comm lib
        if self._general_comm_lib in ["host_mpi", "device_mpi", "nccl"]:
            # All these require MPI as a base layer
            base_comm = base_comm if base_comm is not None else MPI.COMM_WORLD
            rank = base_comm.rank
            size = base_comm.size
        else:
            # No communication (single process mode)
            base_comm = None
            rank = 0
            size = 1

        return base_comm, rank, size

    def _initialize_xccl_communicator(self) -> Optional[Any]:
        """
        Initialize NCCL communicator if needed.

        Returns
        -------
        NcclCommunicator or None
            NCCL communicator instance if NCCL is being used, None otherwise.
        """
        if (
            backend_flags["nccl_avail"]
            and self._general_comm_lib == "nccl"
            or any(lib in self._collective_config.__dict__.values() for lib in ["nccl"])
        ):
            if self.rank == 0:
                nccl_id = cp.cuda.nccl.get_unique_id()
                self._base_comm.bcast(nccl_id, root=0)
            else:
                nccl_id = self._base_comm.bcast(None, root=0)

            # Initialize NCCL communicator
            nccl_comm = cp.cuda.nccl.NcclCommunicator(
                ndev=self.size,
                commId=nccl_id,
                rank=self.rank,
            )

            self.barrier()

            return nccl_comm
        else:
            # No NCCL communicator needed
            return None

    def _resolve_and_validate_collective_config(
        self, config: CommunicatorConfig
    ) -> CollectiveConfig:
        """
        Resolve 'default' collective comm libs to the best available option and validate availability.

        Parameters
        ----------
        config : CommunicatorConfig
            Configuration object containing collective operation preferences.

        Returns
        -------
        CollectiveConfig
            Resolved configuration with specific communication libraries for each collective.
        """
        collective_config: CollectiveConfig = CollectiveConfig()
        for key in [
            "allreduce",
            "allgather",
            "bcast",
            "reduce",
            "reduce_scatter",
            "send_recv",
            "allgatherv",
            "alltoall",
        ]:
            nccl_supported = key not in [
                "allgatherv",
                "alltoall",
            ]  # NCCL does not support these collectives
            resolved_lib = self._resolve_and_validate_comm_lib(
                comm_lib=config.__dict__.get(key, "default"),
                nccl_supported=nccl_supported,
            )
            collective_config.__dict__[key] = resolved_lib
        return collective_config

    def _get_dtype_factor(self, arr: Any) -> int:
        """
        Get the factor for complex vs real arrays (complex arrays need factor of 2).

        Parameters
        ----------
        arr : array_like
            Input array to check data type.

        Returns
        -------
        int
            Factor of 2 for complex arrays, 1 for real arrays.
        """
        return 2 if np.iscomplexobj(arr) else 1

    # Collectives
    def allreduce(self, arr: Any, op: str) -> None:
        """
        Perform an all-reduce operation on the input array.

        All processes contribute data and all receive the reduced result.
        The operation is performed element-wise across all processes.

        Parameters
        ----------
        arr : array_like
            Input array to be reduced. Must be the same shape and type on all processes.
        op : str
            Reduction operation. Must be one of 'sum', 'prod', 'min', 'max'.

        Notes
        -----
        The input array is modified in-place with the reduction result.
        For complex arrays, the factor of 2 is automatically handled.

        Examples
        --------
        >>> import numpy as np
        >>> from dalia.communicator import Communicator
        >>> comm = Communicator()
        >>> data = np.array([comm.rank])
        >>> comm.allreduce(data, 'sum')
        >>> print(data)  # Sum of all ranks
        """

        def _get_allreduce_parameters(arr):
            factor = self._get_dtype_factor(arr)
            count = arr.size * factor
            return count

        count = _get_allreduce_parameters(arr)

        self.barrier()

        if self._collective_config.allreduce == "host_mpi":
            comm_arr = arr if arr.__module__ == "numpy" else arr.get()
            self._base_comm.Allreduce(
                sendbuf=MPI.IN_PLACE,
                recvbuf=comm_arr,
                op=mpi_op[op],
            )
            arr = comm_arr if arr.__module__ == "numpy" else cp.asarray(arr)
        elif self._collective_config.allreduce == "device_mpi":
            self._base_comm.Allreduce(
                sendbuf=MPI.IN_PLACE,
                recvbuf=arr,
                op=mpi_op[op],
            )
        elif self._collective_config.allreduce == "nccl":
            datatype = nccl_datatype[arr.dtype.type]
            self._xccl_comm.allReduce(
                sendbuf=arr.data.ptr,
                recvbuf=arr.data.ptr,
                count=count,
                datatype=datatype,
                op=nccl_op[op],
                stream=cp.cuda.Stream.null.ptr,
            )
        elif self._collective_config.allreduce == "none":
            pass

        self.barrier()

    def allgather(self, arr: Any) -> None:
        """
        Gather data from all processes and distribute to all processes.

        Each process contributes data and all processes receive the concatenated
        result from all processes.

        Parameters
        ----------
        arr : array_like
            Input array to be gathered. Must be the same size on all processes.

        Notes
        -----
        The input array is modified in-place to contain the gathered data.
        The array size must be divisible by the number of processes.

        Examples
        --------
        >>> import numpy as np
        >>> from dalia.communicator import Communicator
        >>> comm = Communicator()
        >>> data = np.full(4, comm.rank)  # Array filled with rank
        >>> comm.allgather(data)
        >>> print(data)  # Contains data from all processes
        """

        def _get_allgather_parameters(arr):
            factor = self._get_dtype_factor(arr)
            count = (arr.size // self.size) * factor
            displacement = count * self.rank * (arr.dtype.itemsize // factor)
            return count, displacement

        count, displacement = _get_allgather_parameters(arr)

        self.barrier()

        if self._collective_config.allgather == "host_mpi":
            comm_arr = arr if arr.__module__ == "numpy" else arr.get()
            self._base_comm.Allgather(
                sendbuf=comm_arr,
                recvbuf=comm_arr,
            )
            arr = comm_arr if arr.__module__ == "numpy" else cp.asarray(arr)
        elif self._collective_config.allgather == "device_mpi":
            self._base_comm.Allgather(
                sendbuf=arr,
                recvbuf=arr,
            )
        elif self._collective_config.allgather == "nccl":
            datatype = nccl_datatype[arr.dtype.type]
            self._xccl_comm.allGather(
                sendbuf=arr.data.ptr + displacement,
                recvbuf=arr.data.ptr,
                count=count,
                datatype=datatype,
                stream=cp.cuda.Stream.null.ptr,
            )
        elif self._collective_config.allgather == "none":
            pass

        self.barrier()

    def allgatherv(
        self,
        sendbuf: Any,
        recvbuf: Any,
        recvcounts: List[int],
        displs: Optional[List[int]] = None,
    ) -> None:
        """
        Variable all-gather operation with different amounts of data per process.

        Gather data from all processes where each process can contribute different
        amounts of data. All processes receive the complete gathered data.

        Parameters
        ----------
        sendbuf : array_like or MPI.IN_PLACE
            Input data to be gathered. If MPI.IN_PLACE, recvbuf is used for both
            send and receive operations.
        recvbuf : array_like
            Output buffer to store gathered data from all processes.
        recvcounts : list of int
            Number of elements to receive from each process.
        displs : list of int, optional
            Displacements in recvbuf where data from each process should be placed.
            If None, calculated automatically based on recvcounts.

        Notes
        -----
        Supports in-place operations when sendbuf is MPI.IN_PLACE.
        Complex number data types are automatically handled.

        Examples
        --------
        >>> import numpy as np
        >>> from dalia.communicator import Communicator
        >>> comm = Communicator()
        >>> send_data = np.array([comm.rank] * (comm.rank + 1))
        >>> recvcounts = [i + 1 for i in range(comm.size)]
        >>> total_size = sum(recvcounts)
        >>> recv_data = np.zeros(total_size)
        >>> comm.allgatherv(send_data, recv_data, recvcounts)
        """

        def _get_allgatherv_parameters(sendbuf, recvbuf, recvcounts, displs):
            factor = self._get_dtype_factor(
                recvbuf if sendbuf is MPI.IN_PLACE else sendbuf
            )

            # Apply factor for complex numbers to recvcounts
            adjusted_recvcounts = [count * factor for count in recvcounts]

            # Calculate displacements if not provided
            if displs is None:
                displs = [0]
                for i in range(1, len(adjusted_recvcounts)):
                    displs.append(displs[-1] + adjusted_recvcounts[i - 1])
            else:
                # Apply factor to provided displacements
                displs = [disp * factor for disp in displs]

            return adjusted_recvcounts, displs

        adjusted_recvcounts, adjusted_displs = _get_allgatherv_parameters(
            sendbuf, recvbuf, recvcounts, displs
        )

        self.barrier()

        if self._collective_config.allgatherv == "host_mpi":
            if sendbuf is MPI.IN_PLACE:
                comm_sendbuf = MPI.IN_PLACE
            else:
                comm_sendbuf = (
                    sendbuf if sendbuf.__module__ == "numpy" else sendbuf.get()
                )
            comm_recvbuf = recvbuf if recvbuf.__module__ == "numpy" else recvbuf.get()
            # Convert back to original counts (without factor) for MPI
            mpi_recvcounts = [
                count
                // self._get_dtype_factor(
                    recvbuf if sendbuf is MPI.IN_PLACE else sendbuf
                )
                for count in adjusted_recvcounts
            ]
            mpi_displs = [
                disp
                // self._get_dtype_factor(
                    recvbuf if sendbuf is MPI.IN_PLACE else sendbuf
                )
                for disp in adjusted_displs
            ]
            self._base_comm.Allgatherv(
                sendbuf=comm_sendbuf,
                recvbuf=[comm_recvbuf, mpi_recvcounts, mpi_displs],
            )
            if recvbuf.__module__ != "numpy":
                recvbuf[:] = cp.asarray(comm_recvbuf)
        elif self._collective_config.allgatherv == "device_mpi":
            # Convert back to original counts (without factor) for MPI
            mpi_recvcounts = [
                count
                // self._get_dtype_factor(
                    recvbuf if sendbuf is MPI.IN_PLACE else sendbuf
                )
                for count in adjusted_recvcounts
            ]
            mpi_displs = [
                disp
                // self._get_dtype_factor(
                    recvbuf if sendbuf is MPI.IN_PLACE else sendbuf
                )
                for disp in adjusted_displs
            ]
            self._base_comm.Allgatherv(
                sendbuf=sendbuf,
                recvbuf=[recvbuf, mpi_recvcounts, mpi_displs],
            )
        elif self._collective_config.allgatherv == "none":
            # In single process mode, just copy sendbuf to recvbuf (unless in-place)
            if sendbuf is not MPI.IN_PLACE:
                recvbuf[: sendbuf.size] = sendbuf[:]

        self.barrier()

    def alltoall(
        self,
        sendbuf: Any,
        recvbuf: Any,
        sendcounts: Optional[List[int]] = None,
        recvcounts: Optional[List[int]] = None,
        sdispls: Optional[List[int]] = None,
        rdispls: Optional[List[int]] = None,
    ) -> None:
        """
        All-to-all scatter/gather operation.

        Each process sends different data to each process and receives different
        data from each process. This is a complete exchange of data between all
        processes.

        Parameters
        ----------
        sendbuf : array_like or MPI.IN_PLACE
            Input data to be sent. If MPI.IN_PLACE, recvbuf is used for both
            send and receive operations.
        recvbuf : array_like
            Output buffer to store received data.
        sendcounts : list of int, optional
            Number of elements to send to each process. If None, equal partitioning
            is used based on sendbuf size.
        recvcounts : list of int, optional
            Number of elements to receive from each process. If None, equal
            partitioning is used based on recvbuf size.
        sdispls : list of int, optional
            Send displacements in sendbuf. If None, calculated automatically.
        rdispls : list of int, optional
            Receive displacements in recvbuf. If None, calculated automatically.

        Notes
        -----
        Supports in-place operations when sendbuf is MPI.IN_PLACE.
        Complex number data types are automatically handled.

        Examples
        --------
        >>> import numpy as np
        >>> from dalia.communicator import Communicator
        >>> comm = Communicator()
        >>> send_data = np.arange(comm.size * 2) + comm.rank * 10
        >>> recv_data = np.zeros(comm.size * 2)
        >>> comm.alltoall(send_data, recv_data)
        """

        def _get_alltoall_parameters(
            sendbuf, recvbuf, sendcounts, recvcounts, sdispls, rdispls
        ):
            # Use recvbuf for dtype factor if sendbuf is MPI.IN_PLACE
            factor = self._get_dtype_factor(
                recvbuf if sendbuf is MPI.IN_PLACE else sendbuf
            )

            # Handle sendcounts and recvcounts
            if sendcounts is None:
                if sendbuf is MPI.IN_PLACE:
                    # For in-place operation, use recvbuf size
                    sendcount_per_proc = (recvbuf.size // self.size) * factor
                else:
                    # Equal partitioning for send: each process sends sendbuf.size // self.size elements
                    sendcount_per_proc = (sendbuf.size // self.size) * factor
                sendcounts = [sendcount_per_proc] * self.size
            else:
                # Apply factor for complex numbers
                sendcounts = [count * factor for count in sendcounts]

            if recvcounts is None:
                # Equal partitioning for receive: each process receives recvbuf.size // self.size elements
                recvcount_per_proc = (recvbuf.size // self.size) * factor
                recvcounts = [recvcount_per_proc] * self.size
            else:
                # Apply factor for complex numbers
                recvcounts = [count * factor for count in recvcounts]

            # Handle displacements
            if sdispls is None:
                sdispls = [0]
                for i in range(1, len(sendcounts)):
                    sdispls.append(sdispls[-1] + sendcounts[i - 1])
            else:
                # Apply factor to provided displacements
                sdispls = [disp * factor for disp in sdispls]

            if rdispls is None:
                rdispls = [0]
                for i in range(1, len(recvcounts)):
                    rdispls.append(rdispls[-1] + recvcounts[i - 1])
            else:
                # Apply factor to provided displacements
                rdispls = [disp * factor for disp in rdispls]

            return sendcounts, recvcounts, sdispls, rdispls

        adjusted_sendcounts, adjusted_recvcounts, adjusted_sdispls, adjusted_rdispls = (
            _get_alltoall_parameters(
                sendbuf, recvbuf, sendcounts, recvcounts, sdispls, rdispls
            )
        )

        self.barrier()

        if self._collective_config.alltoall == "host_mpi":
            if sendbuf is MPI.IN_PLACE:
                comm_sendbuf = MPI.IN_PLACE
            else:
                comm_sendbuf = (
                    sendbuf if sendbuf.__module__ == "numpy" else sendbuf.get()
                )
            comm_recvbuf = recvbuf if recvbuf.__module__ == "numpy" else recvbuf.get()
            # Convert back to original counts (without factor) for MPI
            factor_div = self._get_dtype_factor(
                recvbuf if sendbuf is MPI.IN_PLACE else sendbuf
            )
            mpi_sendcounts = [count // factor_div for count in adjusted_sendcounts]
            mpi_recvcounts = [count // factor_div for count in adjusted_recvcounts]
            mpi_sdispls = [disp // factor_div for disp in adjusted_sdispls]
            mpi_rdispls = [disp // factor_div for disp in adjusted_rdispls]
            self._base_comm.Alltoall(
                sendbuf=[comm_sendbuf, mpi_sendcounts, mpi_sdispls],
                recvbuf=[comm_recvbuf, mpi_recvcounts, mpi_rdispls],
            )
            if recvbuf.__module__ != "numpy":
                recvbuf[:] = cp.asarray(comm_recvbuf)
        elif self._collective_config.alltoall == "device_mpi":
            # Convert back to original counts (without factor) for MPI
            factor_div = self._get_dtype_factor(
                recvbuf if sendbuf is MPI.IN_PLACE else sendbuf
            )
            mpi_sendcounts = [count // factor_div for count in adjusted_sendcounts]
            mpi_recvcounts = [count // factor_div for count in adjusted_recvcounts]
            mpi_sdispls = [disp // factor_div for disp in adjusted_sdispls]
            mpi_rdispls = [disp // factor_div for disp in adjusted_rdispls]
            self._base_comm.Alltoall(
                sendbuf=[sendbuf, mpi_sendcounts, mpi_sdispls],
                recvbuf=[recvbuf, mpi_recvcounts, mpi_rdispls],
            )
        elif self._collective_config.alltoall == "none":
            # In single process mode, just copy sendbuf to recvbuf (unless in-place)
            if sendbuf is not MPI.IN_PLACE:
                recvbuf[:] = sendbuf[:]

        self.barrier()

    def reduce(self, arr: Any, op: str, root: int = 0) -> None:
        """
        Perform a reduction operation with result only on the root process.

        All processes contribute data, but only the root process receives the
        reduced result.

        Parameters
        ----------
        arr : array_like
            Input array to be reduced. Must be the same shape and type on all processes.
        op : str
            Reduction operation. Must be one of 'sum', 'prod', 'min', 'max'.
        root : int, optional
            Rank of the process that receives the result. Default is 0.

        Notes
        -----
        The input array is modified in-place on the root process only.
        On non-root processes, the array remains unchanged.

        Examples
        --------
        >>> import numpy as np
        >>> from dalia.communicator import Communicator
        >>> comm = Communicator()
        >>> data = np.array([comm.rank + 1])
        >>> comm.reduce(data, 'sum', root=0)
        >>> if comm.rank == 0:
        >>>     print(data)  # Sum of all ranks + 1
        """

        def _get_reduce_parameters(arr):
            factor = self._get_dtype_factor(arr)
            count = arr.size * factor
            return count

        count = _get_reduce_parameters(arr)

        self.barrier()

        if self._collective_config.reduce == "host_mpi":
            comm_arr = arr if arr.__module__ == "numpy" else arr.get()
            self._base_comm.Reduce(
                sendbuf=MPI.IN_PLACE,
                recvbuf=comm_arr,
                op=mpi_op[op],
                root=root,
            )
            arr = comm_arr if arr.__module__ == "numpy" else cp.asarray(arr)
        elif self._collective_config.reduce == "device_mpi":
            self._base_comm.Reduce(
                sendbuf=MPI.IN_PLACE,
                recvbuf=arr,
                op=mpi_op[op],
                root=root,
            )
        elif self._collective_config.reduce == "nccl":
            datatype = nccl_datatype[arr.dtype.type]
            self._xccl_comm.reduce(
                sendbuf=arr.data.ptr,
                recvbuf=arr.data.ptr,
                count=count,
                datatype=datatype,
                op=nccl_op[op],
                root=root,
                stream=cp.cuda.Stream.null.ptr,
            )
        elif self._collective_config.reduce == "none":
            pass

        self.barrier()

    def reduce_scatter(
        self,
        sendbuf: Any,
        recvbuf: Any,
        op: str,
        recvcounts: Optional[List[int]] = None,
    ) -> None:
        """
        Perform a reduction operation followed by scattering the result.

        All processes contribute data, the data is reduced, and then scattered
        such that each process receives a portion of the reduced result.

        Parameters
        ----------
        sendbuf : array_like
            Input data to be reduced and scattered.
        recvbuf : array_like
            Output buffer to store the scattered portion of the reduced result.
        op : str
            Reduction operation. Must be one of 'sum', 'prod', 'min', 'max'.
        recvcounts : list of int, optional
            Number of elements each process should receive. If None, equal
            partitioning is used.

        Notes
        -----
        The total size of sendbuf should equal the sum of recvcounts across all processes.
        Complex number data types are automatically handled.

        Examples
        --------
        >>> import numpy as np
        >>> from dalia.communicator import Communicator
        >>> comm = Communicator()
        >>> send_data = np.ones(comm.size) * (comm.rank + 1)
        >>> recv_data = np.zeros(1)
        >>> comm.reduce_scatter(send_data, recv_data, 'sum')
        >>> print(recv_data)  # Portion of reduced result
        """

        def _get_reduce_scatter_parameters(sendbuf, recvcounts):
            factor = self._get_dtype_factor(sendbuf)
            if recvcounts is None:
                # Equal partitioning: each process receives sendbuf.size // self.size elements
                recvcounts = [(sendbuf.size // self.size) * factor] * self.size
            else:
                # Use provided recvcounts, but apply factor for complex numbers
                recvcounts = [count * factor for count in recvcounts]

            return recvcounts

        recvcounts = _get_reduce_scatter_parameters(sendbuf, recvcounts)

        self.barrier()

        if self._collective_config.reduce_scatter == "host_mpi":
            comm_sendbuf = sendbuf if sendbuf.__module__ == "numpy" else sendbuf.get()
            comm_recvbuf = recvbuf if recvbuf.__module__ == "numpy" else recvbuf.get()
            # Convert back to original counts (without factor) for MPI
            mpi_recvcounts = [
                count // self._get_dtype_factor(sendbuf) for count in recvcounts
            ]
            self._base_comm.Reduce_scatter(
                sendbuf=comm_sendbuf,
                recvbuf=comm_recvbuf,
                recvcounts=mpi_recvcounts,
                op=mpi_op[op],
            )
            if recvbuf.__module__ != "numpy":
                recvbuf[:] = cp.asarray(comm_recvbuf)
        elif self._collective_config.reduce_scatter == "device_mpi":
            # Convert back to original counts (without factor) for MPI
            mpi_recvcounts = [
                count // self._get_dtype_factor(sendbuf) for count in recvcounts
            ]
            self._base_comm.Reduce_scatter(
                sendbuf=sendbuf,
                recvbuf=recvbuf,
                recvcounts=mpi_recvcounts,
                op=mpi_op[op],
            )
        elif self._collective_config.reduce_scatter == "nccl":
            datatype = nccl_datatype[sendbuf.dtype.type]
            self._xccl_comm.reduceScatter(
                sendbuf=sendbuf.data.ptr,
                recvbuf=recvbuf.data.ptr,
                recvcount=recvcounts[self.rank],
                datatype=datatype,
                op=nccl_op[op],
                stream=cp.cuda.Stream.null.ptr,
            )
        elif self._collective_config.reduce_scatter == "none":
            # In single process mode, just copy sendbuf to recvbuf
            recvbuf[:] = sendbuf[:]

        self.barrier()

    def bcast(self, arr: Any, root: int = 0) -> None:
        """
        Broadcast data from the root process to all other processes.

        The root process sends its data to all other processes. After the operation,
        all processes have a copy of the root's data.

        Parameters
        ----------
        arr : array_like
            Data to be broadcast. On the root process, this contains the data to send.
            On other processes, this buffer will be overwritten with the broadcast data.
        root : int, optional
            Rank of the process that sends the data. Default is 0.

        Notes
        -----
        The array is modified in-place on all non-root processes.
        The array must have the same size and type on all processes.

        Examples
        --------
        >>> import numpy as np
        >>> from dalia.communicator import Communicator
        >>> comm = Communicator()
        >>> if comm.rank == 0:
        >>>     data = np.array([1, 2, 3, 4])
        >>> else:
        >>>     data = np.zeros(4)
        >>> comm.bcast(data, root=0)
        >>> print(data)  # [1, 2, 3, 4] on all processes
        """

        def _get_bcast_parameters(arr):
            factor = self._get_dtype_factor(arr)
            count = arr.size * factor
            return count

        count = _get_bcast_parameters(arr)

        self.barrier()

        if self._collective_config.allgather == "host_mpi":
            comm_arr = arr if arr.__module__ == "numpy" else arr.get()
            self._base_comm.Bcast(
                buf=comm_arr,
                root=root,
            )
            arr = comm_arr if arr.__module__ == "numpy" else cp.asarray(arr)
        elif self._collective_config.allgather == "device_mpi":
            self._base_comm.Bcast(
                buf=arr,
                root=root,
            )
        elif self._collective_config.allgather == "nccl":
            datatype = nccl_datatype[arr.dtype.type]
            self._xccl_comm.bcast(
                buff=arr.data.ptr,
                count=count,
                datatype=datatype,
                root=root,
                stream=cp.cuda.Stream.null.ptr,
            )
        elif self._collective_config.allgather == "none":
            pass

        self.barrier()

    def send(self, arr: Any, dest: int, tag: int) -> None:
        """
        Send data to a specific process.

        Performs a blocking send operation to transfer data from the current
        process to the destination process.

        Parameters
        ----------
        arr : array_like
            Data to be sent.
        dest : int
            Rank of the destination process.
        tag : int
            Message tag for matching with corresponding receive operation.

        Notes
        -----
        This is a blocking operation that completes when the data has been sent.
        The corresponding process must call recv() with matching source and tag.

        Examples
        --------
        >>> import numpy as np
        >>> from dalia.communicator import Communicator
        >>> comm = Communicator()
        >>> if comm.rank == 0:
        >>>     data = np.array([1, 2, 3])
        >>>     comm.send(data, dest=1, tag=42)
        """

        def _get_send_parameters(arr):
            factor = self._get_dtype_factor(arr)
            count = arr.size * factor
            return count

        count = _get_send_parameters(arr)

        self.barrier()

        if self._collective_config.send_recv == "host_mpi":
            comm_arr = arr if arr.__module__ == "numpy" else arr.get()
            self._base_comm.Send(
                buf=comm_arr,
                dest=dest,
                tag=tag,
            )
            arr = comm_arr if arr.__module__ == "numpy" else cp.asarray(arr)
        elif self._collective_config.send_recv == "device_mpi":
            self._base_comm.Send(
                buf=arr,
                dest=dest,
                tag=tag,
            )
        elif self._collective_config.send_recv == "nccl":
            datatype = nccl_datatype[arr.dtype.type]
            self._xccl_comm.send(
                sendbuf=arr.data.ptr,
                count=count,
                datatype=datatype,
                peer=dest,
                stream=cp.cuda.Stream.null.ptr,
            )
        elif self._collective_config.send_recv == "none":
            pass

        self.barrier()

    def recv(self, arr: Any, source: int, tag: int) -> None:
        """
        Receive data from a specific process.

        Performs a blocking receive operation to get data from the source
        process into the provided buffer.

        Parameters
        ----------
        arr : array_like
            Buffer to store received data. Must be pre-allocated with correct size.
        source : int
            Rank of the source process.
        tag : int
            Message tag for matching with corresponding send operation.

        Notes
        -----
        This is a blocking operation that completes when data is received.
        The buffer is modified in-place with the received data.
        The source process must call send() with matching destination and tag.

        Examples
        --------
        >>> import numpy as np
        >>> from dalia.communicator import Communicator
        >>> comm = Communicator()
        >>> if comm.rank == 1:
        >>>     data = np.zeros(3)
        >>>     comm.recv(data, source=0, tag=42)
        >>>     print(data)  # Received data from process 0
        """

        def _get_recv_parameters(arr):
            factor = self._get_dtype_factor(arr)
            count = arr.size * factor
            return count

        count = _get_recv_parameters(arr)

        self.barrier()

        if self._collective_config.send_recv == "host_mpi":
            comm_arr = arr if arr.__module__ == "numpy" else arr.get()
            self._base_comm.Recv(
                buf=comm_arr,
                source=source,
                tag=tag,
            )
            arr = comm_arr if arr.__module__ == "numpy" else cp.asarray(arr)
        elif self._collective_config.send_recv == "device_mpi":
            self._base_comm.Recv(
                buf=arr,
                source=source,
                tag=tag,
            )
        elif self._collective_config.send_recv == "nccl":
            datatype = nccl_datatype[arr.dtype.type]
            self._xccl_comm.recv(
                recvbuf=arr.data.ptr,
                count=count,
                datatype=datatype,
                peer=source,
                stream=cp.cuda.Stream.null.ptr,
            )
        elif self._collective_config.send_recv == "none":
            pass

        self.barrier()

    # Utilities
    def barrier(self, sync_gpu: bool = True) -> None:
        """
        Synchronize all processes in the communicator.

        Blocks until all processes in the communicator have reached this point.
        Optionally also synchronizes GPU operations.

        Parameters
        ----------
        sync_gpu : bool, optional
            Whether to also synchronize GPU operations using CUDA streams.
            Default is True.

        Notes
        -----
        This is a collective operation - all processes must call barrier().
        When sync_gpu is True and CuPy is available, also waits for GPU operations
        to complete.

        Examples
        --------
        >>> from dalia.communicator import Communicator
        >>> comm = Communicator()
        >>> # Do some work
        >>> comm.barrier()  # Wait for all processes
        >>> # Continue with synchronized execution
        """
        if self._base_comm is not None:
            self._base_comm.Barrier()
        if sync_gpu and backend_flags["cupy_avail"]:
            cp.cuda.Stream.null.synchronize()

    def split(self) -> None:
        """
        Split the communicator into multiple sub-communicators.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError("split method not yet implemented")

    @contextmanager
    def time(self) -> Generator[Any, None, None]:
        """
        Context manager for timing code blocks with process synchronization.

        Provides synchronized timing across all processes by adding barriers
        before and after the timed code block.

        Yields
        ------
        ElapsedTime
            Object with a 'value' attribute containing elapsed time in seconds.
            The value is set when the context manager exits.

        Examples
        --------
        >>> from dalia.communicator import Communicator
        >>> comm = Communicator()
        >>> with comm.time() as elapsed_time:
        ...     # Your code here
        ...     result = expensive_computation()
        >>> print(f"Elapsed time: {elapsed_time.value} seconds")

        Or simply:
        >>> with comm.time():
        ...     # Your code here
        ...     result = expensive_computation()
        """
        self.barrier(sync_gpu=True)

        class ElapsedTime:
            def __init__(self):
                self.value = None

        elapsed_time = ElapsedTime()
        tic = time.perf_counter()

        try:
            yield elapsed_time
        finally:
            self.barrier(sync_gpu=True)
            toc = time.perf_counter()
            elapsed_time.value = toc - tic

    def timed(self, print_result: bool = True, label: Optional[str] = None) -> Callable:
        """
        Decorator for timing function execution with process synchronization.

        Provides synchronized timing across all processes by adding barriers
        before and after the decorated function execution.

        Parameters
        ----------
        print_result : bool, optional
            Whether to print the elapsed time automatically. Default is True.
        label : str, optional
            Optional label to include in the printed output. If None, uses
            the function name.

        Returns
        -------
        callable
            Decorated function with timing capabilities.

        Notes
        -----
        The elapsed time is stored as an attribute 'elapsed_time' on the wrapper
        function and can be accessed after the function call.

        Examples
        --------
        >>> from dalia.communicator import Communicator
        >>> comm = Communicator()
        >>> @comm.timed()
        ... def compute_something():
        ...     return expensive_operation()

        >>> @comm.timed(label="Cholesky decomposition")
        ... def cholesky_decomp(A):
        ...     return cholesky(A)

        >>> @comm.timed(print_result=False)
        ... def silent_function():
        ...     # This won't print timing, but you can access it via function.elapsed_time
        ...     return result
        >>> result = silent_function()
        >>> print(f"Silent function took {silent_function.elapsed_time} seconds")
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.barrier(sync_gpu=True)
                tic = time.perf_counter()

                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.barrier(sync_gpu=True)
                    toc = time.perf_counter()
                    elapsed = toc - tic

                    # Store elapsed time as an attribute of the wrapper function
                    wrapper.elapsed_time = elapsed

                    if print_result:
                        if label:
                            print(f"{label}: {elapsed:.6f} seconds")
                        else:
                            print(f"{func.__name__}: {elapsed:.6f} seconds")

            return wrapper

        return decorator
