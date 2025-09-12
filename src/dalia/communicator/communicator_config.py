# Copyright 2024-2025 DALIA authors. All rights reserved.

import tomllib
from pathlib import Path
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict


class CommunicatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tag: str = "generic"

    # Communication library to use
    # - "default" will use the best available option (nccl > device_mpi > host_mpi > none)
    # - "none" disables communication (single process mode)
    comm_lib: Literal["default", "host_mpi", "device_mpi", "nccl", "none"] = "default"

    # Specific Communication Module to use per Collective
    # - "default" will use the given comm_lib
    allreduce: Literal["default", "host_mpi", "device_mpi", "nccl", "none"] = "default"
    allgather: Literal["default", "host_mpi", "device_mpi", "nccl", "none"] = "default"
    bcast: Literal["default", "host_mpi", "device_mpi", "nccl", "none"] = "default"
    reduce: Literal["default", "host_mpi", "device_mpi", "nccl", "none"] = "default"
    reduce_scatter: Literal["default", "host_mpi", "device_mpi", "nccl", "none"] = (
        "default"
    )
    send_recv: Literal["default", "host_mpi", "device_mpi", "nccl", "none"] = "default"

    # Collectives that are not supported by NCCL
    allgatherv: Literal["default", "host_mpi", "device_mpi", "none"] = "default"
    alltoall: Literal["default", "host_mpi", "device_mpi", "none"] = "default"


class FCommunicatorConfig(CommunicatorConfig):
    """
    Configuration for the communicator using the default backend.
    This is a placeholder for future extensions or specific configurations.
    """

    model_config = ConfigDict(extra="forbid")

    tag: str = "feval"


class QCommunicatorConfig(CommunicatorConfig):
    """
    Configuration for the communicator using the default backend.
    This is a placeholder for future extensions or specific configurations.
    """

    model_config = ConfigDict(extra="forbid")

    tag: str = "qeval"


class SCommunicatorConfig(CommunicatorConfig):
    """
    Configuration for the communicator using the default backend.
    This is a placeholder for future extensions or specific configurations.
    """

    model_config = ConfigDict(extra="forbid")

    tag: str = "seval"


class DALIACommunicatorConfig(BaseModel):
    """Container for all communicator configurations."""

    model_config = ConfigDict(extra="forbid")

    base: CommunicatorConfig
    feval: FCommunicatorConfig
    qeval: QCommunicatorConfig
    solver: SCommunicatorConfig


def parse_dalia_communicator_config(
    config: Union[dict, str, Path],
) -> DALIACommunicatorConfig:
    """
    Parse communicator configuration from TOML file or dict.

    If only 'base' or 'communicator' is specified, its values will be used
    as defaults for f, q, and s communicators unless they are explicitly overridden.

    Example TOML:
    ```toml
    [communicator.base]
    comm_lib = "nccl"
    allgather = "device_mpi"

    [communicator.f]
    # Will inherit all from base

    [communicator.q]
    # Will inherit from base, but can override specific fields
    allgather = "host_mpi"
    allreduce = "host_mpi"

    [communicator.s]
    # Will inherit all from base
    ```
    """
    if isinstance(config, (str, Path)):
        with open(config, "rb") as f:
            config_dict = tomllib.load(f)
    else:
        config_dict = config.copy()

    comm_config = config_dict.get("communicator", {})

    # Check if this is a simple config (no nested sections)
    has_nested = any(key in comm_config for key in ["base", "feval", "qeval", "solver"])

    if not has_nested:
        # Simple config - use the same config for all
        base_config = CommunicatorConfig(**comm_config)
        base_dict = base_config.model_dump()

        return DALIACommunicatorConfig(
            base=base_config,
            feval=FCommunicatorConfig(**base_dict),
            qeval=QCommunicatorConfig(**base_dict),
            solver=SCommunicatorConfig(**base_dict),
        )
    else:
        # Advanced config with inheritance
        base_config_dict = comm_config.get("base", {})
        base_config = CommunicatorConfig(**base_config_dict)
        base_dict = base_config.model_dump()

        feval_dict = {**base_dict, **comm_config.get("feval", {})}
        qeval_dict = {**base_dict, **comm_config.get("qeval", {})}
        solver_dict = {**base_dict, **comm_config.get("solver", {})}

        return DALIACommunicatorConfig(
            base=base_config,
            feval=FCommunicatorConfig(**feval_dict),
            qeval=QCommunicatorConfig(**qeval_dict),
            solver=SCommunicatorConfig(**solver_dict),
        )
