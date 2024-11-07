# Copyright 2024 pyINLA authors. All rights reserved.

import os


def get_host_configuration() -> dict:
    """
    Get host configuration.

    Returns
    -------
    host_configuration: dict.
        A dictionary containing:
            - num_threads: str. The number of threads.
            - host_id: str. The host id.
    """
    host_id = os.getenv("HOSTNAME", os.getenv("COMPUTERNAME"))
    num_threads = os.getenv("OMP_NUM_THREADS")

    return {"num_threads": num_threads, "host_id": host_id}
