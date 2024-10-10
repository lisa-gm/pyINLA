import os
import time

from pyinla.core.pyinla_config import parse_config
from pyinla.core.inla import INLA

from pyinla.utils.other_utils import print_mpi

path = os.path.dirname(__file__)

if __name__ == "__main__":
    config = parse_config(f"{path}/config.toml")
    print_mpi("The work directory is:", path)

    pyinla = INLA(config)

    tic = time.perf_counter()
    f_init = pyinla.run()
    toc = time.perf_counter()

    # print("Initial f:", f_init)
    theta_star = pyinla.get_theta_star()
    # print("Optimal theta:", theta_star)

    print_mpi(f"Elapsed time: {toc - tic:.2f} s")
