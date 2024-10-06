import os

import numpy as np

from pyinla.core.inla import INLA
from pyinla.core.pyinla_config import parse_config

path = os.path.dirname(__file__)

################# regression model - gaussian likelihood ##################

if __name__ == "__main__":
    config = parse_config(f"{path}/config.toml")
    print("The work directory is:", path)

    pyinla = INLA(config)

    # tic = time.perf_counter()
    f_value = pyinla.run()
    # toc = time.perf_counter()

    # print("Initial f:", f_init)

    theta_final = pyinla.theta
    print("Final theta:", theta_final)

    x_final = pyinla.x
    print("Final x:", x_final)

    os.makedirs(f"{path}/outputs", exist_ok=True)

    np.save(f"{path}/outputs/f_value_pyinla.npy", f_value)
    np.save(f"{path}/outputs/theta_mean_pyinla.npy", theta_final)
    np.save(f"{path}/outputs/x_mean_pyinla.npy", x_final)
