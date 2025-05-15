import os

import numpy as np

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.realpath(__file__))
    # set up model without running it

    # load output
    x_permuted = np.load(f"{base_dir}/x_permuted.npy")

    # save as dat file
    np.savetxt("x_permuted.dat", x_permuted)

    # check with reference data
