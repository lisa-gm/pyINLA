from mpi4py import MPI

comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


# make print function that only prints if MPI rank is 0
def print_mpi(*args, **kwargs):
    if comm_rank == 0:
        print(*args, **kwargs)
