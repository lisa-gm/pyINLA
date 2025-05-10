import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="PyINLA example parameters")
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum number of iterations in the optimization process.",
    )
    parser.add_argument(
        "--solver_min_p",
        type=int,
        default=1,
        help="Minimum number of processes for the solver. If greater than 1 a distributed solver is used.",
    )
    args = parser.parse_args()
    print("Parsed parameters:")
    print(f"  max_iter: {args.max_iter}")
    print(f"  solver_min_p: {args.solver_min_p}")
    return args