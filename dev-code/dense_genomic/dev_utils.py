def exit_as_expected():
    """Exit the program with a message indicating that the program has completed successfully."""
    print("Program completed successfully.")
    exit(0)

def matshow_matrices(matrices: list, titles: list = None):
    """Display a list of matrices using matplotlib's matshow."""
    import matplotlib.pyplot as plt

    num_matrices = len(matrices)
    fig, axes = plt.subplots(1, num_matrices, figsize=(5 * num_matrices, 5))

    if titles is None:
        titles = [f"Matrix {i+1}" for i in range(num_matrices)]

    for ax, matrix, title in zip(axes, matrices, titles):
        cax = ax.matshow(matrix)
        ax.set_title(title)
        fig.colorbar(cax, ax=ax)

    plt.show()