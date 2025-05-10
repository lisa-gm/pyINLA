def format_size(size_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def summarize_sparse_matrix(matrix, name):
    """Print a summary of a sparse matrix."""
    data = matrix.data  # Access the non-zero entries
    non_zero_count = data.size
    total_count = matrix.shape[0] * matrix.shape[1]
    sparsity = 100 * (1 - non_zero_count / total_count)

    print(f"\n--- Summary of {name} ---")
    print(f"Shape: {matrix.shape}")
    print(f"Sparsity: {sparsity:.6f}%")
    print(f"Non-zero entries: {non_zero_count}")
    print(f"Min: {data.min():.6f}, Max: {data.max():.6f}")
    print(f"Mean: {data.mean():.6f}, Std: {data.std():.6f}")