# Copyright 2024-2025 pyINLA authors. All rights reserved.


def add_str_header(
    title: str,
    table: str,
):
    """Add a header to a table."""
    # Add the header title
    table_width = max(len(line) for line in table.split("\n"))
    title_width = len(title)
    total_width = max(table_width, title_width)
    title_centered = title.center(total_width)
    table = f"{title_centered}\n{table}"

    return table


def align_tables_side_by_side(tables, padding=2):
    """
    tables: list of multiline strings (tables)
    padding: number of spaces between tables
    """
    # Split each table into lines
    lines_list = [table.splitlines() for table in tables]
    # Compute max number of lines
    max_lines = max(len(lines) for lines in lines_list)
    # Compute max width for each table
    table_widths = [max(len(line) for line in lines) for lines in lines_list]
    # Pad each table's lines to its own width, and pad with empty lines at the bottom if needed
    for idx, lines in enumerate(lines_list):
        width = table_widths[idx]
        # Pad each line to the table's width
        lines_list[idx] = [line.ljust(width) for line in lines]
        # Pad with empty lines if necessary
        if len(lines_list[idx]) < max_lines:
            lines_list[idx].extend([" " * width] * (max_lines - len(lines_list[idx])))
    # Concatenate corresponding lines with padding
    result_lines = []
    for row in zip(*lines_list):
        result_lines.append((" " * padding).join(row))
    return "\n".join(result_lines)


def boxify(text: str, padding: int = 1) -> str:
    """Wraps the given text in a Unicode box with optional horizontal padding."""
    lines = text.splitlines()
    pad = " " * padding
    content_width = max(len(line) for line in lines)
    box_width = content_width + 2 * padding

    top = "╔" + "═" * box_width + "╗"
    bottom = "╚" + "═" * box_width + "╝"
    boxed_lines = [top]
    for line in lines:
        # Pad line to content_width, then add side padding
        boxed_line = f"║{pad}{line.ljust(content_width)}{pad}║"
        boxed_lines.append(boxed_line)
    boxed_lines.append(bottom)
    return "\n".join(boxed_lines)


def ascii_logo():
    logo = r"""
___________________________________ 
___  __ \__    |__  /____  _/__    |
__  / / /_  /| |_  /  __  / __  /| |
_  /_/ /_  ___ |  /____/ /  _  ___ |
/_____/ /_/  |_/_____/___/  /_/  |_|
"""
    return logo
