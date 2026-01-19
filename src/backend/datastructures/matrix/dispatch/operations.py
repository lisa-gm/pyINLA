from enum import Enum


class Operation(Enum):
    MATMUL = "matmul"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    TRUEDIV = "truediv"
    # etc.
