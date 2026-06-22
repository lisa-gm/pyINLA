# src/dalia/backend/datastructures/matrix/dispatch/operations.py
from enum import Enum


class Operation(Enum):
    MUL = "mul"
    MATMUL = "matmul"
    ADD = "add"
    SUB = "sub"
