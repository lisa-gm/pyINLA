# src/backend/datastructures/matrix/dispatch/operations.py
from enum import Enum


class Operation(Enum):
    MATMUL = "matmul"
    ADD = "add"
