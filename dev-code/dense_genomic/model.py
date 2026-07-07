from abc import ABC, abstractmethod


class StatisticalModel(ABC):
    def __init__(self):
        pass

class GenomicModel(StatisticalModel):
    def __init__(self):
        super().__init__()
