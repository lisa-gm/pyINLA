
from math import inf

from scipy.optimize import Bounds

class Hyperparameter:
    name : str
    value : float

    # Wether or not this hyperparameter is fixed or can be optimized
    is_fixed : bool = False

    # Hyperparameter bounds for optimization
    bounds : Bounds = Bounds(lb=-inf, ub=inf)

class HyperparameterManagerConfig:
    ...

class HyperparameterManager:
    def __init__(self, hyperparameters: List[Hyperparameter], config: HyperparameterManagerConfig):
        ...
 
