# Copyright 2024 pyINLA authors. All rights reserved.

from pyinla.core.pyinla_config import PyinlaConfig


class INLA:
    """Integrated Nested Laplace Approximation (INLA).

    Parameters
    ----------
    pyinla_config : Path
        pyinla configuration file.

    """

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
    ) -> None:
        self.pyinla_config = pyinla_config

    def run(self) -> None:
        """Fit the model using INLA."""
        pass
