# Copyright 2024 pyINLA authors. All rights reserved.

import pytest

from pyinla.core.pyinla_config import PyinlaConfig

# import numpy as np
# from pyinla import ArrayLike
# from scipy import sparse

# from pyinla.core.model import Model
# from pyinla.models.spatio_temporal import SpatioTemporalModel


@pytest.fixture(scope="function", autouse=False)
def pyinla_config_model():
    pyinla_config = PyinlaConfig()

    pyinla_config.model.theta_spatial_range = 1.0
    pyinla_config.model.theta_temporal_range = 1.0
    pyinla_config.model.theta_spatio_temporal_variation = 1.0

    return pyinla_config
