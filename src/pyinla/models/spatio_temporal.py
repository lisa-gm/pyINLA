# Copyright 2024 pyINLA authors. All rights reserved.

from scipy.sparse import load_npz

from pyinla.core.model import Model
from pyinla.core.pyinla_config import PyinlaConfig


class SpatioTemporal(Model):
    """Fit a spatio-temporal model."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
        **kwargs,
    ) -> None:
        """Initializes the model."""
        super().__init__(pyinla_config)

        # Load spatial_matrices
        self.c0 = load_npz(pyinla_config.input_dir / "c0.npz")
        self.g1 = load_npz(pyinla_config.input_dir / "g1.npz")
        self.g2 = load_npz(pyinla_config.input_dir / "g2.npz")
        self.g3 = load_npz(pyinla_config.input_dir / "g3.npz")

        self.ns = self.c0.shape[0]
        self._check_dimensions_spatial_matrices()

        # Load temporal_matrices
        self.m0 = load_npz(pyinla_config.input_dir / "m0.npz")
        self.m1 = load_npz(pyinla_config.input_dir / "m1.npz")
        self.m2 = load_npz(pyinla_config.input_dir / "m2.npz")

        self.nt = self.m0.shape[0]
        self._check_dimensions_temporal_matrices()

        # Check that design_matrix shape match spatio-temporal fields
        assert (
            self.a.shape[1] == self.nb + self.ns + self.nt
        ), "Design matrix has incorrect number of columns."

        # Load model hyperparameters
        self.theta = {
            "spatial_range": pyinla_config.theta_spatial_range,
            "temporal_range": pyinla_config.theta_temporal_range,
            "sd_spatio_temporal": pyinla_config.theta_sd_spatio_temporal,
        }

    def _check_dimensions_spatial_matrices(self) -> None:
        """Check the dimensions of the model."""
        # Spatial matrices checks
        assert self.c0.shape[0] == self.c0.shape[1], "Spatial matrix c0 is not square."
        assert self.g1.shape[0] == self.g1.shape[1], "Spatial matrix g1 is not square."
        assert self.g2.shape[0] == self.g2.shape[1], "Spatial matrix g2 is not square."
        assert self.g3.shape[0] == self.g3.shape[1], "Spatial matrix g3 is not square."
        assert (
            self.c0.shape == self.g1.shape == self.g2.shape == self.g3.shape
        ), "Dimensions of spatial matrices do not match."

    def _check_dimensions_temporal_matrices(self) -> None:
        """Check the dimensions of the model."""
        # Temporal matrices checks
        assert self.m0.shape[0] == self.m0.shape[1], "Temporal matrix m0 is not square."
        assert self.m1.shape[0] == self.m1.shape[1], "Temporal matrix m1 is not square."
        assert self.m2.shape[0] == self.m2.shape[1], "Temporal matrix m2 is not square."
        assert (
            self.m0.shape == self.m1.shape == self.m2.shape
        ), "Dimensions of temporal matrices do not match."
