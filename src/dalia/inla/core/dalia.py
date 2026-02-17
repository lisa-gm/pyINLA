# src/dalia/inla/core/dalia.py
# general orchestrator


# from dalia.inla.configs.configs import Config


class Config:
    # Checkpointing for restart durign the optimization
    # ...
    pass


class DALIA:
    """
    ...
    """

    # 1. Class attributes (if any)
    # 2. Initialization

    def __init__(self, model: StatisticalModel, config: Config = Config()) -> None:
        # self.model = model
        # self.config = config
        pass

    def __new__(cls):
        pass

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"<DALIA instance at {hex(id(self))}>"

    # 3. Special representation methods
    # 4. Properties (grouped together)
    # 5. Comparison operators (if needed)
    # 6. Arithmetic operators (standard order)
    # 7. Right-hand operators (same order as above)
    # 8. In-place operators (if supported)
    # 9. Other special methods
    # 10. Public methods

    def learn_from_observations(self, observations: DataModel) -> None:
        # assert utils.validate_observation(self.model, observations)
        # self.observations = observations.validate()
        pass

    # 11. Private/protected methods (start with _)

    def _checkpoint(self) -> None:
        pass
