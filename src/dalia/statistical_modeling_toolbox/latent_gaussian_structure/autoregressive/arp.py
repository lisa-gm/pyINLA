from .autoregressive import Autoregressive


class ARP(Autoregressive):
    """Class for autoregressive of order p latent structures."""

    def __init__(self, p, ...):
        super().__init__()
        self.p = p
        # ...