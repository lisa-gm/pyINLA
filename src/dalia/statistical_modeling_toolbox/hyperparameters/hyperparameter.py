# Linear predictor on hyperparameters
# A parameter coud be depending on covariates, it is a function of ()
# Generalization of hp, e.g.: theta = f(t) = theta_0 + theta_1 * t
# theta_o and theta_1 are both optimized in the BFGS, however theta is used in the construction of the latent field


class Hyperparameter:
    def __init__(self, name, value, is_variable=True):
        self.name = name
        self.value = value
        # True if variable, False if constant (do not estimate, it is exactly provided by the user)
        self.is_variable = is_variable

    def __repr__(self):
        var_str = "Variable" if self.is_variable else "Constant"
        return f"Hyperparameter(name={self.name}, value={self.value}, type={var_str})"
